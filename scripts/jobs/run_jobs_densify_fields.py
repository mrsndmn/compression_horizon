"""Densify the progressive-cramming accuracy *fields* along one trajectory, sharded
one-anchor-per-job for maximum cluster parallelism.

The talk figure / animation (``scripts/paper/visual_abstract.py`` /
``animate_trajectory.py``) draws the >0.9-accuracy region at a handful of trajectory
anchors cached in ``landscape_pca_pairs.npz``. To show the field *evolving along the
path* we need many more anchors. Each anchor's field is an independent GPU job
(mesh_resolution^2 teacher-forced forward passes), so this launcher fans the anchor
set out across ``a100.1gpu`` jobs -- by default ONE anchor per job, the finest split
the scheduler can run in parallel.

Each shard re-fits the (deterministic, ``random_state=42``) per-sample PCA on the same
dataset, so its ``coords`` are byte-identical to the cached NPZ; only the anchor and its
accuracy field differ. Anchor indices and per-anchor neighborhood radii are computed
here from the cached coords (no GPU) and passed explicitly, so the merge step
(``scripts/paper/merge_landscape_shards.py``) can stack the shard NPZs into one dense
``landscape_pca_pairs.npz``.

Usage::

    # preview the 8 jobs without launching
    python scripts/jobs/run_jobs_densify_fields.py --dry
    # launch them
    python scripts/jobs/run_jobs_densify_fields.py
"""

import argparse
import json
import os

import numpy as np
from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

PYTHON_PATH = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
AUTHOR_NAME = "d.tarasov"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
JOB_DESC_PREFIX = "CH: densify-fields"
DEFAULT_NPZ = "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/visualizations/landscape_pca_pairs.npz"


def uniform_new_indices(n_traj: int, num_points: int, existing: np.ndarray) -> np.ndarray:
    """``num_points`` interior, evenly-spaced trajectory indices, skipping the endpoints
    (a length-1 prefix field is degenerate) and any already-cached anchor."""
    cand = np.linspace(0, n_traj - 1, num_points + 2).round().astype(int)[1:-1]
    cand = np.array([i for i in cand if i not in set(existing.tolist())], dtype=int)
    return np.unique(cand)


def auto_radius(xy: np.ndarray, idx: int, window: int, factor: float, rmin: float, rmax: float) -> float:
    """Neighborhood radius for an anchor = clip(factor * local PC1-PC2 span, rmin, rmax).

    The local span (max coordinate range over a +/-window stage window) shrinks along the
    path -- the early trajectory is spread out, the late trajectory tightly packed -- which
    reproduces the hand-tuned radius schedule (300 -> 5) used for the cached figure.
    """
    n = xy.shape[0]
    lo, hi = max(0, idx - window), min(n, idx + window + 1)
    win = xy[lo:hi]
    span = float(np.max(win.max(0) - win.min(0))) if win.shape[0] > 1 else 0.0
    return float(np.clip(factor * span, rmin, rmax))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="Print the jobs without launching.")
    ap.add_argument("--npz", type=str, default=DEFAULT_NPZ, help="Cached NPZ to read coords/dataset/model from.")
    ap.add_argument("--num-points", "--num_points", dest="num_points", type=int, default=8)
    ap.add_argument(
        "--step",
        type=int,
        default=None,
        help="Place an anchor every STEP trajectory indices (uniform in index space): "
        "idx STEP, 2*STEP, ... (idx 0 is skipped -- a length-1 prefix field is degenerate). "
        "Overrides --num-points.",
    )
    ap.add_argument(
        "--anchor-indices",
        "--anchor_indices",
        dest="anchor_indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit anchor indices (overrides the uniform --num-points sampling).",
    )
    ap.add_argument("--mesh-resolution", "--mesh_resolution", dest="mesh_resolution", type=int, default=60)
    ap.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    ap.add_argument("--padding", type=float, default=0.25)
    ap.add_argument("--radius-window", dest="radius_window", type=int, default=20)
    ap.add_argument("--radius-factor", dest="radius_factor", type=float, default=3.0)
    ap.add_argument("--radius-min", dest="radius_min", type=float, default=4.0)
    ap.add_argument("--radius-max", dest="radius_max", type=float, default=400.0)
    ap.add_argument(
        "--output-root",
        dest="output_root",
        type=str,
        default=None,
        help="Where shard NPZs are written (default: <npz dir>/field_shards).",
    )
    ap.add_argument(
        "--bundle",
        action="store_true",
        help="Emit ONE multi-frame job computing ALL anchors for this sample (instead of one job "
        "per anchor). The job's output NPZ is already the dense, animation-ready landscape -- no "
        "merge step needed. Best when per-job model-load dominates the field compute.",
    )
    ap.add_argument(
        "--bundle-subdir",
        dest="bundle_subdir",
        type=str,
        default="dense",
        help="Subdir under the NPZ dir where the --bundle job writes its dense NPZ (default: dense).",
    )
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        raise SystemExit(f"cached NPZ not found: {args.npz} (run visualize_landscale_2pca.py first)")
    z = np.load(args.npz, allow_pickle=True)
    coords = z["coords"].astype(np.float64)
    xy = coords[:, :2]
    n_traj = int(coords.shape[0])
    dataset_path = str(z["dataset_path"][0])
    sample_id = int(z["sample_id"][0])
    model_checkpoint = str(z["model_checkpoint"][0])
    existing = z["sampled_indices"].reshape(-1) if "sampled_indices" in z else np.array([], dtype=int)

    if args.anchor_indices is not None:
        indices = np.unique(np.array(args.anchor_indices, dtype=int))
        if indices.min() < 0 or indices.max() >= n_traj:
            raise SystemExit(f"--anchor-indices out of range 0..{n_traj - 1}")
    elif args.step is not None:
        if args.step < 1:
            raise SystemExit("--step must be >= 1")
        cand = np.arange(args.step, n_traj, args.step, dtype=int)  # skip idx 0 (degenerate)
        cand = np.array([i for i in cand if i not in set(existing.tolist())], dtype=int)
        indices = np.unique(cand)
    else:
        indices = uniform_new_indices(n_traj, args.num_points, existing)

    viz_dir = os.path.dirname(args.npz)
    output_root = args.output_root or os.path.join(viz_dir, "field_shards")
    workdir = os.getcwd()

    radii = {
        int(i): auto_radius(xy, int(i), args.radius_window, args.radius_factor, args.radius_min, args.radius_max)
        for i in indices
    }

    print(f"Densifying fields for {len(indices)} new anchors on {model_checkpoint} (sample {sample_id}):")
    print(f"  existing cached anchors: {existing.tolist()}")
    print(f"  new anchors:             {indices.tolist()}")
    for i in indices.tolist():
        print(f"    idx={i:4d}  PC1={xy[i,0]:+8.2f} PC2={xy[i,1]:+7.2f}  radius={radii[i]:.1f}")

    client, extra_options = training_job_api_from_profile("default")
    in_progress = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    def _payload(script: str, job_desc: str) -> dict:
        return {
            "script": script,
            "job_desc": job_desc,
            "env_variables": {"PYTHONPATH": "./src", "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface"},
            "instance_type": "a100.1gpu",
            "region": extra_options["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": BASE_IMAGE,
            "n_workers": 1,
            "processes_per_worker": 1,
        }

    def _maybe_launch(job_desc: str, out_npz: str, script: str) -> bool:
        if os.path.exists(out_npz):
            print(f"\033[33mSkip (exists):\033[0m {out_npz}")
            return False
        if job_desc in in_progress:
            print(f"\033[33mSkip (in queue):\033[0m {job_desc}")
            return False
        print(f"\033[32mLaunch:\033[0m {job_desc}")
        print(f"\033[90m  {script}\033[0m")
        if args.dry:
            return False
        result = client.run_job(payload=_payload(script, job_desc))
        print(f"  -> {result}")
        return True

    manifest = {
        "npz": args.npz,
        "dataset_path": dataset_path,
        "sample_id": sample_id,
        "model_checkpoint": model_checkpoint,
        "mesh_resolution": args.mesh_resolution,
        "mode": "bundle" if args.bundle else "per-anchor",
        "existing_anchors": existing.tolist(),
    }

    launched = 0
    if args.bundle:
        idx_list = indices.tolist()
        rad_list = [radii[i] for i in idx_list]
        bundle_dir = os.path.join(viz_dir, args.bundle_subdir)
        bundle_npz = os.path.join(bundle_dir, "landscape_pca_pairs.npz")
        manifest["bundle"] = {"output_dir": bundle_dir, "npz": bundle_npz, "anchor_indices": idx_list, "radii": rad_list}
        job_desc = f"{JOB_DESC_PREFIX} s{sample_id} bundle{len(idx_list)} #{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"
        cmd_args = [
            f"--dataset_path {dataset_path}",
            f"--sample_id {sample_id}",
            f"--model_checkpoint {model_checkpoint}",
            f"--batch_size {args.batch_size}",
            f"--mesh_resolution {args.mesh_resolution}",
            f"--num-frames {len(idx_list)}",
            f"--anchor_indices {' '.join(str(i) for i in idx_list)}",
            f"--neighborhood {' '.join(f'{r:.4f}' for r in rad_list)}",
            "--save-npz-only",
            f"--padding {args.padding}",
            f"--output_dir {bundle_dir}",
        ]
        script = f" cd {workdir} && {PYTHON_PATH} scripts/paper/visualize_landscale_2pca.py {' '.join(cmd_args)}"
        launched += int(_maybe_launch(job_desc, bundle_npz, script))
        manifest_root = bundle_dir
    else:
        manifest["output_root"] = output_root
        manifest["shards"] = []
        for idx in indices.tolist():
            r = radii[idx]
            shard_dir = os.path.join(output_root, f"idx_{idx:04d}")
            shard_npz = os.path.join(shard_dir, "landscape_pca_pairs.npz")
            manifest["shards"].append({"anchor_index": idx, "radius": r, "output_dir": shard_dir, "npz": shard_npz})
            job_desc = f"{JOB_DESC_PREFIX} s{sample_id} idx_{idx:04d} #{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"
            cmd_args = [
                f"--dataset_path {dataset_path}",
                f"--sample_id {sample_id}",
                f"--model_checkpoint {model_checkpoint}",
                f"--batch_size {args.batch_size}",
                f"--mesh_resolution {args.mesh_resolution}",
                "--num-frames 1",
                f"--anchor_indices {idx}",
                f"--neighborhood {r:.4f}",
                f"--padding {args.padding}",
                f"--output_dir {shard_dir}",
            ]
            script = f" cd {workdir} && {PYTHON_PATH} scripts/paper/visualize_landscale_2pca.py {' '.join(cmd_args)}"
            launched += int(_maybe_launch(job_desc, shard_npz, script))
        manifest_root = output_root

    os.makedirs(manifest_root, exist_ok=True)
    manifest_path = os.path.join(manifest_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    if args.bundle:
        print(
            f"{'(dry) would launch' if args.dry else 'Launched'} {0 if args.dry else launched} bundle job "
            f"({len(indices)} anchors). Its output IS the dense NPZ (no merge needed):\n  {bundle_npz}"
        )
    else:
        print(
            f"{'(dry) would launch' if args.dry else 'Launched'} {len(indices) if args.dry else launched} job(s). "
            f"After they finish, merge with:\n"
            f"  PYTHONPATH=./src python scripts/paper/merge_landscape_shards.py "
            f"--manifest {manifest_path} --base-npz {args.npz}"
        )


if __name__ == "__main__":
    main()
