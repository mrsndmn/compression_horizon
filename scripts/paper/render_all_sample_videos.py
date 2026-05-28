"""Batch-render the trajectory animations for every progressive-cramming sample.

For each sample we render two variants (both drive ``animate_trajectory.py``):

  * ``pca``   -- PCA-plane view with the accuracy-field overlay and the talk camera arc
                 (zoom IN to the end region -> end-hold pause -> smooth zoom OUT to full).
  * ``steps`` -- step-warped view (``--normalize-by-steps``): each hop's length is proportional
                 to the optimizer steps that converged that token, with one smooth eased
                 zoom-OUT from the small early hops to the full path (``--zoom-out-start``).

Camera tokens are derived per sample from the trajectory length ``n_traj`` (the trajectories
differ a lot in length, ~660 to ~2000), so the same fractions frame "the end" / "the start"
consistently across samples.

Samples are discovered under ``<base>/visualizations`` (sample 0) and
``<base>/visualizations_s{N}/dense`` (samples 1+).

Examples::

    # Fast, low-res drafts for every sample, both variants:
    PYTHONPATH=./src python scripts/paper/render_all_sample_videos.py --draft

    # Just a couple of samples, only the steps variant, print commands without running:
    PYTHONPATH=./src python scripts/paper/render_all_sample_videos.py \
        --samples 0,1 --variants steps --dry-run
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANIMATE = os.path.join(HERE, "animate_trajectory.py")
DEFAULT_BASE = "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1"

# Quality presets. Draft = fast/low-res for review; full = talk quality.
PRESETS = {
    "draft": {"fps": 15, "dpi": 80, "seconds": 14.0, "hold_end": 1.5, "zoom_out_seconds": 3.0},
    "full": {"fps": 30, "dpi": 150, "seconds": 18.0, "hold_end": 2.0, "zoom_out_seconds": 3.0},
}

# Camera tokens as fractions of the trajectory length (see module docstring).
PCA_FROM_FRAC = 0.40  # zoom-in begins when the cursor reaches this token
PCA_START_FRAC = 0.78  # region framed = tokens [start:]
PCA_TO_FRAC = 0.80  # zoom-in completes here
STEPS_OUT_START_FRAC = 0.20  # start framed on tokens [0:K]
STEPS_TO_FRAC = 0.77  # zoom-out-from-start completes here


def dense_npz_path(base: str, sid: int) -> Optional[str]:
    """Path to a sample's dense (merged) landscape NPZ, or None if absent."""
    candidates = [os.path.join(base, "visualizations", "landscape_pca_pairs_dense.npz")] if sid == 0 else []
    candidates += [
        os.path.join(base, f"visualizations_s{sid}", "dense", "landscape_pca_pairs.npz"),
        os.path.join(base, f"visualizations_s{sid}", "landscape_pca_pairs_dense.npz"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def discover_samples(base: str, want: Optional[List[int]]) -> List[Tuple[int, str]]:
    sids = want if want is not None else list(range(0, 10))
    found = []
    for sid in sids:
        p = dense_npz_path(base, sid)
        if p is None:
            print(f"\033[33msample {sid}: no dense NPZ found, skipping.\033[0m")
            continue
        found.append((sid, p))
    return found


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def camera_tokens(n_traj: int) -> Dict[str, int]:
    """Per-sample camera tokens derived from the trajectory length."""
    zfrom = _clamp(round(PCA_FROM_FRAC * n_traj), 1, n_traj - 3)
    zstart = _clamp(round(PCA_START_FRAC * n_traj), zfrom + 1, n_traj - 2)
    zto = _clamp(round(PCA_TO_FRAC * n_traj), zfrom + 1, n_traj - 2)
    if zto <= zfrom:
        zto = _clamp(zfrom + 1, zfrom + 1, n_traj - 2)
    out_start = _clamp(round(STEPS_OUT_START_FRAC * n_traj), 2, n_traj - 2)
    steps_to = _clamp(round(STEPS_TO_FRAC * n_traj), 1, n_traj - 2)
    return {
        "pca_zoom_from": zfrom,
        "pca_zoom_start": zstart,
        "pca_zoom_to": zto,
        "steps_out_start": out_start,
        "steps_zoom_to": steps_to,
    }


def build_commands(sid: int, npz: str, n_traj: int, q: Dict, out_dir: str, variants: List[str]) -> List[Tuple[str, List[str]]]:
    tok = camera_tokens(n_traj)
    common = [
        "--fps",
        str(q["fps"]),
        "--dpi",
        str(q["dpi"]),
        "--seconds",
        str(q["seconds"]),
        "--hold-end",
        str(q["hold_end"]),
    ]
    cmds = []
    if "pca" in variants:
        out = os.path.join(out_dir, f"sample{sid}_pca")
        cmds.append(
            (
                "pca",
                [sys.executable, ANIMATE, "--npz_path", npz, "--output", out]
                + common
                + [
                    "--zoom-start",
                    str(tok["pca_zoom_start"]),
                    "--zoom-from",
                    str(tok["pca_zoom_from"]),
                    "--zoom-to",
                    str(tok["pca_zoom_to"]),
                    "--zoom-out-seconds",
                    str(q["zoom_out_seconds"]),
                ],
            )
        )
    if "steps" in variants:
        out = os.path.join(out_dir, f"sample{sid}_steps")
        cmds.append(
            (
                "steps",
                [sys.executable, ANIMATE, "--npz_path", npz, "--output", out]
                + common
                + [
                    "--normalize-by-steps",
                    "--zoom-out-start",
                    str(tok["steps_out_start"]),
                    "--zoom-to",
                    str(tok["steps_zoom_to"]),
                ],
            )
        )
    return cmds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", dest="base_dir", default=DEFAULT_BASE, help="Experiment dir holding visualizations*/.")
    ap.add_argument("--samples", default="all", help="Comma-separated sample ids (e.g. 0,1,2) or 'all'.")
    ap.add_argument("--variants", default="both", choices=["pca", "steps", "both"], help="Which video(s) per sample.")
    ap.add_argument("--draft", action="store_true", help="Fast low-res preset (fps 15, dpi 80) for review.")
    ap.add_argument("--out-dir", dest="out_dir", default=None, help="Output dir (default: <base>/videos[_draft]).")
    ap.add_argument("--fps", type=int, default=None, help="Override preset fps.")
    ap.add_argument("--dpi", type=int, default=None, help="Override preset dpi.")
    ap.add_argument("--seconds", type=float, default=None, help="Override preset reveal seconds.")
    ap.add_argument("--hold-end", dest="hold_end", type=float, default=None, help="Override preset end-hold pause.")
    ap.add_argument("--zoom-out-seconds", dest="zoom_out_seconds", type=float, default=None, help="Override PCA end zoom-out.")
    ap.add_argument("--dry-run", action="store_true", help="Print the per-sample commands without rendering.")
    args = ap.parse_args()

    want = None if args.samples.strip().lower() == "all" else [int(s) for s in args.samples.split(",") if s.strip() != ""]
    variants = ["pca", "steps"] if args.variants == "both" else [args.variants]

    q = dict(PRESETS["draft" if args.draft else "full"])
    for k in ("fps", "dpi", "seconds", "hold_end", "zoom_out_seconds"):
        if getattr(args, k) is not None:
            q[k] = getattr(args, k)

    out_dir = args.out_dir or os.path.join(args.base_dir, "videos_draft" if args.draft else "videos")
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)

    samples = discover_samples(args.base_dir, want)
    if not samples:
        raise SystemExit("No samples with a dense NPZ were found.")

    env = dict(os.environ)
    env["PYTHONPATH"] = "./src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    print(f"Mode: {'DRAFT' if args.draft else 'FULL'}  fps={q['fps']} dpi={q['dpi']} seconds={q['seconds']}")
    print(f"Output dir: {out_dir}")
    print(f"Samples: {[s for s, _ in samples]}  variants={variants}\n")

    jobs: List[Tuple[int, str, List[str]]] = []
    for sid, npz in samples:
        n_traj = int(np.load(npz, allow_pickle=True)["coords"].shape[0])
        tok = camera_tokens(n_traj)
        print(
            f"sample {sid}: n_traj={n_traj}  PCA[from={tok['pca_zoom_from']} start={tok['pca_zoom_start']} "
            f"to={tok['pca_zoom_to']}]  STEPS[out_start={tok['steps_out_start']} to={tok['steps_zoom_to']}]"
        )
        for variant, cmd in build_commands(sid, npz, n_traj, q, out_dir, variants):
            jobs.append((sid, variant, cmd))

    print(f"\n{len(jobs)} render job(s).\n")
    failures = []
    for i, (sid, variant, cmd) in enumerate(jobs, 1):
        label = f"[{i}/{len(jobs)}] sample {sid} / {variant}"
        if args.dry_run:
            print(f"{label}:\n  {' '.join(cmd)}\n")
            continue
        print(f"\033[36m{label} -> rendering...\033[0m")
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            print(f"\033[31m{label}: FAILED (exit {res.returncode})\033[0m")
            failures.append(label)
        else:
            print(f"\033[32m{label}: done\033[0m")

    if not args.dry_run:
        print(f"\nFinished. {len(jobs) - len(failures)}/{len(jobs)} succeeded.")
        if failures:
            print("Failed: " + "; ".join(failures))
            raise SystemExit(1)


if __name__ == "__main__":
    main()
