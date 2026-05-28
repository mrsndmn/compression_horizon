"""Merge per-anchor accuracy-field shard NPZs into one dense ``landscape_pca_pairs.npz``.

``run_jobs_densify_fields.py`` fans accuracy-field computation out one-anchor-per-job;
each shard writes its own ``landscape_pca_pairs.npz``. Because every shard re-fits the
deterministic per-sample PCA (``random_state=42``) on the same dataset, their ``coords``
/ ``explained_variance_ratio`` are identical, so the only per-shard differences are the
anchor and its accuracy field -- which means we can stack them along a frame axis.

Two NPZ schemas are normalized here:
  * multi-frame (the cached base NPZ): ``accuracy [F,P,H,W]``, ``anchor_coords [F,D]``,
    ``sampled_indices [F]``.
  * single-frame (one-anchor shard jobs): ``accuracy [P,H,W]``, ``anchor_coords [D]``,
    ``current_idx [1]`` and no ``sampled_indices``.

Output is the multi-frame schema that ``animate_trajectory.py`` / ``visual_abstract.py``
already consume, with frames deduplicated by anchor index and sorted along the path.

Usage::

    PYTHONPATH=./src python scripts/paper/merge_landscape_shards.py \
        --manifest .../field_shards/manifest.json --base-npz .../landscape_pca_pairs.npz
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def _normalize(npz) -> List[Dict]:
    """Return a list of per-frame dicts from one NPZ (either schema)."""
    acc = npz["accuracy"]
    gx = npz["grid_x"]
    gy = npz["grid_y"]
    anchor = npz["anchor_coords"]

    if "sampled_indices" in npz:  # multi-frame
        idxs = npz["sampled_indices"].reshape(-1).astype(np.int64)
        seq = npz["sampled_seq_len"].reshape(-1).astype(np.int64) if "sampled_seq_len" in npz else np.full(idxs.shape, -1)
        rad = npz["neighborhood_radius"].reshape(-1) if "neighborhood_radius" in npz else np.array([])
        F = idxs.shape[0]
        # grid may be shared [P,H,W] (no neighborhood) or per-frame [F,P,H,W].
        gx_f = gx if gx.ndim == 4 else np.broadcast_to(gx, (F,) + gx.shape).copy()
        gy_f = gy if gy.ndim == 4 else np.broadcast_to(gy, (F,) + gy.shape).copy()
        anchor_f = anchor if anchor.ndim == 2 else np.broadcast_to(anchor, (F,) + anchor.shape).copy()
    else:  # single-frame
        idxs = npz["current_idx"].reshape(-1).astype(np.int64)
        seq = npz["current_seq_len"].reshape(-1).astype(np.int64) if "current_seq_len" in npz else np.array([-1])
        rad = npz["neighborhood_radius"].reshape(-1) if "neighborhood_radius" in npz else np.array([])
        acc = acc[None, ...]  # [P,H,W] -> [1,P,H,W]
        gx_f = gx[None, ...]
        gy_f = gy[None, ...]
        anchor_f = anchor[None, ...]  # [D] -> [1,D]

    out = []
    for k in range(idxs.shape[0]):
        out.append(
            {
                "idx": int(idxs[k]),
                "seq": int(seq[k]) if k < seq.shape[0] else -1,
                "radius": float(rad[k]) if k < rad.shape[0] else float("nan"),
                "accuracy": acc[k],  # [P,H,W]
                "grid_x": gx_f[k],
                "grid_y": gy_f[k],
                "anchor": anchor_f[k],  # [D]
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=str, default=None, help="field_shards/manifest.json from the launcher.")
    ap.add_argument("--inputs", type=str, nargs="+", default=None, help="Explicit shard NPZ paths or globs.")
    ap.add_argument("--base-npz", dest="base_npz", type=str, default=None, help="Base NPZ whose frames are also included.")
    ap.add_argument("--no-base", dest="no_base", action="store_true", help="Do not include --base-npz frames.")
    ap.add_argument("--output", type=str, default=None, help="Output NPZ (default: <base dir>/landscape_pca_pairs_dense.npz).")
    ap.add_argument("--coords-atol", dest="coords_atol", type=float, default=1e-3, help="Tolerance for PCA-agreement check.")
    args = ap.parse_args()

    shard_npzs: List[str] = []
    if args.manifest:
        man = json.load(open(args.manifest))
        shard_npzs += [s["npz"] for s in man.get("shards", [])]
        if not args.base_npz and man.get("npz"):
            args.base_npz = man["npz"]
    if args.inputs:
        for pat in args.inputs:
            shard_npzs += sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]

    present = [p for p in shard_npzs if os.path.exists(p)]
    missing = [p for p in shard_npzs if not os.path.exists(p)]
    if missing:
        print(f"\033[33m{len(missing)} shard NPZ(s) not found yet (skipping):\033[0m")
        for p in missing:
            print(f"  {p}")
    if not present and not (args.base_npz and not args.no_base):
        raise SystemExit("No shard NPZs found and no base NPZ to merge.")

    # Reference shared arrays come from the base (or first available shard).
    ref_path = args.base_npz if (args.base_npz and not args.no_base) else present[0]
    ref = np.load(ref_path, allow_pickle=True)
    ref_coords = ref["coords"].astype(np.float64)
    pair_indices = ref["pair_indices"]
    # The accuracy fields live in the PCs named by pair_indices, and those are the only
    # coords used downstream. High-rank, ~0-variance tail PCs are numerically unstable
    # across deterministic refits (sign/ordering noise), so only require agreement on the
    # plotted PCs -- not all 1013 dims -- when verifying the shared basis.
    used_pcs = np.unique(np.asarray(pair_indices).reshape(-1)).astype(np.int64)
    P = int(ref["accuracy"].shape[-3]) if ref["accuracy"].ndim == 4 else int(ref["accuracy"].shape[0])
    H, W = int(ref["accuracy"].shape[-2]), int(ref["accuracy"].shape[-1])

    inputs: List[str] = []
    if args.base_npz and not args.no_base:
        inputs.append(args.base_npz)
    inputs += present

    frames: Dict[int, Dict] = {}
    for path in inputs:
        z = np.load(path, allow_pickle=True)
        # Hard guarantee that frames are stackable + on the same PCA basis.
        c = z["coords"].astype(np.float64)
        if c.shape != ref_coords.shape or not np.allclose(c[:, used_pcs], ref_coords[:, used_pcs], atol=args.coords_atol):
            raise SystemExit(
                f"PCA/coords mismatch in {path} on the plotted PCs {used_pcs.tolist()} "
                "-- shards must share the deterministic PCA fit. Aborting."
            )
        if not np.array_equal(z["pair_indices"], pair_indices):
            raise SystemExit(f"pair_indices mismatch in {path}.")
        for fr in _normalize(z):
            if fr["accuracy"].shape != (P, H, W):
                raise SystemExit(
                    f"mesh shape {fr['accuracy'].shape} != ({P},{H},{W}) in {path}; rerun shards at the same mesh_resolution."
                )
            if fr["idx"] in frames:
                print(
                    f"\033[33mDuplicate anchor idx={fr['idx']} (from {os.path.basename(os.path.dirname(path))}); overwriting.\033[0m"
                )
            frames[fr["idx"]] = fr

    order = sorted(frames)
    fr = [frames[i] for i in order]
    print(f"Merged {len(fr)} unique anchors: {order}")

    out = args.output or os.path.join(os.path.dirname(ref_path), "landscape_pca_pairs_dense.npz")
    np.savez_compressed(
        out,
        pair_indices=pair_indices,
        grid_x=np.stack([f["grid_x"] for f in fr], axis=0),
        grid_y=np.stack([f["grid_y"] for f in fr], axis=0),
        accuracy=np.stack([f["accuracy"] for f in fr], axis=0),
        coords=ref_coords.astype(np.float32),
        stage_index=ref["stage_index"],
        stage_seq_len=ref["stage_seq_len"],
        sampled_indices=np.array(order, dtype=np.int64),
        sampled_seq_len=np.array([f["seq"] for f in fr], dtype=np.int64),
        anchor_coords=np.stack([f["anchor"] for f in fr], axis=0),
        neighborhood_radius=np.array([f["radius"] for f in fr], dtype=np.float32),
        explained_variance_ratio=ref["explained_variance_ratio"],
        model_checkpoint=ref["model_checkpoint"],
        dataset_path=ref["dataset_path"],
        sample_id=ref["sample_id"],
        created_at=np.array([__import__("datetime").datetime.now().isoformat()]),
    )
    print(f"Saved dense NPZ: {out}  (accuracy {np.stack([f['accuracy'] for f in fr]).shape})")


if __name__ == "__main__":
    main()
