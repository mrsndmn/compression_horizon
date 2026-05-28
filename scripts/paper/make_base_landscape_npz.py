"""Compute the per-sample *base trajectory* NPZ (PCA coords + metadata) on CPU.

The densify launcher (``scripts/jobs/run_jobs_densify_fields.py``) and the merge step
both key off a base ``landscape_pca_pairs.npz`` that carries the trajectory ``coords``
(the per-sample PCA projection) plus ``dataset_path`` / ``sample_id`` /
``model_checkpoint`` metadata. For sample 0 that file was produced as a side effect of a
GPU ``visualize_landscale_2pca.py`` run, but the *coords themselves need no GPU* -- they
are a deterministic PCA (``random_state=42``) over the cached compression embeddings.

This script computes that base NPZ for one or more samples without loading the model, so
the launcher can plan shard/bundle jobs for new samples immediately. It writes the exact
same PCA basis a GPU shard would re-fit (identical ``X`` row order + identical PCA call),
so coords are consistent with the field jobs that attach to them.

Output per sample: ``<exp_dir>/visualizations_s<N>/landscape_pca_pairs.npz`` (override the
parent with ``--output-root``). Only the launcher-relevant fields are written (no accuracy
field) -- the dense field NPZ is produced later by the GPU bundle job.

Usage::

    PYTHONPATH=./src python scripts/paper/make_base_landscape_npz.py \
        --dataset_path .../progressive_prefixes --sample_ids 1 2 3 4 5 6 7 8 9
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
from sklearn.decomposition import PCA

# Allow `import scripts.paper...` when run as a script file (sys.path[0] is this dir).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.paper.visualize_landscale_2pca import (
    _infer_output_dir,
    _most_common_str,
    flatten_embedding,
    load_progressive_dataset,
)


def _collect_rows_for_samples(ds, sample_ids: List[int]) -> Dict[int, List[dict]]:
    """One pass over the dataset, grouping rows by sample_id (heavy cols dropped).

    Keeps only the light fields we need per row (stage_index, stage_seq_len,
    model_checkpoint) plus the flattened embedding, so peak memory stays bounded.
    """
    drop_cols = [c for c in ["orig_embedding", "initialization_embedding"] if c in ds.column_names]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    wanted = set(int(s) for s in sample_ids)
    out: Dict[int, List[dict]] = {s: [] for s in wanted}
    from tqdm import tqdm

    for i in tqdm(range(len(ds)), desc=f"Grouping samples {sorted(wanted)}"):
        r = ds[i]
        sid = int(r.get("sample_id", -1))
        if sid not in wanted:
            continue
        out[sid].append(
            {
                "stage_index": int(r.get("stage_index", 0)),
                "stage_seq_len": int(r.get("stage_seq_len", -1)),
                "model_checkpoint": str(r.get("model_checkpoint", "")).strip(),
                "emb": flatten_embedding(r),
            }
        )
    return out


def _base_npz_for_sample(rows: List[dict], dataset_path: str, sample_id: int, model_override: str | None) -> Dict:
    rows_sorted = sorted(rows, key=lambda r: r["stage_index"])
    X = np.stack([r["emb"] for r in rows_sorted], axis=0)
    if X.shape[0] < 2:
        raise ValueError(f"sample {sample_id}: need >=2 stages for PCA, got {X.shape[0]}")
    n_components = int(min(X.shape[0] - 1, X.shape[1]))
    if n_components < 2:
        raise ValueError(f"sample {sample_id}: PCA needs >=2 components, got {n_components}")
    pca = PCA(n_components=n_components, random_state=42)  # matches visualize_landscale_2pca
    coords = pca.fit_transform(X)
    model_checkpoint = model_override or _most_common_str([r["model_checkpoint"] for r in rows_sorted])
    if not model_checkpoint:
        raise ValueError(f"sample {sample_id}: could not infer model_checkpoint; pass --model_checkpoint")
    return {
        "pair_indices": np.array([[0, 1]], dtype=np.int64),
        "coords": coords.astype(np.float32),
        "stage_index": np.array([r["stage_index"] for r in rows_sorted], dtype=np.int64),
        "stage_seq_len": np.array([r["stage_seq_len"] for r in rows_sorted], dtype=np.int64),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "model_checkpoint": np.array([model_checkpoint]),
        "dataset_path": np.array([dataset_path]),
        "sample_id": np.array([int(sample_id)], dtype=np.int64),
        "created_at": np.array([datetime.now().isoformat()]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_path", "--dataset-path", dest="dataset_path", type=str, required=True)
    ap.add_argument("--sample_ids", "--sample-ids", dest="sample_ids", type=int, nargs="+", required=True)
    ap.add_argument("--model_checkpoint", "--model-checkpoint", dest="model_checkpoint", type=str, default=None)
    ap.add_argument(
        "--output-root",
        dest="output_root",
        type=str,
        default=None,
        help="Parent dir for per-sample viz dirs (default: <exp_dir>, i.e. dirname(dataset_path)).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if the base NPZ already exists.")
    args = ap.parse_args()

    if not os.path.isdir(args.dataset_path):
        raise SystemExit(f"dataset not found: {args.dataset_path}")
    exp_dir = os.path.dirname(_infer_output_dir(args.dataset_path))  # dirname(.../visualizations) == exp_dir
    output_root = args.output_root or exp_dir

    todo = []
    for s in args.sample_ids:
        out_npz = os.path.join(output_root, f"visualizations_s{s}", "landscape_pca_pairs.npz")
        if os.path.exists(out_npz) and not args.overwrite:
            print(f"\033[33mSkip (exists):\033[0m {out_npz}")
            continue
        todo.append(s)
    if not todo:
        print("Nothing to do.")
        return

    ds = load_progressive_dataset(args.dataset_path)
    grouped = _collect_rows_for_samples(ds, todo)

    for s in todo:
        rows = grouped.get(s, [])
        if not rows:
            print(f"\033[31mSkip (no rows):\033[0m sample {s} not found in dataset")
            continue
        payload = _base_npz_for_sample(rows, args.dataset_path, s, args.model_checkpoint)
        out_dir = os.path.join(output_root, f"visualizations_s{s}")
        os.makedirs(out_dir, exist_ok=True)
        out_npz = os.path.join(out_dir, "landscape_pca_pairs.npz")
        np.savez_compressed(out_npz, **payload)
        c = payload["coords"]
        print(
            f"\033[32mSaved:\033[0m {out_npz}  coords={c.shape}  "
            f"PC1={payload['explained_variance_ratio'][0]:.3f} "
            f"PC1+2={payload['explained_variance_ratio'][:2].sum():.3f}"
        )


if __name__ == "__main__":
    main()
