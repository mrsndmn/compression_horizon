"""Post-hoc trajectory analysis on a saved Progressive run (paper Section 5.1).

Reads the per-stage embeddings stored in ``progressive_prefixes/``, groups them
by ``sample_id`` and ordered by ``stage_index``, and computes paper-canonical
trajectory metrics from Table 13:

    - L_traj  (eq. 3): sum of L2 distances between consecutive stages
    - PCA 99%: minimum #components reaching 99% cumulative explained variance

Output: ``--output_dir/trajectory.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset

from compression_horizon.analysis import (
    compute_pca_99,
    compute_trajectory_length,
    summarize_trajectory,
)


def _load_progressive_dataset(source_dir: str) -> Dataset:
    path = os.path.join(source_dir, "progressive_prefixes")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist. This analyzer requires a Progressive run "
            f"(Full Cramming saves only the final embedding, not the trajectory)."
        )
    return Dataset.load_from_disk(path)


def _group_stages_by_sample(ds: Dataset) -> dict[int, list[dict]]:
    """Return {sample_id: [row, ...]} ordered by stage_index ascending."""
    by_sample: dict[int, list[dict]] = {}
    for row in ds:
        by_sample.setdefault(int(row["sample_id"]), []).append(row)
    for sample_id in by_sample:
        by_sample[sample_id].sort(key=lambda r: int(r["stage_index"]))
    return by_sample


def _stack_stage_embeddings(rows: list[dict]) -> torch.Tensor:
    """Build a [n_stages, num_compression_tokens, hidden] tensor."""
    tensors = [torch.tensor(row["embedding"], dtype=torch.float32) for row in rows]
    if tensors[0].dim() == 1:
        tensors = [t.unsqueeze(0) for t in tensors]
    return torch.stack(tensors, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory length + PCA 99% (paper Section 5.1).")
    parser.add_argument(
        "--source_dir",
        required=True,
        help="Artifacts dir from train.py (Progressive run).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write trajectory.json (default: source_dir).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Cap the number of samples evaluated.",
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=0.99,
        help="Cumulative variance threshold for PCA (paper default: 0.99).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.source_dir
    os.makedirs(output_dir, exist_ok=True)

    ds = _load_progressive_dataset(args.source_dir)
    by_sample = _group_stages_by_sample(ds)
    sample_ids = sorted(by_sample.keys())
    if args.num_samples is not None:
        sample_ids = sample_ids[: args.num_samples]
    print(f"Analyzing {len(sample_ids)} samples")

    sample_records: list[dict] = []
    per_sample_lengths: list[float] = []
    per_sample_pca99: list[int | None] = []
    per_sample_num_stages: list[int] = []

    for sample_id in sample_ids:
        rows = by_sample[sample_id]
        stages = _stack_stage_embeddings(rows)
        n_stages = stages.shape[0]

        length = compute_trajectory_length(stages)
        pca99 = compute_pca_99(stages, variance_threshold=args.variance_threshold)

        sample_records.append(
            {
                "sample_id": sample_id,
                "num_stages": n_stages,
                "final_stage_seq_len": int(rows[-1]["stage_seq_len"]),
                "trajectory_length": length,
                "pca_99": pca99,
            }
        )
        per_sample_lengths.append(length)
        per_sample_pca99.append(pca99)
        per_sample_num_stages.append(n_stages)

    summary = summarize_trajectory(per_sample_lengths, per_sample_pca99, per_sample_num_stages)

    output = {
        "config": {
            "source_dir": args.source_dir,
            "num_samples": len(sample_records),
            "variance_threshold": args.variance_threshold,
        },
        "summary": summary,
        "samples": sample_records,
    }
    output_path = Path(output_dir) / "trajectory.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {output_path}")

    print()
    print("Summary:")
    print(f"  trajectory_length: {summary['trajectory_length']['mean']:.2f} ± {summary['trajectory_length']['std']:.2f}")
    print(f"  pca_99:            {summary['pca_99']['mean']:.2f} ± {summary['pca_99']['std']:.2f}")
    print(f"  num_stages:        {summary['num_stages']['mean']:.2f} ± {summary['num_stages']['std']:.2f}")
    if summary["num_pca_excluded"] > 0:
        print(f"  (excluded from PCA aggregate: {summary['num_pca_excluded']} sample(s) with <2 stages)")


if __name__ == "__main__":
    main()
