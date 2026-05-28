"""Post-hoc dimensionality analysis of a progressive-cramming trajectory.

Reads the per-stage embeddings saved in ``progressive_prefixes/`` and, for a
chosen sample, renders 2-D projections of its optimization trajectory
``{E^(1), ..., E^(n)}`` via PCA, t-SNE and UMAP, and estimates the
trajectory's intrinsic dimension with Two-NN -- cross-checked against the
linear PCA-99 % metric used in the paper (Section 5.1).

The non-linear projections (t-SNE, UMAP) and the Two-NN estimate are an
extension beyond the paper, added for visualization completeness; the
quantitative reconstruction analysis still relies on linear PCA
(``run_pca_reconstruction.py``).

Outputs under ``--output_dir``:
    dimreduction.json                       Two-NN / PCA-99 % / trajectory length
    dimreduction_sample<sid>_<method>.png    per-method 2-D projection
    dimreduction_sample<sid>_panel.png       the methods side by side
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402

from compression_horizon.analysis import (  # noqa: E402
    compute_pca_99,
    compute_trajectory_length,
    estimate_twonn,
    project_2d,
)

METHODS = ("pca", "tsne", "umap")
_TITLES = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}
_AXIS_LABELS = {
    "pca": ("Главная компонента 1", "Главная компонента 2"),
    "tsne": ("Компонента t-SNE 1", "Компонента t-SNE 2"),
    "umap": ("Компонента UMAP 1", "Компонента UMAP 2"),
}


def _load_progressive_dataset(source_dir: str) -> Dataset:
    path = os.path.join(source_dir, "progressive_prefixes")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- dimensionality analysis needs a Progressive run.")
    return Dataset.load_from_disk(path)


def _build_trajectory(rows: list[dict]) -> torch.Tensor:
    """[n_stages, flat_dim] tensor; rows ordered by stage_index ascending."""
    rows_sorted = sorted(rows, key=lambda r: int(r["stage_index"]))
    tensors = [torch.tensor(r["embedding"], dtype=torch.float32).reshape(-1) for r in rows_sorted]
    return torch.stack(tensors, dim=0)


def _scatter(ax, coords: np.ndarray, method: str) -> None:
    """Paper-Figure-3-style trajectory plot: black points + thin grey path.

    Style matches ``run_trajectory_landscape.py``: ``s=10`` black markers, a
    grey connecting line (``color="0.35"``, ``linewidth=0.7``) showing
    optimization order, default axis ticks, and Russian axis labels.
    """
    ax.plot(coords[:, 0], coords[:, 1], color="0.35", linewidth=0.7, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], c="black", s=10, zorder=2)
    ax.set_title(_TITLES.get(method, method))
    x_label, y_label = _AXIS_LABELS.get(method, ("Компонента 1", "Компонента 2"))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    t = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(t.mean().item()), "std": float(t.std(unbiased=False).item()), "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-linear dimensionality analysis of a progressive trajectory.")
    parser.add_argument("--source_dir", required=True, help="Artifacts dir from a Progressive train.py run.")
    parser.add_argument("--output_dir", default=None, help="Where to write outputs (default: source_dir).")
    parser.add_argument(
        "--sample_id",
        type=int,
        default=None,
        help="Sample to render figures for (default: the one with the most stages).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Cap the number of samples for the Two-NN / PCA-99 aggregate.",
    )
    parser.add_argument("--methods", default="pca,tsne,umap", help="Comma-separated projection methods to render.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discard_fraction", type=float, default=0.1, help="Two-NN upper-tail discard fraction.")
    parser.add_argument(
        "--min_stages_twonn",
        type=int,
        default=10,
        help="Skip Two-NN for trajectories shorter than this many stages.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.source_dir
    os.makedirs(output_dir, exist_ok=True)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected from {METHODS}")

    ds = _load_progressive_dataset(args.source_dir)
    by_sample: dict[int, list[dict]] = defaultdict(list)
    for row in ds:
        by_sample[int(row["sample_id"])].append(row)
    sample_ids = sorted(by_sample.keys())
    if args.num_samples is not None:
        sample_ids = sample_ids[: args.num_samples]
    print(f"Loaded {len(sample_ids)} sample(s) from {args.source_dir}")

    per_sample: list[dict] = []
    trajectories: dict[int, torch.Tensor] = {}
    for sample_id in sample_ids:
        trajectory = _build_trajectory(by_sample[sample_id])
        trajectories[sample_id] = trajectory
        n_stages = int(trajectory.shape[0])
        record = {
            "sample_id": sample_id,
            "n_stages": n_stages,
            "trajectory_length": compute_trajectory_length(trajectory),
            "pca_99": compute_pca_99(trajectory),
            "twonn": None,
        }
        if n_stages >= args.min_stages_twonn:
            record["twonn"] = estimate_twonn(trajectory.numpy(), discard_fraction=args.discard_fraction)["intrinsic_dim"]
        per_sample.append(record)
        print(f"  sample {sample_id}: stages={n_stages}  PCA-99%={record['pca_99']}  TwoNN={record['twonn']}")

    if not per_sample:
        raise RuntimeError("No samples found in progressive_prefixes/.")

    # Choose the sample to render figures for.
    if args.sample_id is not None:
        if args.sample_id not in trajectories:
            raise ValueError(f"--sample_id {args.sample_id} not found in {sorted(trajectories)}")
        figure_sample_id = args.sample_id
    else:
        figure_sample_id = max(per_sample, key=lambda r: r["n_stages"])["sample_id"]

    figure_trajectory = trajectories[figure_sample_id].numpy()
    print(f"Rendering figures for sample {figure_sample_id} ({figure_trajectory.shape[0]} stages)")

    projections = {method: project_2d(figure_trajectory, method=method, seed=args.seed) for method in methods}

    for method in methods:
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        _scatter(ax, projections[method], method)
        fig.tight_layout()
        path = Path(output_dir) / f"dimreduction_sample{figure_sample_id}_{method}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  wrote {path}")

    if len(methods) > 1:
        fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.5))
        for ax, method in zip(axes, methods):
            _scatter(ax, projections[method], method)
        fig.tight_layout()
        path = Path(output_dir) / f"dimreduction_sample{figure_sample_id}_panel.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")

    output = {
        "config": {
            "source_dir": args.source_dir,
            "figure_sample_id": figure_sample_id,
            "methods": methods,
            "seed": args.seed,
            "discard_fraction": args.discard_fraction,
        },
        "aggregate": {
            "twonn": _mean_std([r["twonn"] for r in per_sample if r["twonn"] is not None]),
            "pca_99": _mean_std([r["pca_99"] for r in per_sample if r["pca_99"] is not None]),
        },
        "samples": per_sample,
    }
    output_path = Path(output_dir) / "dimreduction.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {output_path}")

    twonn = output["aggregate"]["twonn"]
    pca99 = output["aggregate"]["pca_99"]
    print()
    print("Aggregate intrinsic dimension:")
    if twonn["mean"] is not None:
        print(f"  TwoNN   : {twonn['mean']:.2f} ± {twonn['std']:.2f}  (n={twonn['n']})")
    if pca99["mean"] is not None:
        print(f"  PCA-99% : {pca99['mean']:.2f} ± {pca99['std']:.2f}  (n={pca99['n']})")


if __name__ == "__main__":
    main()
