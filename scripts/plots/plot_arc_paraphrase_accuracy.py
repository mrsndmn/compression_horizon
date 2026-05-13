"""Plot baseline ARC accuracy vs. paraphrase temperature for both subsets.

Scans the output dirs produced by `run_jobs_arc_evaluate.py` with
`--temperatures` flag, parses each `results.json`, and renders a line chart
of `baseline.accuracy` per (subset, temperature).

Example:
    python scripts/plots/plot_arc_paraphrase_accuracy.py \
        --model Meta-Llama-3.1-8B \
        --temperatures 0.0,0.5,1.0,1.5,2.0 \
        --output artifacts/arc_paraphrase_accuracy.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

SUBSETS = ("ARC-Challenge", "ARC-Easy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ARC accuracy vs. paraphrase temperature.")
    parser.add_argument("--model", default="Meta-Llama-3.1-8B", help="Model tag used in the eval output dir names.")
    parser.add_argument(
        "--temperatures",
        default="0.0,0.5,1.0,1.5,2.0",
        help="Comma-separated temperatures matching the eval output dir suffixes (e.g. '0.0,0.5,1.0,1.5,2.0').",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=512,
        help="--limit_samples value baked into the output dir name (default: 512, matching the canonical runner).",
    )
    parser.add_argument("--learning_rate", default="0.1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--eval_root",
        default="artifacts/arc_evaluation",
        help="Parent dir where eval output subdirs live.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/arc_paraphrase_accuracy.png",
        help="Where to write the PNG (and a sibling .json with the underlying numbers).",
    )
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace, subset: str, temperature: float) -> str:
    subset_token = subset.replace("-", "_")
    return os.path.join(
        args.eval_root,
        f"arc_{subset_token}_{args.model}"
        f"_samples_{args.samples}"
        f"_lr_{args.learning_rate}"
        f"_batch_{args.batch_size}"
        f"_paraphrase_t{temperature:.2f}",
    )


def baseline_dir_for(args: argparse.Namespace, subset: str) -> str:
    """Non-paraphrased eval output dir (no _paraphrase_t suffix)."""
    subset_token = subset.replace("-", "_")
    return os.path.join(
        args.eval_root,
        f"arc_{subset_token}_{args.model}" f"_samples_{args.samples}" f"_lr_{args.learning_rate}" f"_batch_{args.batch_size}",
    )


def load_baseline_accuracy(path: str) -> dict[str, float] | None:
    results_path = os.path.join(path, "results.json")
    if not os.path.isfile(results_path):
        return None
    with open(results_path) as fh:
        data = json.load(fh)
    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        return None
    return {
        "accuracy": float(baseline.get("accuracy", float("nan"))),
        "token_normalized_accuracy": float(baseline.get("token_normalized_accuracy", float("nan"))),
        "char_normalized_accuracy": float(baseline.get("char_normalized_accuracy", float("nan"))),
        "total_predictions": int(baseline.get("total_predictions", 0)),
    }


def main() -> int:
    args = parse_args()
    temperatures = [float(t.strip()) for t in args.temperatures.split(",") if t.strip()]

    series: dict[str, list[tuple[float, dict[str, float]]]] = {s: [] for s in SUBSETS}
    upstream_baselines: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for subset in SUBSETS:
        for temp in temperatures:
            out_dir = output_dir_for(args, subset, temp)
            metrics = load_baseline_accuracy(out_dir)
            if metrics is None:
                missing.append(f"{subset} t={temp:.2f} -> {out_dir}")
                continue
            series[subset].append((temp, metrics))
        # Non-paraphrased upstream baseline (rendered as a star)
        upstream_dir = baseline_dir_for(args, subset)
        upstream_metrics = load_baseline_accuracy(upstream_dir)
        if upstream_metrics is not None:
            upstream_baselines[subset] = upstream_metrics
        else:
            missing.append(f"{subset} UPSTREAM baseline -> {upstream_dir}")

    if missing:
        print("Missing or unparseable results for:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    if not any(series.values()):
        print("No results to plot. Aborting.", file=sys.stderr)
        return 1

    # Build the plot.
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {"ARC-Challenge": "tab:red", "ARC-Easy": "tab:blue"}
    for subset, points in series.items():
        if not points:
            continue
        points.sort(key=lambda p: p[0])
        xs = [t for t, _ in points]
        ys = [m["accuracy"] for _, m in points]
        ax.plot(xs, ys, marker="o", label=subset, color=colors.get(subset))
        for x, y, (_, m) in zip(xs, ys, points):
            ax.annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=colors.get(subset, "black"),
            )

    # Upstream (non-paraphrased) baselines as stars at the leftmost x.
    if upstream_baselines:
        star_x = min(temperatures) - 0.15  # park stars just left of the curve start
        for subset, metrics in upstream_baselines.items():
            y = metrics["accuracy"]
            ax.scatter(
                [star_x],
                [y],
                marker="*",
                s=220,
                color=colors.get(subset, "black"),
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
                label=f"{subset} upstream (no paraphrase)",
            )
            ax.annotate(
                f"{y:.3f}",
                (star_x, y),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=8,
                color=colors.get(subset, "black"),
            )

    ax.set_xlabel("Paraphrase temperature")
    ax.set_ylabel("Baseline accuracy")
    ax.set_title(f"{args.model} on ARC (paraphrased validation)")
    ax.set_xticks(temperatures)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")

    # Also dump the numbers next to the PNG for easy inspection.
    numbers_path = os.path.splitext(args.output)[0] + ".json"
    payload = {subset: [{"temperature": t, **m} for t, m in sorted(points)] for subset, points in series.items()}
    with open(numbers_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {numbers_path}")

    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
