"""Plot baseline HellaSwag accuracy vs. paraphrase temperature.

Scans the output dirs produced by `run_jobs_hellaswag_evaluate.py` with
`--temperatures` flag, parses each `results.json`, and renders a line chart
of `baseline.accuracy` per temperature, plus the upstream (non-paraphrased)
baseline as a star at the leftmost x.

Example:
    python scripts/plots/plot_hellaswag_paraphrase_accuracy.py \
        --model Meta-Llama-3.1-8B \
        --temperatures 0.0,0.5,1.0,1.5,2.0 \
        --output artifacts/hellaswag_paraphrase_accuracy.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HellaSwag accuracy vs. paraphrase temperature.")
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
        default="artifacts/hellaswag_evaluation",
        help="Parent dir where eval output subdirs live.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/hellaswag_paraphrase_accuracy.png",
        help="Where to write the PNG (and a sibling .json with the underlying numbers).",
    )
    return parser.parse_args()


def output_dir_for(args: argparse.Namespace, temperature: float) -> str:
    return os.path.join(
        args.eval_root,
        f"hellaswag_{args.model}"
        f"_samples_{args.samples}"
        f"_lr_{args.learning_rate}"
        f"_batch_{args.batch_size}"
        f"_paraphrase_t{temperature:.2f}",
    )


def baseline_dir_for(args: argparse.Namespace) -> str:
    """Non-paraphrased eval output dir (no _paraphrase_t suffix)."""
    return os.path.join(
        args.eval_root,
        f"hellaswag_{args.model}" f"_samples_{args.samples}" f"_lr_{args.learning_rate}" f"_batch_{args.batch_size}",
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

    series: list[tuple[float, dict[str, float]]] = []
    missing: list[str] = []
    for temp in temperatures:
        out_dir = output_dir_for(args, temp)
        metrics = load_baseline_accuracy(out_dir)
        if metrics is None:
            missing.append(f"HellaSwag t={temp:.2f} -> {out_dir}")
            continue
        series.append((temp, metrics))

    upstream_dir = baseline_dir_for(args)
    upstream_metrics = load_baseline_accuracy(upstream_dir)
    if upstream_metrics is None:
        missing.append(f"HellaSwag UPSTREAM baseline -> {upstream_dir}")

    if missing:
        print("Missing or unparseable results for:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    if not series and upstream_metrics is None:
        print("No results to plot. Aborting.", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    color = "tab:green"

    if series:
        series.sort(key=lambda p: p[0])
        xs = [t for t, _ in series]
        ys = [m["accuracy"] for _, m in series]
        ax.plot(xs, ys, marker="o", label="HellaSwag", color=color)
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=color,
            )

    if upstream_metrics is not None:
        star_x = min(temperatures) - 0.15
        y = upstream_metrics["accuracy"]
        ax.scatter(
            [star_x],
            [y],
            marker="*",
            s=220,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
            label="HellaSwag upstream (no paraphrase)",
        )
        ax.annotate(
            f"{y:.3f}",
            (star_x, y),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
            color=color,
        )

    ax.set_xlabel("Paraphrase temperature")
    ax.set_ylabel("Baseline accuracy")
    ax.set_title(f"{args.model} on HellaSwag (paraphrased validation)")
    ax.set_xticks(temperatures)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")

    numbers_path = os.path.splitext(args.output)[0] + ".json"
    payload = {
        "temperatures": [{"temperature": t, **m} for t, m in sorted(series)],
        "upstream": upstream_metrics,
    }
    with open(numbers_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {numbers_path}")

    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
