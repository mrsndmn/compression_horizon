#!/usr/bin/env python3
"""
Print a table of token-normalized accuracies for HellaSwag and ARC.

Rows: model checkpoints
Columns: HS/ARC baseline and compressed token accuracies
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from tabulate import tabulate


@dataclass
class RunMetrics:
    model_checkpoint: Optional[str]
    baseline_token_accuracy: Optional[float]
    compressed_token_accuracy: Optional[float]
    arc_split: Optional[str] = None
    run_dir: Optional[str] = None


def discover_results(base_dirs: Iterable[str]) -> List[str]:
    results: List[str] = []
    for base in base_dirs:
        if not base:
            continue
        base_path = Path(base)
        if not base_path.exists():
            continue
        for results_file in base_path.rglob("results.json"):
            if results_file.parent == base_path:
                continue
            results.append(str(results_file))
    return results


def load_metrics(results_file: str) -> Optional[RunMetrics]:
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Failed to load {results_file}: {exc}", file=sys.stderr)
        return None

    args = data.get("args", {})
    baseline = data.get("baseline", {})
    compressed = data.get("compressed", {})
    arc_split = data.get("arc_split") or args.get("arc_split")

    model_checkpoint = args.get("model_checkpoint")
    if args.get("no_bos_token", False):
        model_checkpoint = model_checkpoint + " NoBOS"

    return RunMetrics(
        model_checkpoint=model_checkpoint,
        baseline_token_accuracy=baseline.get("token_normalized_accuracy"),
        compressed_token_accuracy=compressed.get("token_normalized_accuracy"),
        arc_split=arc_split,
        run_dir=str(Path(results_file).parent),
    )


def pick_best_run(runs: List[RunMetrics]) -> Optional[RunMetrics]:
    if not runs:
        return None

    def key_fn(run: RunMetrics) -> float:
        val = run.compressed_token_accuracy
        return float(val) if val is not None else float("-inf")

    best = max(runs, key=key_fn)
    if best.compressed_token_accuracy is None and best.baseline_token_accuracy is None:
        return None
    return best


def to_percentage_cell(val: Optional[float]) -> str:
    if val is None:
        return ""
    return f"{val * 100:.2f}{{\\small %}}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize semantic benchmark token accuracies.")
    parser.add_argument(
        "--hs-dirs",
        nargs="*",
        default=["artifacts/hellaswag_evaluation"],
        help="Base directories to scan for HellaSwag runs.",
    )
    parser.add_argument(
        "--arc-dirs",
        nargs="*",
        default=["artifacts/arc_evaluation"],
        help="Base directories to scan for ARC runs.",
    )
    parser.add_argument(
        "--arc-split",
        type=str,
        default=None,
        help="Optional ARC split filter (ARC-Easy or ARC-Challenge).",
    )
    parser.add_argument(
        "--tablefmt",
        type=str,
        default="github",
        help="Tabulate table format (e.g., github, grid, latex, plain).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If set, write the rendered table to <save-dir>/semantic_evaluation.tex.",
    )
    args = parser.parse_args()

    hs_results = []
    for results_file in discover_results(args.hs_dirs):
        metrics = load_metrics(results_file)
        print("results_file", results_file, "metrics", metrics)
        if metrics is not None:
            hs_results.append(metrics)

    arc_results = []
    for results_file in discover_results(args.arc_dirs):
        print("results_file", results_file, "metrics", metrics)
        metrics = load_metrics(results_file)
        if metrics is None:
            continue
        if args.arc_split is not None and (metrics.arc_split or "").lower() != args.arc_split.lower():
            continue
        arc_results.append(metrics)

    hs_by_model = {}
    for run in hs_results:
        model = run.model_checkpoint or "unknown"
        hs_by_model.setdefault(model, []).append(run)

    arc_by_model = {}
    for run in arc_results:
        model = run.model_checkpoint or "unknown"
        arc_by_model.setdefault(model, []).append(run)

    all_models = sorted(set(hs_by_model.keys()) | set(arc_by_model.keys()))

    rows = []
    model_mapping = {
        "Meta-Llama-3.1-8B": "Llama-3.1-8B",
        "Meta-Llama-3.1-8B NoBOS": "Llama-3.1-8B NoBOS",
        "SmolLM2-1.7B": "SLM2-1.7B",
        "SmolLM2-1.7B NoBOS": "SLM2-1.7B NoBOS",
        # # "Meta-Llama-3.1-8B": "L-3.1-8B",
        # "pythia-1.4b": "p-1.4b",
        # "pythia-1.4b NoBOS": "p-1.4b NoBOS",
        "gemma-3-4b-pt": "gemma-3-4b",
        "gemma-3-4b-pt NoBOS": "gemma-3-4b NoBOS",
    }

    for model in all_models:
        hs_best = pick_best_run(hs_by_model.get(model, []))
        arc_best = pick_best_run(arc_by_model.get(model, []))
        model_slug = model.split("/")[-1]
        model_slug = model_mapping.get(model_slug, model_slug)
        model_slug = model_slug.replace(" NoBOS", " \\bcancel{B}")
        rows.append(
            [
                "\\small " + model_slug,
                to_percentage_cell(hs_best.baseline_token_accuracy if hs_best else None),
                to_percentage_cell(hs_best.compressed_token_accuracy if hs_best else None),
                to_percentage_cell(arc_best.baseline_token_accuracy if arc_best else None),
                to_percentage_cell(arc_best.compressed_token_accuracy if arc_best else None),
            ]
        )

    headers = [
        "Model",
        "Base",
        "Cram",
        "Base",
        "Cram",
    ]
    result = tabulate(rows, headers=headers, tablefmt=args.tablefmt)
    result = result.split("\n")
    result.insert(
        2, "                   & \\multicolumn{2}{c}{\\textbf{HellaSwag}} & \\multicolumn{2}{c}{\\textbf{ARC-E}} \\\\"
    )
    result = "\n".join(result)
    result = result.replace("\\textbackslash{}", "\\")
    result = result.replace("\\$", "$")
    result = result.replace("\\{", "{")
    result = result.replace("\\}", "}")

    print(result)

    if args.save_dir is not None:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "semantic_evaluation.tex"
        out_path.write_text(result + "\n", encoding="utf-8")
        print(f"\nSaved 'tab:semantic_evaluation' to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
