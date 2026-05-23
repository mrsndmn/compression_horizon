#!/usr/bin/env python3
"""
Print a table of token-normalized accuracies for HellaSwag and ARC.

Rows: model checkpoints
Columns per benchmark: Base | Progressive Acc | Progressive Conv% | Full Acc | Full Conv%

Where:
- Base = baseline.token_normalized_accuracy (unchanged, all samples)
- Progressive / Full Acc = sum(is_correct & is_fully_converged) / sum(is_fully_converged)
  (accuracy among samples that reached strict convergence == 1.0)
- Progressive / Full Conv% = sum(is_fully_converged) / total_samples
- Mode (Progressive vs Full) detected from args.progressive in each run's results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional

from tabulate import tabulate

Mode = Literal["full", "progressive"]


@dataclass
class RunMetrics:
    model_checkpoint: Optional[str]
    mode: Mode
    baseline_token_accuracy: Optional[float]
    compressed_token_accuracy: Optional[float]  # legacy: all-samples accuracy from top-level aggregate
    converged_pct: Optional[float]  # fraction of samples with is_fully_converged == True
    converged_accuracy: Optional[float]  # accuracy over converged subset
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


def _detect_mode(args_dict: dict) -> Mode:
    return "progressive" if bool(args_dict.get("progressive", False)) else "full"


def _aggregate_converged(per_sample: List[dict]) -> tuple[Optional[float], Optional[float]]:
    """Return (converged_pct, converged_accuracy) from per-sample results.

    converged_pct = sum(is_fully_converged) / total
    converged_accuracy = sum(is_correct & is_fully_converged) / sum(is_fully_converged)
    Returns (None, None) if per-sample data lacks the required fields.

    Supports both per-sample schemas in this repo:
      - HellaSwag: r["compressed"]["is_fully_converged"], r["compressed"]["is_correct"]
      - ARC:       r["is_fully_converged"] (top-level), r["compression"]["is_correct"]
    """
    if not per_sample:
        return None, None
    has_field = False
    n_total = 0
    n_conv = 0
    n_correct_conv = 0
    for r in per_sample:
        if not isinstance(r, dict):
            continue
        comp = r.get("compressed")
        if not isinstance(comp, dict):
            comp = r.get("compression")
        if not isinstance(comp, dict):
            continue
        # is_fully_converged may live at top-level (ARC) or nested under comp (HellaSwag)
        is_conv = r.get("is_fully_converged")
        if is_conv is None:
            is_conv = comp.get("is_fully_converged")
        if is_conv is None:
            continue
        has_field = True
        n_total += 1
        if is_conv:
            n_conv += 1
            if comp.get("is_correct"):
                n_correct_conv += 1
    if not has_field or n_total == 0:
        return None, None
    converged_pct = n_conv / n_total
    converged_accuracy = (n_correct_conv / n_conv) if n_conv > 0 else None
    return converged_pct, converged_accuracy


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

    mode = _detect_mode(args)
    converged_pct, converged_accuracy = _aggregate_converged(data.get("results", []))

    return RunMetrics(
        model_checkpoint=model_checkpoint,
        mode=mode,
        baseline_token_accuracy=baseline.get("token_normalized_accuracy"),
        compressed_token_accuracy=compressed.get("token_normalized_accuracy"),
        converged_pct=converged_pct,
        converged_accuracy=converged_accuracy,
        arc_split=arc_split,
        run_dir=str(Path(results_file).parent),
    )


def pick_best_run(runs: List[RunMetrics]) -> Optional[RunMetrics]:
    """Pick best run by (converged_accuracy desc, converged_pct desc, compressed_token_accuracy desc)."""
    if not runs:
        return None

    def key_fn(run: RunMetrics) -> tuple:
        ca = run.converged_accuracy if run.converged_accuracy is not None else float("-inf")
        cp = run.converged_pct if run.converged_pct is not None else float("-inf")
        cta = run.compressed_token_accuracy if run.compressed_token_accuracy is not None else float("-inf")
        return (ca, cp, cta)

    best = max(runs, key=key_fn)
    if best.converged_accuracy is None and best.compressed_token_accuracy is None and best.baseline_token_accuracy is None:
        return None
    return best


def to_percentage_cell(val: Optional[float]) -> str:
    if val is None:
        return "--"
    return f"{val * 100:.2f}{{\\small %}}"


def _split_by_mode(runs: List[RunMetrics]) -> dict[Mode, List[RunMetrics]]:
    out: dict[Mode, List[RunMetrics]] = {"full": [], "progressive": []}
    for r in runs:
        out[r.mode].append(r)
    return out


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

    hs_results: List[RunMetrics] = []
    for results_file in discover_results(args.hs_dirs):
        metrics = load_metrics(results_file)
        if metrics is not None:
            hs_results.append(metrics)

    arc_results: List[RunMetrics] = []
    for results_file in discover_results(args.arc_dirs):
        metrics = load_metrics(results_file)
        if metrics is None:
            continue
        if args.arc_split is not None and (metrics.arc_split or "").lower() != args.arc_split.lower():
            continue
        arc_results.append(metrics)

    hs_by_model: dict[str, List[RunMetrics]] = {}
    for run in hs_results:
        model = run.model_checkpoint or "unknown"
        hs_by_model.setdefault(model, []).append(run)

    arc_by_model: dict[str, List[RunMetrics]] = {}
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
        "gemma-3-4b-pt": "gemma-3-4b",
        "gemma-3-4b-pt NoBOS": "gemma-3-4b NoBOS",
    }

    for model in all_models:
        hs_by_mode = _split_by_mode(hs_by_model.get(model, []))
        arc_by_mode = _split_by_mode(arc_by_model.get(model, []))

        hs_full = pick_best_run(hs_by_mode["full"])
        hs_prog = pick_best_run(hs_by_mode["progressive"])
        arc_full = pick_best_run(arc_by_mode["full"])
        arc_prog = pick_best_run(arc_by_mode["progressive"])

        # Baseline is mode-agnostic; prefer full's baseline if both exist, else progressive's.
        hs_base = hs_full or hs_prog
        arc_base = arc_full or arc_prog

        model_slug = model.split("/")[-1]
        model_slug = model_mapping.get(model_slug, model_slug)
        model_slug = model_slug.replace(" NoBOS", " \\bcancel{B}")

        rows.append(
            [
                "\\small " + model_slug,
                to_percentage_cell(hs_base.baseline_token_accuracy if hs_base else None),
                to_percentage_cell(hs_prog.converged_accuracy if hs_prog else None),
                to_percentage_cell(hs_prog.converged_pct if hs_prog else None),
                to_percentage_cell(hs_full.converged_accuracy if hs_full else None),
                to_percentage_cell(hs_full.converged_pct if hs_full else None),
                to_percentage_cell(arc_base.baseline_token_accuracy if arc_base else None),
                to_percentage_cell(arc_prog.converged_accuracy if arc_prog else None),
                to_percentage_cell(arc_prog.converged_pct if arc_prog else None),
                to_percentage_cell(arc_full.converged_accuracy if arc_full else None),
                to_percentage_cell(arc_full.converged_pct if arc_full else None),
            ]
        )

    headers = [
        "Model",
        "Base",
        "Prog Acc",
        "Prog Conv\\%",
        "Full Acc",
        "Full Conv\\%",
        "Base",
        "Prog Acc",
        "Prog Conv\\%",
        "Full Acc",
        "Full Conv\\%",
    ]
    result = tabulate(rows, headers=headers, tablefmt=args.tablefmt)

    if args.tablefmt == "latex":
        result_lines = result.split("\n")
        # Insert a multicolumn header row (with cline rules underneath) above the column headers.
        # tabulate latex emits:
        #   \begin{tabular}{...}      (index 0)
        #   \hline                    (index 1)
        #   Model & Base & ... \\     (index 2)
        # We insert the group-header row at index 2 and a cline separator immediately after,
        # so the rendered table has a rule under each group label.
        result_lines.insert(
            2,
            " & \\multicolumn{5}{c}{\\textbf{HellaSwag}} " "& \\multicolumn{5}{c}{\\textbf{ARC-E}} \\\\",
        )
        result_lines.insert(3, "\\cline{2-6}\\cline{7-11}")
        result = "\n".join(result_lines)
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
