#!/usr/bin/env python3
"""
Results aggregation and LaTeX table printer for MMLU evaluation experiments.

This script scans experiment artifact folders produced by mmlu_compress_evaluate.py
and builds a LaTeX table with:
- experiment properties (model, loss type, hybrid alpha, etc.)
- metrics (baseline accuracy, compressed accuracy, difference)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from tqdm.auto import tqdm

# ------------------------------- Utilities --------------------------------- #


@dataclass
class MMLURunSummary:
    # Identifiers
    run_dir: str
    run_name: str
    # Properties (from args and/or run_dir name)
    model_checkpoint: Optional[str] = None
    limit_samples: Optional[int] = None
    num_compression_tokens: Optional[int] = None
    max_optimization_steps: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    dtype: Optional[str] = None
    loss_type: Optional[str] = None
    hybrid_alpha: Optional[float] = None
    num_alignment_layers: Optional[int] = None
    inverted_alignment: Optional[bool] = None
    random_seed: Optional[int] = None
    # MMLU-specific properties
    subject: Optional[str] = None
    num_few_shot: Optional[int] = None
    compression_mode: Optional[str] = None
    # Metrics
    baseline_accuracy: Optional[float] = None
    baseline_token_accuracy: Optional[float] = None
    baseline_char_accuracy: Optional[float] = None
    baseline_correct: Optional[int] = None
    baseline_valid_pct: Optional[float] = None
    baseline_total: Optional[int] = None
    compressed_accuracy: Optional[float] = None
    compressed_token_accuracy: Optional[float] = None
    compressed_char_accuracy: Optional[float] = None
    compressed_correct: Optional[int] = None
    compressed_valid_pct: Optional[float] = None
    compressed_total: Optional[int] = None
    accuracy_difference: Optional[float] = None
    token_accuracy_difference: Optional[float] = None
    char_accuracy_difference: Optional[float] = None
    num_subjects: Optional[int] = None


LATEX_ESCAPE_MAP = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def latex_escape(text: Optional[str]) -> str:
    if text is None:
        return ""
    # Escape backslashes first so they are not affected by later replacements.
    sentinel = "LATEXBACKSLASHSENTINEL"
    escaped = str(text).replace("\\", sentinel)
    out = []
    for ch in escaped:
        out.append(LATEX_ESCAPE_MAP.get(ch, ch))
    return "".join(out).replace(sentinel, LATEX_ESCAPE_MAP["\\"])


def parse_run_name_for_properties(run_name: str) -> Dict[str, Optional[str]]:
    """
    Parse fields from run directory name:
    - mmlu_{model}_samples_{N}_lr_{lr}_batch_{B}_loss_{loss}_hybrid_{alpha}_align_{L}
    """
    props: Dict[str, Optional[str]] = {
        "model_checkpoint": None,
        "limit_samples": None,
        "learning_rate": None,
        "batch_size": None,
        "loss_type": None,
        "hybrid_alpha": None,
        "num_alignment_layers": None,
    }

    # Remove "mmlu_" prefix
    name = run_name.replace("mmlu_", "")

    # Extract model (everything until "_samples_")
    m_model = re.search(r"^([^_]+(?:_[^_]+)*?)_samples_", name)
    if m_model:
        model = m_model.group(1)
        model = model.replace("Meta-Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B")
        model = model.replace("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B")
        props["model_checkpoint"] = model

    m_samples = re.search(r"_samples_([0-9]+)", name)
    if m_samples:
        props["limit_samples"] = int(m_samples.group(1))

    m_lr = re.search(r"_lr_([0-9.]+)", name)
    if m_lr:
        props["learning_rate"] = float(m_lr.group(1))

    m_batch = re.search(r"_batch_([0-9]+)", name)
    if m_batch:
        props["batch_size"] = int(m_batch.group(1))

    m_loss = re.search(r"_loss_([^_]+)", name)
    if m_loss:
        props["loss_type"] = m_loss.group(1)

    m_hybrid = re.search(r"_hybrid_([0-9.]+)", name)
    if m_hybrid:
        props["hybrid_alpha"] = float(m_hybrid.group(1))

    m_align = re.search(r"_align_([0-9]+)", name)
    if m_align:
        props["num_alignment_layers"] = int(m_align.group(1))

    return props


def discover_run_results(base_dirs: Iterable[str]) -> List[str]:
    """
    Return list of results.json file paths.
    """
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


abbreviation = {
    "loss_type": {
        "cosine": "cos",
        "cross_entropy": "CE",
    },
    "model_checkpoint": {
        "HuggingFaceTB/SmolLM2-1.7B": "SLM2-1.7B",
        "HuggingFaceTB/SmolLM2-135M": "SLM2-135M",
        "Qwen/Qwen3-4B": "Q3-4B",
        "unsloth/Llama-3.2-3B": "L3.2-3B",
        "unsloth/Llama-3.2-1B": "L3.2-1B",
        "unsloth/Meta-Llama-3.1-8B": "L3.1-8B",
        "allenai/OLMo-1B-hf": "OLM-1B",
        "allenai/Olmo-3-1025-7B": "OLM3-7B",
    },
}


def aggregate_results(results_file: str) -> Optional[MMLURunSummary]:
    """
    Load and aggregate results from a results.json file.
    """
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {results_file}: {e}", file=sys.stderr)
        return None

    args = data.get("args", {})
    baseline = data.get("baseline", {})
    compressed = data.get("compressed", {})
    per_subject = data.get("per_subject", {})

    run_dir = str(Path(results_file).parent)
    run_name = Path(results_file).parent.name

    parsed = parse_run_name_for_properties(run_name)

    model_checkpoint = args.get("model_checkpoint") or parsed.get("model_checkpoint")
    if model_checkpoint in abbreviation.get("model_checkpoint", {}):
        model_checkpoint = abbreviation["model_checkpoint"][model_checkpoint]

    loss_type = args.get("loss_type") or parsed.get("loss_type")
    if loss_type in abbreviation.get("loss_type", {}):
        loss_type = abbreviation["loss_type"][loss_type]

    baseline_accuracy = baseline.get("accuracy")
    baseline_token_accuracy = baseline.get("token_normalized_accuracy")
    baseline_char_accuracy = baseline.get("char_normalized_accuracy")
    compressed_accuracy = compressed.get("accuracy")
    compressed_token_accuracy = compressed.get("token_normalized_accuracy")
    compressed_char_accuracy = compressed.get("char_normalized_accuracy")
    accuracy_difference = None
    if baseline_accuracy is not None and compressed_accuracy is not None:
        accuracy_difference = compressed_accuracy - baseline_accuracy
    token_accuracy_difference = None
    if baseline_token_accuracy is not None and compressed_token_accuracy is not None:
        token_accuracy_difference = compressed_token_accuracy - baseline_token_accuracy
    char_accuracy_difference = None
    if baseline_char_accuracy is not None and compressed_char_accuracy is not None:
        char_accuracy_difference = compressed_char_accuracy - baseline_char_accuracy

    baseline_valid = baseline.get("valid_predictions")
    baseline_total_pred = baseline.get("total_predictions")
    baseline_valid_pct = baseline_valid / baseline_total_pred if baseline_valid is not None and baseline_total_pred else None

    compressed_valid = compressed.get("valid_predictions")
    compressed_total_pred = compressed.get("total_predictions")
    compressed_valid_pct = (
        compressed_valid / compressed_total_pred if compressed_valid is not None and compressed_total_pred else None
    )

    summary = MMLURunSummary(
        run_dir=run_dir,
        run_name=run_name,
        model_checkpoint=model_checkpoint,
        limit_samples=args.get("limit_samples") or parsed.get("limit_samples"),
        num_compression_tokens=args.get("num_compression_tokens"),
        max_optimization_steps=args.get("max_optimization_steps"),
        learning_rate=args.get("learning_rate") or parsed.get("learning_rate"),
        batch_size=args.get("batch_size") or parsed.get("batch_size"),
        dtype=args.get("dtype"),
        loss_type=loss_type,
        hybrid_alpha=args.get("hybrid_alpha") or parsed.get("hybrid_alpha"),
        num_alignment_layers=args.get("num_alignment_layers") or parsed.get("num_alignment_layers"),
        inverted_alignment=args.get("inverted_alignment"),
        random_seed=args.get("random_seed"),
        subject=args.get("subject"),
        num_few_shot=args.get("num_few_shot"),
        compression_mode=args.get("compression_mode"),
        baseline_accuracy=baseline_accuracy,
        baseline_token_accuracy=baseline_token_accuracy,
        baseline_char_accuracy=baseline_char_accuracy,
        baseline_correct=baseline.get("correct_predictions"),
        baseline_valid_pct=baseline_valid_pct,
        baseline_total=baseline.get("total_predictions"),
        compressed_accuracy=compressed_accuracy,
        compressed_token_accuracy=compressed_token_accuracy,
        compressed_char_accuracy=compressed_char_accuracy,
        compressed_correct=compressed.get("correct_predictions"),
        compressed_valid_pct=compressed_valid_pct,
        compressed_total=compressed.get("total_predictions"),
        accuracy_difference=accuracy_difference,
        token_accuracy_difference=token_accuracy_difference,
        char_accuracy_difference=char_accuracy_difference,
        num_subjects=len(per_subject) if per_subject else None,
    )
    return summary


def to_percentage_cell(val: Optional[float]) -> str:
    if val is None:
        return ""
    return f"{val * 100:.2f}%"


def to_float_cell(val: Optional[float], decimals: int = 4) -> str:
    if val is None:
        return ""
    return f"{val:.{decimals}f}".rstrip("0").rstrip(".")


def build_latex_table(
    summaries: List[MMLURunSummary], selected_columns: Optional[List[str]] = None, tablefmt: str = "latex_raw"
) -> str:
    """
    Build a LaTeX tabular with key properties and metrics using tabulate.

    Args:
        summaries: List of MMLURunSummary objects to include in the table
        selected_columns: Optional list of column field names to include. If None, includes all columns.
    """
    prop_cols = [
        ("run_name", "RunName"),
        ("model_checkpoint", "Model"),
        ("limit_samples", "Samples"),
        ("num_compression_tokens", "MemT"),
        ("max_optimization_steps", "MaxSteps"),
        ("learning_rate", "LR"),
        ("batch_size", "Batch"),
        ("dtype", "DType"),
        ("loss_type", "Loss"),
        ("hybrid_alpha", "Hybrid $\\alpha$"),
        ("num_alignment_layers", "AlignLayers"),
        ("inverted_alignment", "InvAlign"),
        ("random_seed", "Seed"),
        ("subject", "Subject"),
        ("num_few_shot", "FewShot"),
        ("compression_mode", "CompMode"),
        ("num_subjects", "Subjects"),
    ]
    metric_cols = [
        ("baseline_accuracy", "Baseline Acc"),
        ("baseline_valid_pct", "Baseline Valid"),
        ("baseline_token_accuracy", "Baseline Tok Acc"),
        ("baseline_char_accuracy", "Baseline Char Acc"),
        ("compressed_accuracy", "Compressed Acc"),
        ("compressed_valid_pct", "Compressed Valid"),
        ("compressed_token_accuracy", "Compressed Tok Acc"),
        ("compressed_char_accuracy", "Compressed Char Acc"),
        ("accuracy_difference", "Diff"),
        ("token_accuracy_difference", "Tok Diff"),
        ("char_accuracy_difference", "Char Diff"),
    ]

    all_cols = prop_cols + metric_cols
    col_lookup = {field: hdr for field, hdr in all_cols}

    if selected_columns is not None:
        # Preserve CLI argument order
        cols = [(field, col_lookup[field]) for field in selected_columns if field in col_lookup]
    else:
        cols = all_cols

    percentage_fields = {
        "baseline_accuracy",
        "baseline_valid_pct",
        "baseline_token_accuracy",
        "baseline_char_accuracy",
        "compressed_accuracy",
        "compressed_valid_pct",
        "compressed_token_accuracy",
        "compressed_char_accuracy",
        "accuracy_difference",
        "token_accuracy_difference",
        "char_accuracy_difference",
    }

    headers = [hdr for _, hdr in cols]

    table_rows: List[List[str]] = []
    for s in summaries:
        row: List[str] = []
        for field_name, _hdr in cols:
            val = getattr(s, field_name)
            if field_name in percentage_fields:
                row.append(to_percentage_cell(val))
            elif isinstance(val, bool):
                row.append("True" if val else "False")
            else:
                cell = "" if val is None else str(val)
                row.append(cell)
                # row.append(latex_escape(cell))
        table_rows.append(row)

    result = tabulate(table_rows, headers=headers, tablefmt=tablefmt)
    # Use booktabs rules (\toprule/\midrule/\bottomrule) instead of tabulate's \hline.
    lines = result.split("\n")
    rule_idx = [i for i, ln in enumerate(lines) if ln.strip() == "\\hline"]
    for n, i in enumerate(rule_idx):
        if n == 0:
            lines[i] = "\\toprule"
        elif n == len(rule_idx) - 1:
            lines[i] = "\\bottomrule"
        else:
            lines[i] = "\\midrule"
    return "\n".join(lines)


# ----------------------- Per-Subject Breakdown Plot ------------------------ #


def plot_per_subject_comparison(
    results_path: str,
    output_path: Optional[str] = None,
    model_label: Optional[str] = None,
) -> None:
    """Bar chart comparing baseline vs compressed accuracy per MMLU subject."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_subject = data.get("per_subject", {})
    if not per_subject:
        print("No per_subject data found, skipping plot.", file=sys.stderr)
        return

    if output_path is None:
        output_path = str(Path(results_path).parent / "per_subject_comparison.png")

    if model_label is None:
        model_label = data.get("args", {}).get("model_checkpoint", "")

    # Sort subjects by baseline accuracy descending
    subjects = sorted(per_subject.keys(), key=lambda s: per_subject[s].get("baseline_accuracy", 0), reverse=True)
    baseline_accs = [per_subject[s].get("baseline_accuracy", 0) for s in subjects]
    compressed_accs = [per_subject[s].get("compressed_accuracy", 0) for s in subjects]

    x = np.arange(len(subjects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(subjects) * 0.4), 6))

    ax.bar(x - width / 2, baseline_accs, width, label="Baseline", color="#2563eb", alpha=0.8)
    ax.bar(x + width / 2, compressed_accs, width, label="Compressed", color="#dc2626", alpha=0.8)

    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy")
    title = "MMLU Per-Subject: Baseline vs Compressed"
    if model_label:
        title += f" ({model_label})"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in subjects], rotation=90, fontsize=6)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-subject comparison plot to {output_path}")


def plot_accuracy_difference_distribution(
    results_path: str,
    output_path: Optional[str] = None,
    model_label: Optional[str] = None,
) -> None:
    """Histogram of per-subject accuracy difference (compressed - baseline)."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_subject = data.get("per_subject", {})
    if not per_subject:
        print("No per_subject data found, skipping plot.", file=sys.stderr)
        return

    if output_path is None:
        output_path = str(Path(results_path).parent / "accuracy_difference_dist.png")

    if model_label is None:
        model_label = data.get("args", {}).get("model_checkpoint", "")

    diffs = []
    for s, stats in per_subject.items():
        b = stats.get("baseline_accuracy", 0)
        c = stats.get("compressed_accuracy", 0)
        diffs.append(c - b)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(diffs, bins=20, color="#2563eb", alpha=0.7, edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
    mean_diff = np.mean(diffs)
    ax.axvline(x=mean_diff, color="#16a34a", linestyle="--", linewidth=1, label=f"Mean = {mean_diff:.4f}")

    ax.set_xlabel("Accuracy Difference (Compressed - Baseline)")
    ax.set_ylabel("Number of Subjects")
    title = "MMLU Per-Subject Accuracy Difference Distribution"
    if model_label:
        title += f"\n({model_label})"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved accuracy difference distribution plot to {output_path}")


# ------------------------------- Main -------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mmlu_results.py",
        description="Aggregate MMLU evaluation experiment artifacts and print a LaTeX table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            By default scans:
              - artifacts/mmlu_evaluation/*/results.json
            """
        ),
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=["artifacts/mmlu_evaluation"],
        help="Base directories to scan for runs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the LaTeX table; prints to stdout otherwise.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Specify which columns to include in the table. Available columns: "
        "run_name, model_checkpoint, limit_samples, num_compression_tokens, "
        "max_optimization_steps, learning_rate, batch_size, dtype, loss_type, "
        "hybrid_alpha, num_alignment_layers, inverted_alignment, random_seed, "
        "subject, num_few_shot, compression_mode, num_subjects, "
        "baseline_accuracy, baseline_token_accuracy, baseline_char_accuracy, "
        "compressed_accuracy, compressed_token_accuracy, compressed_char_accuracy, "
        "accuracy_difference, token_accuracy_difference, char_accuracy_difference. "
        "If not specified, all columns are included.",
    )
    parser.add_argument(
        "--tablefmt",
        type=str,
        default="latex_raw",
        help="Table format passed to tabulate (e.g., latex_raw, github, grid, pipe, plain, html, tsv, etc.).",
    )

    # Property filters
    def _parse_bool(x: str) -> bool:
        val = x.strip().lower()
        if val in {"1", "true", "t", "yes", "y"}:
            return True
        if val in {"0", "false", "f", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")

    parser.add_argument(
        "--loss-type", type=str, default=None, help="Filter by loss type (e.g., l2, l1, cosine, cross_entropy)."
    )
    parser.add_argument("--hybrid-alpha", type=float, default=None, help="Filter by hybrid alpha value (float).")
    parser.add_argument("--model", type=str, default=None, help="Filter by model checkpoint substring (case-insensitive).")
    parser.add_argument("--samples", type=int, default=None, help="Filter by limit_samples (int).")
    parser.add_argument("--mem-tokens", type=int, default=None, help="Filter by number of compression tokens (int).")
    parser.add_argument("--max-steps", type=int, default=None, help="Filter by max optimization steps (int).")
    parser.add_argument("--learning-rate", type=float, default=None, help="Filter by learning rate (float).")
    parser.add_argument("--batch-size", type=int, default=None, help="Filter by batch size (int).")
    parser.add_argument("--dtype", type=str, default=None, help="Filter by dtype (e.g., float32, float16, bfloat16).")
    parser.add_argument(
        "--align-layers",
        type=int,
        default=None,
        help="Filter by number of alignment layers (int).",
    )
    parser.add_argument(
        "--inverted-alignment",
        type=_parse_bool,
        default=None,
        help="Filter by inverted_alignment. Accepts: true/false, 1/0, yes/no, t/f, y/n (case-insensitive).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Filter by random seed (int).")
    parser.add_argument("--subject-filter", type=str, default=None, help="Filter by MMLU subject (e.g., 'abstract_algebra').")
    parser.add_argument(
        "--compression-mode", type=str, default=None, help="Filter by compression mode (prefix_only, full_prompt, random)."
    )
    parser.add_argument("--num-few-shot", type=int, default=None, help="Filter by number of few-shot examples (int).")

    # Plotting
    parser.add_argument(
        "--plot-subjects",
        type=str,
        default=None,
        metavar="RESULTS_JSON",
        help="Path to a results.json with per_subject data. Generates per-subject comparison plots and exits.",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default=None,
        help="Model label for plot titles. If omitted, uses model_checkpoint from results args.",
    )
    args = parser.parse_args(argv)

    # If --plot-subjects is given, generate plots and exit
    if args.plot_subjects:
        plot_per_subject_comparison(
            results_path=args.plot_subjects,
            model_label=args.model_label,
        )
        plot_accuracy_difference_distribution(
            results_path=args.plot_subjects,
            model_label=args.model_label,
        )
        return 0

    results_files = discover_run_results(args.dirs)
    if not results_files:
        print("No experiment results found. Searched:", file=sys.stderr)
        for d in args.dirs:
            print(f" - {d}", file=sys.stderr)
        return 1

    summaries: List[MMLURunSummary] = []
    for results_file in tqdm(results_files, desc="Processing Runs"):
        try:
            summary = aggregate_results(results_file)
        except Exception as e:
            print(f"Failed to process {results_file}: {e}", file=sys.stderr)
            continue
        if summary is None:
            continue
        summaries.append(summary)

        # Auto-generate per-subject plots if data is present
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if "per_subject" in raw and len(raw["per_subject"]) > 1:
                plot_per_subject_comparison(
                    results_path=results_file,
                    model_label=args.model_label,
                )
                plot_accuracy_difference_distribution(
                    results_path=results_file,
                    model_label=args.model_label,
                )
        except Exception as e:
            print(f"Failed to generate plots for {results_file}: {e}", file=sys.stderr)

    # Sort for readability
    def sort_key(s: MMLURunSummary):
        return (
            s.model_checkpoint or "",
            str(s.loss_type or ""),
            str(s.compression_mode or ""),
            float(s.hybrid_alpha or 0),
            int(s.num_alignment_layers or 0),
            float(s.learning_rate or 0),
            int(s.batch_size or 0),
        )

    summaries_sorted = sorted(summaries, key=sort_key)

    # Apply property filters
    def matches_filters(s: MMLURunSummary) -> bool:
        if args.loss_type is not None and (s.loss_type or "").lower() != args.loss_type.lower():
            return False
        if args.hybrid_alpha is not None:
            if s.hybrid_alpha is None or abs(float(s.hybrid_alpha) - float(args.hybrid_alpha)) > 1e-6:
                return False
        if args.model is not None:
            model_val = (s.model_checkpoint or "").lower()
            if args.model.lower() not in model_val:
                return False
        if args.samples is not None and (s.limit_samples is None or int(s.limit_samples) != int(args.samples)):
            return False
        if args.mem_tokens is not None and (
            s.num_compression_tokens is None or int(s.num_compression_tokens) != int(args.mem_tokens)
        ):
            return False
        if args.max_steps is not None and (
            s.max_optimization_steps is None or int(s.max_optimization_steps) != int(args.max_steps)
        ):
            return False
        if args.learning_rate is not None:
            if s.learning_rate is None or abs(float(s.learning_rate) - float(args.learning_rate)) > 1e-6:
                return False
        if args.batch_size is not None and (s.batch_size is None or int(s.batch_size) != int(args.batch_size)):
            return False
        if args.dtype is not None and (s.dtype or "").lower() != args.dtype.lower():
            return False
        if args.align_layers is not None and (
            s.num_alignment_layers is None or int(s.num_alignment_layers) != int(args.align_layers)
        ):
            return False
        if args.inverted_alignment is not None and (
            s.inverted_alignment is None or bool(s.inverted_alignment) != bool(args.inverted_alignment)
        ):
            return False
        if args.seed is not None and (s.random_seed is None or int(s.random_seed) != int(args.seed)):
            return False
        if args.subject_filter is not None and (s.subject or "").lower() != args.subject_filter.lower():
            return False
        if args.compression_mode is not None and (s.compression_mode or "").lower() != args.compression_mode.lower():
            return False
        if args.num_few_shot is not None and (s.num_few_shot is None or int(s.num_few_shot) != int(args.num_few_shot)):
            return False
        return True

    summaries_sorted = [s for s in summaries_sorted if matches_filters(s)]
    latex = build_latex_table(summaries_sorted, selected_columns=args.columns, tablefmt=args.tablefmt)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex)
        print(f"Wrote LaTeX table to {args.output}")
    else:
        print(latex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
