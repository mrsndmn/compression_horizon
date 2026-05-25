#!/usr/bin/env python3
"""
Results aggregation and LaTeX table printer for ARC evaluation experiments.

This script scans experiment artifact folders produced by arc_compress_evaluate.py
and builds a LaTeX table with:
- experiment properties (model, arc_split, loss type, hybrid alpha, etc.)
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

from tabulate import tabulate
from tqdm.auto import tqdm

# ------------------------------- Utilities --------------------------------- #


@dataclass
class ARCRunSummary:
    # Identifiers
    run_dir: str
    run_name: str
    # Properties (from args and/or run_dir name)
    model_checkpoint: Optional[str] = None
    arc_split: Optional[str] = None
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
    # Metrics
    baseline_accuracy: Optional[float] = None
    baseline_token_accuracy: Optional[float] = None
    baseline_char_accuracy: Optional[float] = None
    baseline_correct: Optional[int] = None
    baseline_total: Optional[int] = None
    compressed_accuracy: Optional[float] = None
    compressed_token_accuracy: Optional[float] = None
    compressed_char_accuracy: Optional[float] = None
    compressed_correct: Optional[int] = None
    compressed_total: Optional[int] = None
    accuracy_difference: Optional[float] = None
    token_accuracy_difference: Optional[float] = None
    char_accuracy_difference: Optional[float] = None


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
    out = []
    for ch in str(text):
        out.append(LATEX_ESCAPE_MAP.get(ch, ch))
    return "".join(out)


def parse_run_name_for_properties(run_name: str) -> Dict[str, Optional[str]]:
    """
    Parse fields from run directory name:
    - arc_{split}_{model}_samples_{N}_lr_{lr}_batch_{B}_loss_{loss}_hybrid_{alpha}_align_{L}
    """
    props: Dict[str, Optional[str]] = {
        "arc_split": None,
        "model_checkpoint": None,
        "limit_samples": None,
        "learning_rate": None,
        "batch_size": None,
        "loss_type": None,
        "hybrid_alpha": None,
        "num_alignment_layers": None,
    }

    # Remove "arc_" prefix
    name = run_name.replace("arc_", "")

    # Extract split (ARC_Easy or ARC_Challenge) - comes first after "arc_"
    m_split = re.search(r"^(ARC_Easy|ARC_Challenge)_", name)
    if m_split:
        split = m_split.group(1)
        # Convert back to original format
        props["arc_split"] = split.replace("_", "-")
        # Remove split from name for further parsing
        name = name[len(split) + 1 :]

    # Extract model (everything until "_samples_" or end if no samples)
    m_model = re.search(
        r"^([^_]+(?:_[^_]+)*?)(?:_samples_|_seed_|_tokens_|_steps_|_lr_|_batch_|_dtype_|_loss_|_hybrid_|_align_|$)", name
    )
    if m_model:
        model = m_model.group(1)
        # Replace common model name patterns
        model = model.replace("Meta-Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B")
        model = model.replace("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B")
        model = model.replace("SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M")
        model = model.replace("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M")
        props["model_checkpoint"] = model

    # Extract samples
    m_samples = re.search(r"_samples_([0-9]+)", name)
    if m_samples:
        props["limit_samples"] = int(m_samples.group(1))

    # Extract learning rate
    m_lr = re.search(r"_lr_([0-9.]+)", name)
    if m_lr:
        props["learning_rate"] = float(m_lr.group(1))

    # Extract batch size
    m_batch = re.search(r"_batch_([0-9]+)", name)
    if m_batch:
        props["batch_size"] = int(m_batch.group(1))

    # Extract loss type
    m_loss = re.search(r"_loss_([^_]+)", name)
    if m_loss:
        props["loss_type"] = m_loss.group(1)

    # Extract hybrid alpha
    m_hybrid = re.search(r"_hybrid_([0-9.]+)", name)
    if m_hybrid:
        props["hybrid_alpha"] = float(m_hybrid.group(1))

    # Extract alignment layers
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
        # Look for results.json files in subdirectories
        for results_file in base_path.rglob("results.json"):
            # Skip if it's in the root directory (might be a summary file)
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
        "HuggingFaceTB/SmolLM2-360M": "SLM2-360M",
        "Qwen/Qwen3-4B": "Q3-4B",
        "unsloth/Llama-3.2-3B": "L3.2-3B",
        "unsloth/Llama-3.2-1B": "L3.2-1B",
        "unsloth/Meta-Llama-3.1-8B": "L3.1-8B",
        "allenai/OLMo-1B-hf": "OLM-1B",
        "allenai/Olmo-3-1025-7B": "OLM3-7B",
    },
    "arc_split": {
        "ARC-Easy": "Easy",
        "ARC-Challenge": "Challenge",
    },
}


def aggregate_results(results_file: str) -> Optional[ARCRunSummary]:
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

    run_dir = str(Path(results_file).parent)
    run_name = Path(results_file).parent.name

    # Parse properties from run name
    parsed = parse_run_name_for_properties(run_name)

    # Get properties from args (args take precedence)
    arc_split = args.get("arc_split") or parsed.get("arc_split")
    if arc_split in abbreviation.get("arc_split", {}):
        arc_split = abbreviation["arc_split"][arc_split]

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

    summary = ARCRunSummary(
        run_dir=run_dir,
        run_name=run_name,
        model_checkpoint=model_checkpoint,
        arc_split=arc_split,
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
        baseline_accuracy=baseline_accuracy,
        baseline_token_accuracy=baseline_token_accuracy,
        baseline_char_accuracy=baseline_char_accuracy,
        baseline_correct=baseline.get("correct_predictions"),
        baseline_total=baseline.get("total_predictions"),
        compressed_accuracy=compressed_accuracy,
        compressed_token_accuracy=compressed_token_accuracy,
        compressed_char_accuracy=compressed_char_accuracy,
        compressed_correct=compressed.get("correct_predictions"),
        compressed_total=compressed.get("total_predictions"),
        accuracy_difference=accuracy_difference,
        token_accuracy_difference=token_accuracy_difference,
        char_accuracy_difference=char_accuracy_difference,
    )
    return summary


def to_percentage_cell(val: Optional[float]) -> str:
    if val is None:
        return ""
    return f"{val * 100:.2f}\\%"


def to_float_cell(val: Optional[float], decimals: int = 4) -> str:
    if val is None:
        return ""
    return f"{val:.{decimals}f}".rstrip("0").rstrip(".")


def build_latex_table(summaries: List[ARCRunSummary], selected_columns: Optional[List[str]] = None) -> str:
    """
    Build a LaTeX tabular with key properties and metrics using tabulate.

    Args:
        summaries: List of ARCRunSummary objects to include in the table
        selected_columns: Optional list of column field names to include. If None, includes all columns.
    """
    # Columns for properties
    prop_cols = [
        ("run_name", "RunName"),
        ("arc_split", "Split"),
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
    ]
    # Metric columns
    metric_cols = [
        ("baseline_accuracy", "Baseline Acc"),
        ("baseline_token_accuracy", "Baseline Tok Acc"),
        ("baseline_char_accuracy", "Baseline Char Acc"),
        ("compressed_accuracy", "Compressed Acc"),
        ("compressed_token_accuracy", "Compressed Tok Acc"),
        ("compressed_char_accuracy", "Compressed Char Acc"),
        ("accuracy_difference", "Diff"),
        ("token_accuracy_difference", "Tok Diff"),
        ("char_accuracy_difference", "Char Diff"),
    ]

    # Filter columns if selected_columns is provided
    if selected_columns is not None:
        selected_set = set(selected_columns)
        prop_cols = [col for col in prop_cols if col[0] in selected_set]
        metric_cols = [col for col in metric_cols if col[0] in selected_set]

    headers = [hdr for _, hdr in prop_cols] + [hdr for _, hdr in metric_cols]

    table_rows: List[List[str]] = []
    for s in summaries:
        row: List[str] = []
        # Properties
        for field_name, _hdr in prop_cols:
            val = getattr(s, field_name)
            if isinstance(val, bool):
                cell = "True" if val else "False"
            else:
                cell = "" if val is None else str(val)
            row.append(latex_escape(cell))
        # Metrics
        for field_name, _hdr in metric_cols:
            val = getattr(s, field_name)
            if field_name in (
                "baseline_accuracy",
                "baseline_token_accuracy",
                "baseline_char_accuracy",
                "compressed_accuracy",
                "compressed_token_accuracy",
                "compressed_char_accuracy",
                "accuracy_difference",
                "token_accuracy_difference",
                "char_accuracy_difference",
            ):
                row.append(to_percentage_cell(val))
            else:
                row.append(to_float_cell(val))
        table_rows.append(row)

    # Use latex_raw to respect our own escaping and math cells
    result = tabulate(table_rows, headers=headers, tablefmt="latex_raw")
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="arc_results.py",
        description="Aggregate ARC evaluation experiment artifacts and print a LaTeX table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            By default scans:
              - artifacts/arc_evaluation/*/results.json
            """
        ),
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=["artifacts/arc_evaluation"],
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
        nargs="*",
        default=None,
        help="Specify which columns to include in the table. Available columns: "
        "run_name, arc_split, model_checkpoint, limit_samples, num_compression_tokens, "
        "max_optimization_steps, learning_rate, batch_size, dtype, loss_type, "
        "hybrid_alpha, num_alignment_layers, inverted_alignment, random_seed, "
        "baseline_accuracy, baseline_token_accuracy, baseline_char_accuracy, "
        "compressed_accuracy, compressed_token_accuracy, compressed_char_accuracy, "
        "accuracy_difference, token_accuracy_difference, char_accuracy_difference. "
        "If not specified, all columns are included.",
    )

    # Property filters
    def _parse_bool(x: str) -> bool:
        val = x.strip().lower()
        if val in {"1", "true", "t", "yes", "y"}:
            return True
        if val in {"0", "false", "f", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")

    parser.add_argument("--arc-split", type=str, default=None, help="Filter by ARC split (ARC-Easy or ARC-Challenge).")
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
    args = parser.parse_args(argv)

    results_files = discover_run_results(args.dirs)
    if not results_files:
        print("No experiment results found. Searched:", file=sys.stderr)
        for d in args.dirs:
            print(f" - {d}", file=sys.stderr)
        return 1

    summaries: List[ARCRunSummary] = []
    for results_file in tqdm(results_files, desc="Processing Runs"):
        try:
            summary = aggregate_results(results_file)
        except Exception as e:
            print(f"Failed to process {results_file}: {e}", file=sys.stderr)
            continue
        if summary is None:
            continue
        summaries.append(summary)

    # Sort for readability
    def sort_key(s: ARCRunSummary):
        return (
            s.arc_split or "",
            s.model_checkpoint or "",
            str(s.loss_type or ""),
            float(s.hybrid_alpha or 0),
            int(s.num_alignment_layers or 0),
            float(s.learning_rate or 0),
            int(s.batch_size or 0),
        )

    summaries_sorted = sorted(summaries, key=sort_key)

    # Apply property filters
    def matches_filters(s: ARCRunSummary) -> bool:
        if args.arc_split is not None and (s.arc_split or "").lower() != args.arc_split.lower():
            return False
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
        return True

    summaries_sorted = [s for s in summaries_sorted if matches_filters(s)]
    latex = build_latex_table(summaries_sorted, selected_columns=args.columns)

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
