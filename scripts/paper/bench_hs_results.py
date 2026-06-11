#!/usr/bin/env python3
"""
Results aggregation and LaTeX table printer for HellaSwag evaluation experiments.

This script scans experiment artifact folders produced by hellaswag_compress_evaluate.py
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
class HSRunSummary:
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
    - hellaswag_{model}_samples_{N}_lr_{lr}_batch_{B}_loss_{loss}_hybrid_{alpha}_align_{L}
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

    # Remove "hellaswag_" prefix
    name = run_name.replace("hellaswag_", "")

    # Extract model (everything until "_samples_")
    m_model = re.search(r"^([^_]+(?:_[^_]+)*?)_samples_", name)
    if m_model:
        model = m_model.group(1)
        # Replace common model name patterns
        model = model.replace("Meta-Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B")
        model = model.replace("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B")
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
        "Qwen/Qwen3-4B": "Q3-4B",
        "unsloth/Llama-3.2-3B": "L3.2-3B",
        "unsloth/Llama-3.2-1B": "L3.2-1B",
        "unsloth/Meta-Llama-3.1-8B": "L3.1-8B",
        "allenai/OLMo-1B-hf": "OLM-1B",
        "allenai/Olmo-3-1025-7B": "OLM3-7B",
    },
}


def aggregate_results(results_file: str) -> Optional[HSRunSummary]:
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

    summary = HSRunSummary(
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


def build_latex_table(
    summaries: List[HSRunSummary], selected_columns: Optional[List[str]] = None, tablefmt: str = "latex_raw"
) -> str:
    """
    Build a LaTeX tabular with key properties and metrics using tabulate.

    Args:
        summaries: List of HSRunSummary objects to include in the table
        selected_columns: Optional list of column field names to include. If None, includes all columns.
    """
    # Columns for properties
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
    ]
    # Metric columns
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
                row.append(latex_escape(cell))
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


# ----------------------- Intervention Knockout Plots ----------------------- #


def load_intervention_results(results_path: str) -> Dict:
    """Load a results.json that contains intervention_summary."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "intervention_summary" not in data:
        raise ValueError(f"No intervention_summary in {results_path}")
    return data


def load_attention_mass(attention_mass_path: str) -> List[float]:
    """Load per-layer attention mass from a cache JSON file.

    Supports two formats:
    - List of floats (one per layer): direct per-layer attention mass %
    - Dict with 'avg_attention_mass_per_layer_compression' key: list of floats
    """
    with open(attention_mass_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [float(v) for v in data]
    if isinstance(data, dict) and "avg_attention_mass_per_layer_compression" in data:
        return [float(v) for v in data["avg_attention_mass_per_layer_compression"]]
    raise ValueError(f"Unrecognized attention mass format in {attention_mass_path}")


def plot_per_layer_knockout(
    data: Dict,
    output_path: str,
    attention_mass: Optional[List[float]] = None,
    model_label: Optional[str] = None,
) -> None:
    """Per-layer knockout plot: x=layer, y1=accuracy with knockout at that layer, y2=attention mass.

    Also overlays teacher-forced reconstruction accuracy if available.

    Args:
        data: Loaded intervention results JSON.
        output_path: Path to save the plot.
        attention_mass: Optional per-layer attention mass (% values).
        model_label: Optional label for the model.
    """
    summary = data["intervention_summary"]
    if "per_layer_knockout" not in summary:
        print("No per_layer_knockout data found, skipping plot.", file=sys.stderr)
        return

    per_layer = summary["per_layer_knockout"]
    num_layers = data.get("num_model_layers", len(per_layer))
    layers = list(range(num_layers))
    accuracies = [per_layer[str(li)]["accuracy"] for li in layers]

    # Reconstruction accuracy (teacher-forced prefix reconstruction)
    per_layer_recon = summary.get("per_layer_reconstruction")
    recon_accuracies = None
    if per_layer_recon is not None:
        recon_accuracies = [per_layer_recon[str(li)]["avg_accuracy"] for li in layers]

    base_acc = data.get("baseline", {}).get("accuracy")
    cram_acc = data.get("compressed", {}).get("accuracy")

    fig, ax1 = plt.subplots(figsize=(12, 4))

    color_acc = "#2563eb"
    ax1.plot(layers, accuracies, "o-", color=color_acc, linewidth=1.5, markersize=4, label="KO accuracy")
    if recon_accuracies is not None:
        color_recon = "#a855f7"
        ax1.plot(
            layers,
            recon_accuracies,
            "s-",
            color=color_recon,
            linewidth=1.5,
            markersize=4,
            label="Teacher-Forcing reconstruction accuracy",
        )
    if base_acc is not None:
        ax1.axhline(y=base_acc, color="#16a34a", linestyle="--", linewidth=1, label=f"Base = {base_acc:.3f}")
    if cram_acc is not None:
        ax1.axhline(y=cram_acc, color="#dc2626", linestyle="--", linewidth=1, label=f"Cram = {cram_acc:.3f}")
    ax1.set_xlabel("Layer index", fontsize=18)
    ax1.set_ylabel("Accuracy (KO at layer)", color=color_acc, fontsize=18)
    ax1.tick_params(axis="y", labelcolor=color_acc, labelsize=15)
    ax1.tick_params(axis="x", labelsize=15)
    ax1.set_xlim(-0.5, num_layers - 0.5)

    if attention_mass is not None and len(attention_mass) == num_layers:
        color_attn = "#f97316"
        ax2 = ax1.twinx()
        ax2.bar(layers, attention_mass, alpha=0.3, color=color_attn, label="Attention mass %")
        ax2.set_ylabel("Attention mass on\ncompression token (%)", color=color_attn, fontsize=18)
        ax2.tick_params(axis="y", labelcolor=color_attn, labelsize=15)
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(
            lines2 + lines1,
            labels2 + labels1,
            loc="upper right",
            bbox_to_anchor=(1.0, 0.63),
            fontsize=14,
            framealpha=0.9,
        )
    else:
        ax1.legend(loc="upper right", fontsize=14, framealpha=1.0)

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-layer knockout plot to {output_path}")


def plot_scatter_attention_vs_recovery(
    data: Dict,
    output_path: str,
    attention_mass: List[float],
    model_label: Optional[str] = None,
) -> None:
    """Scatter plot: x=attention mass at layer l, y=accuracy delta from Cram baseline.

    Args:
        data: Loaded intervention results JSON.
        output_path: Path to save the plot.
        attention_mass: Per-layer attention mass (% values). Required.
        model_label: Optional label for the model.
    """
    summary = data["intervention_summary"]
    if "per_layer_knockout" not in summary:
        print("No per_layer_knockout data found, skipping scatter plot.", file=sys.stderr)
        return

    per_layer = summary["per_layer_knockout"]
    num_layers = data.get("num_model_layers", len(per_layer))
    cram_acc = data.get("compressed", {}).get("accuracy", 0.0)

    if len(attention_mass) != num_layers:
        print(
            f"Attention mass length ({len(attention_mass)}) != num_layers ({num_layers}), skipping.",
            file=sys.stderr,
        )
        return

    x_mass = np.array(attention_mass)
    y_delta = np.array([per_layer[str(li)]["accuracy"] - cram_acc for li in range(num_layers)])

    fig, ax = plt.subplots(figsize=(8, 6))

    title = "Attention Mass vs. Accuracy Recovery"
    if model_label:
        title += f" ({model_label})"

    ax.scatter(x_mass, y_delta, c=np.arange(num_layers), cmap="viridis", s=80, edgecolors="k", linewidths=0.5)

    # Add layer labels to a few notable points
    for li in range(num_layers):
        ax.annotate(str(li), (x_mass[li], y_delta[li]), fontsize=16, ha="center", va="bottom", alpha=0.7)

    # Fit and plot trend line
    if np.std(x_mass) > 1e-8 and np.std(y_delta) > 1e-8:
        z = np.polyfit(x_mass, y_delta, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_mass.min(), x_mass.max(), 100)
        ax.plot(x_line, p(x_line), "--", color="red", linewidth=1, alpha=0.7, label=f"Linear fit (slope={z[0]:.4f})")
        corr = np.corrcoef(x_mass, y_delta)[0, 1]
        ax.set_title(f"{title}\nPearson r = {corr:.3f}")
        ax.legend(fontsize=8)
    else:
        ax.set_title(title)

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Attention mass on compression token (%)")
    ax.set_ylabel("Accuracy delta from Cram baseline")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot to {output_path}")


def plot_cumulative_knockout(
    data: Dict,
    output_path: str,
    model_label: Optional[str] = None,
    vertical: bool = False,
) -> None:
    """Cumulative knockout curve: x=number of layers knocked out (0..L), y=accuracy.

    Plots both forward (layers 0..li) and reverse (layers li..L-1) cumulative
    knockout on the same chart when both are available. Also overlays teacher-forced
    reconstruction accuracy if available.

    Args:
        data: Loaded intervention results JSON.
        output_path: Path to save the plot.
        model_label: Optional label for the model.
    """
    summary = data["intervention_summary"]
    has_forward = "cumulative_knockout" in summary
    has_reverse = "reverse_cumulative_knockout" in summary

    if not has_forward and not has_reverse:
        print("No cumulative_knockout data found, skipping plot.", file=sys.stderr)
        return

    num_layers = data.get("num_model_layers")
    if num_layers is None:
        if has_forward:
            num_layers = len(summary["cumulative_knockout"])
        else:
            num_layers = len(summary["reverse_cumulative_knockout"])

    base_acc = data.get("baseline", {}).get("accuracy")
    cram_acc = data.get("compressed", {}).get("accuracy")

    if vertical:
        fig, (ax_fwd, ax_rev) = plt.subplots(2, 1, figsize=(8, 8), sharey=True)
    else:
        fig, (ax_fwd, ax_rev) = plt.subplots(1, 2, figsize=(16, 4), sharey=True)

    # --- Left subplot: Forward cumulative knockout (layers 0..k) ---
    if has_forward:
        cumulative = summary["cumulative_knockout"]
        x_fwd = [0] + [li + 1 for li in range(num_layers)]
        y_fwd = [cram_acc if cram_acc is not None else 0.0] + [cumulative[str(li)]["accuracy"] for li in range(num_layers)]
        ax_fwd.plot(
            x_fwd,
            y_fwd,
            "o-",
            color="#2563eb",
            linewidth=1.5,
            markersize=4,
            label="Forward knockout HellaSwag Accuracy (layers 0..k)",
        )

    if "cumulative_reconstruction" in summary:
        cum_recon = summary["cumulative_reconstruction"]
        x_fwd_r = [0] + [li + 1 for li in range(num_layers)]
        y_fwd_r = [1.0] + [cum_recon[str(li)]["avg_accuracy"] for li in range(num_layers)]
        ax_fwd.plot(
            x_fwd_r,
            y_fwd_r,
            "o--",
            color="#60a5fa",
            linewidth=1.5,
            markersize=4,
            alpha=0.8,
            label="Forward Teacher-Forcing reconstruction Accuracy",
        )

    if base_acc is not None:
        ax_fwd.axhline(y=base_acc, color="#16a34a", linestyle="--", linewidth=1, label=f"Base = {base_acc:.3f}")
    if cram_acc is not None:
        ax_fwd.axhline(y=cram_acc, color="#dc2626", linestyle="--", linewidth=1, label=f"Cram = {cram_acc:.3f}")
    ax_fwd.set_xlabel("Number of layers knocked out (0 = Cram, L = Base)", fontsize=18)
    ax_fwd.set_ylabel("Accuracy", fontsize=18)
    ax_fwd.set_xlim(-0.5, num_layers + 0.5)
    ax_fwd.set_title("Forward knockout HellaSwag Accuracy (layers 0..k)", fontsize=18)
    ax_fwd.tick_params(labelsize=15)
    ax_fwd.legend(loc="best", fontsize=14)
    ax_fwd.grid(True, alpha=0.3)

    # --- Right subplot: Reverse cumulative knockout (layers k..L-1) ---
    if has_reverse:
        reverse_cumulative = summary["reverse_cumulative_knockout"]
        x_rev = [0] + [num_layers - li for li in range(num_layers - 1, -1, -1)]
        y_rev = [cram_acc if cram_acc is not None else 0.0] + [
            reverse_cumulative[str(li)]["accuracy"] for li in range(num_layers - 1, -1, -1)
        ]
        ax_rev.plot(x_rev, y_rev, "s-", color="#9333ea", linewidth=1.5, markersize=4, label="Reverse knockout (layers k..L-1)")

    if "reverse_cumulative_reconstruction" in summary:
        rev_recon = summary["reverse_cumulative_reconstruction"]
        x_rev_r = [0] + [num_layers - li for li in range(num_layers - 1, -1, -1)]
        y_rev_r = [1.0] + [rev_recon[str(li)]["avg_accuracy"] for li in range(num_layers - 1, -1, -1)]
        ax_rev.plot(
            x_rev_r,
            y_rev_r,
            "s--",
            color="#c084fc",
            linewidth=1.5,
            markersize=4,
            alpha=0.8,
            label="Reverse Teacher-Forcing reconstruction Acc",
        )

    if base_acc is not None:
        ax_rev.axhline(y=base_acc, color="#16a34a", linestyle="--", linewidth=1, label=f"Base = {base_acc:.3f}")
    if cram_acc is not None:
        ax_rev.axhline(y=cram_acc, color="#dc2626", linestyle="--", linewidth=1, label=f"Cram = {cram_acc:.3f}")
    ax_rev.set_xlabel("Number of layers knocked out (0 = Cram, L = Base)", fontsize=18)
    if vertical:
        ax_rev.set_ylabel("Accuracy", fontsize=18)
    ax_rev.set_xlim(-0.5, num_layers + 0.5)
    ax_rev.set_title("Reverse knockout (layers k..L-1)", fontsize=18)
    ax_rev.tick_params(labelsize=15)
    ax_rev.legend(loc="best", fontsize=14)
    ax_rev.grid(True, alpha=0.3)

    fig.tight_layout()
    if vertical:
        fig.subplots_adjust(hspace=0.55)
    fig.savefig(output_path, dpi=200 if vertical else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cumulative knockout plot to {output_path}")


def run_intervention_plots(
    results_path: str,
    attention_mass_path: Optional[str] = None,
    model_label: Optional[str] = None,
) -> None:
    """Generate all intervention knockout plots and save to the same dir as results_path."""
    data = load_intervention_results(results_path)
    out = Path(results_path).parent
    out.mkdir(parents=True, exist_ok=True)

    if model_label is None:
        model_label = data.get("args", {}).get("model_checkpoint", "")

    # Get attention mass: prefer external file, fall back to results.json embedded data
    attention_mass = None
    if attention_mass_path:
        attention_mass = load_attention_mass(attention_mass_path)
    elif "avg_attention_mass_per_layer" in data.get("intervention_summary", {}):
        attention_mass = data["intervention_summary"]["avg_attention_mass_per_layer"]

    summary = data.get("intervention_summary", {})

    if "per_layer_knockout" in summary:
        plot_per_layer_knockout(
            data, str(out / "per_layer_knockout.png"), attention_mass=attention_mass, model_label=model_label
        )

    if "per_layer_knockout" in summary and attention_mass is not None:
        plot_scatter_attention_vs_recovery(
            data, str(out / "scatter_attention_vs_recovery.png"), attention_mass, model_label=model_label
        )

    if "cumulative_knockout" in summary:
        plot_cumulative_knockout(data, str(out / "cumulative_knockout.png"), model_label=model_label)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hs_results.py",
        description="Aggregate HellaSwag evaluation experiment artifacts and print a LaTeX table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            By default scans:
              - artifacts/hellaswag_evaluation/*/results.json
            """
        ),
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=["artifacts/hellaswag_evaluation"],
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

    # Intervention knockout plotting
    parser.add_argument(
        "--plot-intervention",
        type=str,
        default=None,
        metavar="RESULTS_JSON",
        help="Path to a results.json with intervention_summary. Generates knockout plots and exits.",
    )
    parser.add_argument(
        "--attention-mass-file",
        type=str,
        default=None,
        metavar="JSON",
        help="Optional attention mass cache JSON for overlay on per-layer plot and scatter plot.",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default=None,
        help="Model label for plot titles. If omitted, uses model_checkpoint from results args.",
    )
    args = parser.parse_args(argv)

    # If --plot-intervention is given, generate plots and exit
    if args.plot_intervention:
        run_intervention_plots(
            results_path=args.plot_intervention,
            attention_mass_path=args.attention_mass_file,
            model_label=args.model_label,
        )
        return 0

    results_files = discover_run_results(args.dirs)
    if not results_files:
        print("No experiment results found. Searched:", file=sys.stderr)
        for d in args.dirs:
            print(f" - {d}", file=sys.stderr)
        return 1

    summaries: List[HSRunSummary] = []
    for results_file in tqdm(results_files, desc="Processing Runs"):
        try:
            summary = aggregate_results(results_file)
        except Exception as e:
            print(f"Failed to process {results_file}: {e}", file=sys.stderr)
            continue
        if summary is None:
            continue
        summaries.append(summary)

        # Auto-generate intervention plots if results contain intervention data
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if "intervention_summary" in raw:
                run_intervention_plots(
                    results_path=results_file,
                    attention_mass_path=args.attention_mass_file,
                    model_label=args.model_label,
                )
        except Exception as e:
            print(f"Failed to generate intervention plots for {results_file}: {e}", file=sys.stderr)

    # Sort for readability
    def sort_key(s: HSRunSummary):
        return (
            s.model_checkpoint or "",
            str(s.loss_type or ""),
            float(s.hybrid_alpha or 0),
            int(s.num_alignment_layers or 0),
            float(s.learning_rate or 0),
            int(s.batch_size or 0),
        )

    summaries_sorted = sorted(summaries, key=sort_key)

    # Apply property filters
    def matches_filters(s: HSRunSummary) -> bool:
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
