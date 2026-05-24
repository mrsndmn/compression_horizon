#!/usr/bin/env python3
"""
Results aggregation and table printer for compression_horizon experiments.

This script scans experiment artifact folders produced by MyTrainer (see
src/compression_horizon/train/trainer.py) and builds a tabulate table with:
- experiment properties (loss type, init, seq len, etc.)
- metrics averaged over samples with standard deviation (mean ± std)

Supported artifact layouts:
- Non-progressive runs in: artifacts/experiments/<run_name>/compressed_prefixes
- Progressive runs in:     artifacts/experiments_progressive/<run_name>/progressive_prefixes
- Prefix tuning runs in:   artifacts/experiments_prefix_tuning/<run_name>/prefix_tuning_prefixes
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from datasets import load_from_disk
from tabulate import tabulate
from tqdm.auto import tqdm

from compression_horizon.utils import hlines_to_booktabs, to_mean_std_cell

# ------------------------------- Utilities --------------------------------- #


@dataclass
class RunSummary:
    # Identifiers
    run_dir: str
    run_hash: str
    dataset_type: str  # "compressed_prefixes" | "progressive_prefixes" | "prefix_tuning_prefixes"
    # Properties (best-effort from saved rows and/or run_dir name)
    dtype: Optional[str] = None
    loss_type: Optional[str] = None
    hybrid_alpha: Optional[str] = None
    embedding_init_method: Optional[str] = None
    max_sequence_length: Optional[int] = None
    number_of_compressed_tokens: Optional[float] = None
    number_of_compressed_tokens_std: Optional[float] = None
    num_alignment_layers: Optional[int] = None
    inverted_alignment: Optional[bool] = None
    fix_position_ids: Optional[bool] = None
    model_checkpoint: Optional[str] = None
    max_optimization_steps_per_sample: Optional[int] = None
    learning_rate: Optional[str] = None
    low_dim_size: Optional[str] = None
    num_alignment_layers: Optional[str] = None
    # Metrics (aggregated across samples)
    convergence_after_steps_mean: Optional[float] = None
    convergence_after_steps_std: Optional[float] = None
    final_convergence_mean: Optional[float] = None
    final_convergence_std: Optional[float] = None
    final_loss_mean: Optional[float] = None
    final_loss_std: Optional[float] = None
    information_gain_bits_mean: Optional[float] = None
    information_gain_bits_std: Optional[float] = None
    # Progressive-only optional extras
    steps_taken_mean: Optional[float] = None
    steps_taken_std: Optional[float] = None
    convergence_threshold: Optional[float] = None


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
    Parse fields injected in activation_distillation.py:
    - Non-progressive: ch_{loss}_hybrid_alpha_{alpha}_init_{init}_seq_len_{L}_{suffix}
    - Progressive:     ch_{loss}_init_{init}_seq_len_{L}_{suffix}
    - Prefix tuning:   ch_prefix_tuning_{loss}_hybrid_alpha_{alpha}_init_{init}_seq_len_{L}
    - Prefix tuning (exp_suffix): pt_sl_{L}_{model_name} (from run_jobs_prefix_tuning.py)
    """
    props: Dict[str, Optional[str]] = {
        "loss_type": None,
        "hybrid_alpha": None,
        "embedding_init_method": None,
        "max_sequence_length": None,
    }
    # Handle prefix tuning format: ch_prefix_tuning_{loss}_hybrid_alpha_{alpha}_init_{init}_seq_len_{L}
    if run_name.startswith("ch_prefix_tuning_"):
        # Extract loss_type (everything between ch_prefix_tuning_ and _hybrid_alpha_)
        m_loss = re.search(r"ch_prefix_tuning_(.+?)_hybrid_alpha_", run_name)
        if m_loss:
            props["loss_type"] = m_loss.group(1)
        # Extract hybrid_alpha (everything between _hybrid_alpha_ and _init_)
        m_alpha = re.search(r"_hybrid_alpha_(.+?)_init_", run_name)
        if m_alpha:
            props["hybrid_alpha"] = m_alpha.group(1)
        # Extract embedding_init_method (everything between _init_ and _seq_len_)
        m_init = re.search(r"_init_(.+?)_seq_len_", run_name)
        if m_init:
            props["embedding_init_method"] = m_init.group(1)
        # Extract max_sequence_length
        m_len = re.search(r"_seq_len_([0-9]+)", run_name)
        if m_len:
            props["max_sequence_length"] = int(m_len.group(1))
        return props
    # Handle prefix tuning exp_suffix format: pt_sl_{max_seq_len}_{model_name}
    elif run_name.startswith("pt_sl_"):
        m_len = re.search(r"pt_sl_([0-9]+)", run_name)
        if m_len:
            props["max_sequence_length"] = int(m_len.group(1))
        # Return early since exp_suffix format doesn't have other fields in the name
        return props
    else:
        # Generic patterns for non-progressive and progressive
        m_loss = re.search(r"ch_([^_]+)", run_name)
        if m_loss:
            props["loss_type"] = m_loss.group(1)
        m_alpha = re.search(r"hybrid_alpha_([^_]+)", run_name)
        if m_alpha:
            props["hybrid_alpha"] = m_alpha.group(1)
        m_init = re.search(r"init_((neg_)?random(_norm_)?\d?(\.\d+)?|mvnormal|mean|single_random)", run_name)
        if m_init:
            props["embedding_init_method"] = m_init.group(1)
        else:
            print("Failed to parse init method:", run_name)

        m_len = re.search(r"seq_len_([0-9]+)", run_name)
        if m_len:
            props["max_sequence_length"] = int(m_len.group(1))
    return props


def parse_cmd_args(run_dir: str) -> Dict[str, str]:
    cmd_path = Path(run_dir) / "cmd.txt"
    if not cmd_path.exists():
        return {}
    content = cmd_path.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    tokens = shlex.split(content)
    parsed: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            val = "True"
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                val = tokens[i + 1]
                i += 1
            parsed[key] = val
        i += 1
    return parsed


def discover_run_datasets(base_dirs: Iterable[str]) -> List[Tuple[str, str]]:
    """
    Return list of tuples: (dataset_path, dataset_type)
    dataset_type in {"compressed_prefixes", "progressive_prefixes", "prefix_tuning_prefixes"}
    """
    results: List[Tuple[str, str]] = []
    for base in base_dirs:
        if not base:
            continue
        base_path = Path(base)
        if not base_path.exists():
            continue
        for run_dir in sorted([p for p in base_path.iterdir() if p.is_dir()]):
            # Look for known dataset subfolders
            for ds_type in ("compressed_prefixes", "progressive_prefixes", "prefix_tuning_prefixes"):
                ds_path = run_dir / ds_type
                # A HF dataset saved_to_disk has a "dataset_info.json" or "state.json" and an "arrow" subdir
                if ds_path.exists() and ds_path.is_dir():
                    results.append((str(ds_path), ds_type))
    return results


def safe_mean(values: List[float]) -> Optional[float]:
    return mean(values) if values else None


def safe_std(values: List[float]) -> Optional[float]:
    # population std (to keep deterministic for small N), use pstdev
    if not values:
        return None
    if np.isnan(values[0]):
        return None
    return pstdev(values)


abbreviation = {
    "loss_type": {
        "cosine": "cos",
        "cross_entropy": "CE",
    },
    "embedding_init_method": {
        "mvnormal": "mvnorm",
        "random_norm_": "random_norm",
    },
    "model_checkpoint": {
        "HuggingFaceTB/SmolLM2-1.7B": "SLM2-1.7B",
        "HuggingFaceTB/SmolLM2-360M": "SLM2-360M",
        "HuggingFaceTB/SmolLM2-135M": "SLM2-135M",
        "Qwen/Qwen3-4B": "Q3-4B",
        "unsloth/Llama-3.2-3B": "L3.2-3B",
        "unsloth/Llama-3.2-1B": "L3.2-1B",
        "unsloth/Meta-Llama-3.1-8B": "L3.1-8B",
        "allenai/OLMo-1B-hf": "OLM-1B",
        "allenai/Olmo-3-1025-7B": "OLM3-7B",
        "EleutherAI/pythia-160m": "P-160m",
        "EleutherAI/pythia-410m": "P-410m",
        "EleutherAI/pythia-1.4b": "P-1.4b",
    },
}


def aggregate_non_progressive(run_dir: str, ds_rows: List[dict]) -> RunSummary:
    # Pull common properties – they should be constant within a run
    props_from_rows: Dict[str, Optional[object]] = {}
    for key in (
        "loss_type",
        "hybrid_alpha",
        "num_alignment_layers",
        "fix_position_ids",
        "model_checkpoint",
        "max_optimization_steps_per_sample",
        "num_compression_tokens",
        "dtype",
    ):
        val = None
        for r in ds_rows:
            if key in r:
                val = r[key]
                break
        props_from_rows[key] = val

    run_name = Path(run_dir).parent.name
    parsed = parse_run_name_for_properties(run_name)

    conv_steps = [r.get("convergence_after_steps") for r in ds_rows if r.get("convergence_after_steps") is not None]
    fin_conv = [r.get("final_convergence") for r in ds_rows if r.get("final_convergence") is not None]
    fin_loss = [r.get("final_loss") for r in ds_rows if r.get("final_loss") is not None]
    info_gain_bits = [r.get("information_gain_bits") for r in ds_rows if r.get("information_gain_bits") is not None]

    run_dir_parent = str(Path(run_dir).parent)
    cmd_args = parse_cmd_args(run_dir_parent)
    run_hash_file = os.path.join(run_dir_parent, "cmd_hash.txt")
    if not os.path.exists(run_hash_file):
        print("Can't find run hash file:", run_hash_file)
        return None

    with open(run_hash_file, "r") as hash_file:
        run_hash = hash_file.readline()

    loss_type = props_from_rows.get("loss_type") or parsed.get("loss_type")
    if loss_type in abbreviation["loss_type"]:
        loss_type = abbreviation["loss_type"][loss_type]

    embedding_init_method = parsed.get("embedding_init_method")
    if embedding_init_method in abbreviation.get("embedding_init_method", {}):
        embedding_init_method = abbreviation["embedding_init_method"][embedding_init_method]

    model_checkpoint = str(props_from_rows["model_checkpoint"]) if props_from_rows.get("model_checkpoint") is not None else None
    if model_checkpoint in abbreviation.get("model_checkpoint", {}):
        model_checkpoint = abbreviation["model_checkpoint"][model_checkpoint]

    summary = RunSummary(
        run_dir=run_dir_parent,
        run_hash=run_hash,
        dataset_type="compressed_prefixes",
        loss_type=loss_type,
        hybrid_alpha=str(
            props_from_rows.get("hybrid_alpha")
            if props_from_rows.get("hybrid_alpha") is not None
            else parsed.get("hybrid_alpha")
        ),
        dtype=(props_from_rows.get("dtype")),
        embedding_init_method=embedding_init_method,
        max_sequence_length=(int(parsed["max_sequence_length"]) if parsed.get("max_sequence_length") is not None else None),
        number_of_compressed_tokens=(
            int(parsed["max_sequence_length"]) if parsed.get("max_sequence_length") is not None else None
        ),
        number_of_compressed_tokens_std=None,
        # num_alignment_layers=(
        #     int(props_from_rows["num_alignment_layers"]) if props_from_rows.get("num_alignment_layers") is not None else None
        # ),
        inverted_alignment=None,  # not persisted in rows; unknown
        fix_position_ids=(
            bool(props_from_rows["fix_position_ids"]) if props_from_rows.get("fix_position_ids") is not None else None
        ),
        model_checkpoint=model_checkpoint,
        max_optimization_steps_per_sample=(
            int(props_from_rows["max_optimization_steps_per_sample"])
            if props_from_rows.get("max_optimization_steps_per_sample") is not None
            else None
        ),
        learning_rate=cmd_args.get("learning_rate"),
        low_dim_size=cmd_args.get("low_dim_size"),
        num_alignment_layers=cmd_args.get("num_alignment_layers"),
        convergence_after_steps_mean=safe_mean([float(x) for x in conv_steps]),
        convergence_after_steps_std=safe_std([float(x) for x in conv_steps]),
        final_convergence_mean=safe_mean([float(x) for x in fin_conv]),
        final_convergence_std=safe_std([float(x) for x in fin_conv]),
        final_loss_mean=safe_mean([float(x) for x in fin_loss]),
        final_loss_std=safe_std([float(x) for x in fin_loss]),
        information_gain_bits_mean=safe_mean([float(x) for x in info_gain_bits]),
        information_gain_bits_std=safe_std([float(x) for x in info_gain_bits]),
    )
    return summary


def aggregate_prefix_tuning(run_dir: str, ds_rows: List[dict]) -> RunSummary:
    """
    Aggregate prefix tuning runs. Similar to non-progressive but:
    - Uses num_virtual_tokens instead of num_compression_tokens
    - No convergence_after_steps (set to None)
    """
    # Pull common properties – they should be constant within a run
    props_from_rows: Dict[str, Optional[object]] = {}
    for key in (
        "loss_type",
        "hybrid_alpha",
        "num_alignment_layers",
        "fix_position_ids",
        "model_checkpoint",
        "max_optimization_steps_per_sample",
        "num_virtual_tokens",
        "dtype",
    ):
        val = None
        for r in ds_rows:
            if key in r:
                val = r[key]
                break
        props_from_rows[key] = val

    run_name = Path(run_dir).parent.name
    parsed = parse_run_name_for_properties(run_name)

    fin_conv = [r.get("final_convergence") for r in ds_rows if r.get("final_convergence") is not None]
    fin_loss = [r.get("final_loss") for r in ds_rows if r.get("final_loss") is not None]
    info_gain_bits = [r.get("information_gain_bits") for r in ds_rows if r.get("information_gain_bits") is not None]

    run_dir_parent = str(Path(run_dir).parent)
    cmd_args = parse_cmd_args(run_dir_parent)
    run_hash_file = os.path.join(run_dir_parent, "cmd_hash.txt")
    if not os.path.exists(run_hash_file):
        print("Can't find run hash file:", run_hash_file)
        return None

    with open(run_hash_file, "r") as hash_file:
        run_hash = hash_file.readline()

    loss_type = props_from_rows.get("loss_type") or parsed.get("loss_type")
    if loss_type in abbreviation["loss_type"]:
        loss_type = abbreviation["loss_type"][loss_type]

    embedding_init_method = parsed.get("embedding_init_method")
    if embedding_init_method in abbreviation.get("embedding_init_method", {}):
        embedding_init_method = abbreviation["embedding_init_method"][embedding_init_method]

    model_checkpoint = str(props_from_rows["model_checkpoint"]) if props_from_rows.get("model_checkpoint") is not None else None
    if model_checkpoint in abbreviation.get("model_checkpoint", {}):
        model_checkpoint = abbreviation["model_checkpoint"][model_checkpoint]

    summary = RunSummary(
        run_dir=run_dir_parent,
        run_hash=run_hash,
        dataset_type="prefix_tuning_prefixes",
        loss_type=loss_type,
        hybrid_alpha=str(
            props_from_rows.get("hybrid_alpha")
            if props_from_rows.get("hybrid_alpha") is not None
            else parsed.get("hybrid_alpha")
        ),
        dtype=(props_from_rows.get("dtype")),
        embedding_init_method=embedding_init_method,
        max_sequence_length=(int(parsed["max_sequence_length"]) if parsed.get("max_sequence_length") is not None else None),
        number_of_compressed_tokens=(
            int(parsed["max_sequence_length"]) if parsed.get("max_sequence_length") is not None else None
        ),
        number_of_compressed_tokens_std=None,
        # num_alignment_layers=(
        #     int(props_from_rows["num_alignment_layers"]) if props_from_rows.get("num_alignment_layers") is not None else None
        # ),
        inverted_alignment=None,  # not persisted in rows; unknown
        fix_position_ids=(
            bool(props_from_rows["fix_position_ids"]) if props_from_rows.get("fix_position_ids") is not None else None
        ),
        model_checkpoint=model_checkpoint,
        max_optimization_steps_per_sample=(
            int(props_from_rows["max_optimization_steps_per_sample"])
            if props_from_rows.get("max_optimization_steps_per_sample") is not None
            else None
        ),
        learning_rate=cmd_args.get("learning_rate"),
        low_dim_size=cmd_args.get("low_dim_size"),
        num_alignment_layers=cmd_args.get("num_alignment_layers"),
        convergence_after_steps_mean=None,  # N/A for prefix tuning
        convergence_after_steps_std=None,
        final_convergence_mean=safe_mean([float(x) for x in fin_conv]),
        final_convergence_std=safe_std([float(x) for x in fin_conv]),
        final_loss_mean=safe_mean([float(x) for x in fin_loss]),
        final_loss_std=safe_std([float(x) for x in fin_loss]),
        information_gain_bits_mean=safe_mean([float(x) for x in info_gain_bits]),
        information_gain_bits_std=safe_std([float(x) for x in info_gain_bits]),
    )
    return summary


def aggregate_progressive(run_dir: str, ds_rows: List[dict]) -> RunSummary:
    """
    For progressive runs, aggregate final stage per sample_id.
    We compute final_loss, final_convergence for the last stage and steps_taken stats.
    """
    # Properties from run name (progressive template lacks hybrid_alpha)
    run_name = Path(run_dir).parent.name
    parsed = parse_run_name_for_properties(run_name)

    # For progressive rows, group by sample_id and take the last stage_index
    by_sample: Dict[int, List[dict]] = {}
    for r in ds_rows:
        sid = int(r.get("sample_id"))
        by_sample.setdefault(sid, []).append(r)
    last_rows: List[dict] = []
    for sid, rows in by_sample.items():
        rows_sorted = sorted(rows, key=lambda x: int(x.get("stage_index", 0)))
        last_rows.append(rows_sorted[-1])

    # Collect stats
    fin_conv = [r.get("final_convergence") for r in last_rows if r.get("final_convergence") is not None]
    fin_loss = [r.get("final_loss") for r in last_rows if r.get("final_loss") is not None]
    steps_taken = [r.get("steps_taken") for r in last_rows if r.get("steps_taken") is not None]
    info_gain_bits = [r.get("information_gain_bits") for r in last_rows if r.get("information_gain_bits") is not None]
    num_embeddings = [len(rows) for rows in by_sample.values()]

    # Extract a few more properties if present in rows
    props_from_rows: Dict[str, Optional[object]] = {}
    for key in ("num_compression_tokens", "model_checkpoint", "max_optimization_steps_per_sample", "loss_type", "dtype"):
        val = None
        for r in ds_rows:
            if key in r:
                val = r[key]
                break
        props_from_rows[key] = val

    # Convergence threshold is constant
    cthr = None
    for r in ds_rows:
        if "convergence_threshold" in r:
            cthr = r["convergence_threshold"]
            break

    run_dir_parent = str(Path(run_dir).parent)
    cmd_args = parse_cmd_args(run_dir_parent)
    run_hash_file = os.path.join(run_dir_parent, "cmd_hash.txt")
    if not os.path.exists(run_hash_file):
        print("Can't find run hash file:", run_hash_file)
        return None

    with open(run_hash_file, "r") as hash_file:
        run_hash = hash_file.readline()

    loss_type = props_from_rows.get("loss_type") or parsed.get("loss_type")
    if loss_type in abbreviation.get("loss_type", {}):
        loss_type = abbreviation["loss_type"][loss_type]

    embedding_init_method = parsed.get("embedding_init_method")
    if embedding_init_method in abbreviation.get("embedding_init_method", {}):
        embedding_init_method = abbreviation["embedding_init_method"][embedding_init_method]

    model_checkpoint = str(props_from_rows["model_checkpoint"]) if props_from_rows.get("model_checkpoint") is not None else None
    if model_checkpoint in abbreviation.get("model_checkpoint", {}):
        model_checkpoint = abbreviation["model_checkpoint"][model_checkpoint]

    summary = RunSummary(
        run_dir=str(Path(run_dir).parent),
        run_hash=run_hash,
        dataset_type="progressive_prefixes",
        loss_type=loss_type,
        hybrid_alpha=None,
        embedding_init_method=embedding_init_method,
        dtype=(props_from_rows.get("dtype")),
        max_sequence_length=(int(parsed["max_sequence_length"]) if parsed.get("max_sequence_length") is not None else None),
        number_of_compressed_tokens=safe_mean([float(x) for x in num_embeddings]),
        number_of_compressed_tokens_std=safe_std([float(x) for x in num_embeddings]),
        # num_alignment_layers=None,
        inverted_alignment=None,
        fix_position_ids=None,
        model_checkpoint=model_checkpoint,
        max_optimization_steps_per_sample=(
            int(props_from_rows["max_optimization_steps_per_sample"])
            if props_from_rows.get("max_optimization_steps_per_sample") is not None
            else None
        ),
        learning_rate=cmd_args.get("learning_rate"),
        low_dim_size=cmd_args.get("low_dim_size"),
        num_alignment_layers=cmd_args.get("num_alignment_layers"),
        convergence_after_steps_mean=None,  # N/A for progressive
        convergence_after_steps_std=None,
        final_convergence_mean=safe_mean([float(x) for x in fin_conv]),
        final_convergence_std=safe_std([float(x) for x in fin_conv]),
        final_loss_mean=safe_mean([float(x) for x in fin_loss]),
        final_loss_std=safe_std([float(x) for x in fin_loss]),
        information_gain_bits_mean=safe_mean([float(x) for x in info_gain_bits]),
        information_gain_bits_std=safe_std([float(x) for x in info_gain_bits]),
        steps_taken_mean=safe_mean([float(x) for x in steps_taken]),
        steps_taken_std=safe_std([float(x) for x in steps_taken]),
        convergence_threshold=(float(cthr) if cthr is not None else None),
    )
    return summary


def is_latex_tablefmt(tablefmt: str) -> bool:
    return tablefmt.startswith("latex")


def build_latex_table(
    summaries: List[RunSummary],
    include_progressive: bool,
    selected_columns: Optional[List[str]] = None,
    tablefmt: str = "latex_raw",
) -> str:
    """
    Build a LaTeX tabular with key properties and metrics using tabulate.

    Args:
        summaries: List of RunSummary objects to include in the table
        include_progressive: Whether to include progressive-specific columns
        selected_columns: Optional list of column field names to include. If None, includes all columns.
    """
    # Columns for properties
    prop_cols = [
        ("run_hash", "RunHash"),
        ("loss_type", "Loss"),
        ("hybrid_alpha", "Hybrid $\\alpha$"),
        ("embedding_init_method", "Init"),
        ("max_sequence_length", "SeqLen"),
        ("number_of_compressed_tokens", "MemT"),
        ("num_alignment_layers", "AlignL"),
        ("fix_position_ids", "FixPosIds"),
        ("model_checkpoint", "Model"),
        ("dtype", "DType"),
        ("max_optimization_steps_per_sample", "MaxSteps"),
        ("learning_rate", "LR"),
        ("low_dim_size", "LowDim"),
    ]
    # Metric columns (non-progressive)
    metric_cols = [
        ("convergence_after_steps_mean", "ConvSteps", True),  # True => integer formatting
        ("final_convergence_mean", "FinalConv (mean $\\pm$ std)", False),
        ("final_loss_mean", "FinalLoss (mean $\\pm$ std)", False),
        ("information_gain_bits_mean", "InfoGain (bits)", False),
    ]
    # Progressive specific (optional tail columns)
    progressive_metric_cols = [
        ("steps_taken_mean", "StepsTaken (mean $\\pm$ std)", True),
        ("convergence_threshold", "ConvThresh", False),
    ]

    # Filter columns if selected_columns is provided
    if selected_columns is not None:
        selected_set = set(selected_columns)
        prop_cols = [col for col in prop_cols if col[0] in selected_set]
        metric_cols = [col for col in metric_cols if col[0] in selected_set]
        progressive_metric_cols = [col for col in progressive_metric_cols if col[0] in selected_set]

    headers = [hdr for _, hdr in prop_cols] + [hdr for _, hdr, _ in metric_cols]
    # Include progressive columns if: (1) include_progressive flag is set, OR (2) user explicitly selected them
    if include_progressive or (selected_columns is not None and progressive_metric_cols):
        headers += [hdr for _, hdr, _ in progressive_metric_cols]

    use_latex = is_latex_tablefmt(tablefmt)
    table_rows: List[List[str]] = []
    for s in summaries:
        row: List[str] = []
        # Properties
        for field_name, _hdr in prop_cols:
            val = getattr(s, field_name)
            if field_name == "number_of_compressed_tokens":
                if (
                    s.dataset_type == "progressive_prefixes"
                    and s.number_of_compressed_tokens is not None
                    and s.number_of_compressed_tokens_std is not None
                ):
                    cell = to_mean_std_cell(
                        s.number_of_compressed_tokens,
                        s.number_of_compressed_tokens_std,
                        is_int=True,
                        use_latex=use_latex,
                    )
                else:
                    cell = "" if val is None else str(int(round(val)))
            elif isinstance(val, bool):
                cell = "True" if val else "False"
            else:
                cell = "" if val is None else str(val)
            row.append(latex_escape(cell) if use_latex else cell)
        # Metrics
        for field_name, _hdr, is_int in metric_cols:
            if field_name == "convergence_after_steps_mean":
                if s.dataset_type == "compressed_prefixes":
                    row.append(
                        to_mean_std_cell(
                            s.convergence_after_steps_mean,
                            s.convergence_after_steps_std,
                            is_int=is_int,
                            use_latex=use_latex,
                        )
                    )
                else:
                    row.append("")
            elif field_name == "final_convergence_mean":
                row.append(
                    to_mean_std_cell(s.final_convergence_mean, s.final_convergence_std, is_int=is_int, use_latex=use_latex)
                )
            elif field_name == "final_loss_mean":
                row.append(to_mean_std_cell(s.final_loss_mean, s.final_loss_std, is_int=is_int, use_latex=use_latex))
            elif field_name == "information_gain_bits_mean":
                row.append(
                    to_mean_std_cell(
                        s.information_gain_bits_mean, s.information_gain_bits_std, is_int=is_int, use_latex=use_latex
                    )
                )
        # Progressive extras if requested or explicitly selected
        if include_progressive or (selected_columns is not None and progressive_metric_cols):
            for field_name, _hdr, is_int in progressive_metric_cols:
                if field_name == "steps_taken_mean":
                    if s.dataset_type == "progressive_prefixes":
                        row.append(to_mean_std_cell(s.steps_taken_mean, s.steps_taken_std, is_int=is_int, use_latex=use_latex))
                    else:
                        row.append("")
                elif field_name == "convergence_threshold":
                    if s.dataset_type == "progressive_prefixes":
                        row.append("" if s.convergence_threshold is None else f"{s.convergence_threshold:.2f}")
                    else:
                        row.append("")
        table_rows.append(row)

    result = tabulate(table_rows, headers=headers, tablefmt=tablefmt)
    if is_latex_tablefmt(tablefmt):
        result = hlines_to_booktabs(result)
    return result


def load_dataset_rows(ds_path: str) -> List[dict]:
    """
    Load dataset rows, preferring a cached version without embeddings.
    If the cached version doesn't exist, load the original, remove embeddings, and save the stripped version.
    """
    ds_path_stripped = ds_path + "_no_embeddings"

    # Try to load the stripped version first
    if Path(ds_path_stripped).exists():
        ds = load_from_disk(ds_path_stripped)
    else:
        # Load original dataset
        ds = load_from_disk(ds_path)
        # Remove embedding column if it exists to save memory
        columns_to_remove = ["embedding", "low_dim_prjoection_b", "low_dim_prjoection_w"]
        for ctr in columns_to_remove:
            if ctr in ds.column_names:
                ds = ds.remove_columns(ctr)

        # Save the stripped version for future use
        ds.save_to_disk(ds_path_stripped)

    # Ensure plain Python data (avoid pandas requirement)
    return [dict(r) for r in ds]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="results.py",
        description="Aggregate experiment artifacts and print a table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            By default scans:
              - artifacts/experiments/*/compressed_prefixes
              - artifacts/experiments_progressive/*/progressive_prefixes
              - artifacts/experiments_prefix_tuning/*/prefix_tuning_prefixes
            """
        ),
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=["artifacts/experiments", "artifacts/experiments_progressive", "artifacts/experiments_prefix_tuning"],
        help="Base directories to scan for runs.",
    )
    parser.add_argument(
        "--include-progressive",
        action="store_true",
        help="Include progressive-only columns (StepsTaken, ConvThresh).",
    )
    parser.add_argument(
        "--only-non-progressive",
        action="store_true",
        help="Only include non-progressive runs (ignore progressive_prefixes).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the table; prints to stdout otherwise.",
    )
    parser.add_argument(
        "--tablefmt",
        type=str,
        default="grid",
        help="Tabulate table format (e.g., grid, simple, github, latex_raw). Default: grid.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Specify which columns to include in the table. Available columns: "
        "run_hash, loss_type, hybrid_alpha, embedding_init_method, max_sequence_length, "
        "number_of_compressed_tokens, num_alignment_layers, fix_position_ids, model_checkpoint, "
        "dtype, max_optimization_steps_per_sample, learning_rate, low_dim_size, "
        "num_alignment_layers, "
        "convergence_after_steps_mean, "
        "final_convergence_mean, final_loss_mean, information_gain_bits_mean, steps_taken_mean, convergence_threshold. "
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

    parser.add_argument(
        "--loss-type", type=str, default=None, help="Filter by loss type (e.g., l2, l1, cosine, cross_entropy)."
    )
    parser.add_argument("--hybrid-alpha", type=str, default=None, help="Filter by hybrid alpha value (string match).")
    parser.add_argument("--init", type=str, default=None, help="Filter by embedding init method (e.g., random, mvnormal).")
    parser.add_argument(
        "--embedding-init-method", type=str, default=None, help="Filter by embedding init method (e.g., random, mvnormal)."
    )
    parser.add_argument("--seq-len", type=int, default=None, help="Filter by max sequence length (int).")
    parser.add_argument("--mem-tokens", type=int, default=None, help="Filter by number of mem tokens (int).")
    parser.add_argument("--align-layers", type=int, default=None, help="Filter by number of alignment layers (int).")
    parser.add_argument(
        "--fix-position-ids",
        type=_parse_bool,
        default=None,
        help="Filter by fix_position_ids. Accepts: true/false, 1/0, yes/no, t/f, y/n (case-insensitive).",
    )
    parser.add_argument("--model", type=str, default=None, help="Filter by model checkpoint substring (case-insensitive).")
    parser.add_argument("--max-steps", type=int, default=None, help="Filter by max optimization steps per sample (int).")
    parser.add_argument("--dtype", type=str, default=None, help="Filter by dtype (e.g., float32, float16, bfloat16).")
    args = parser.parse_args(argv)

    ds_paths = discover_run_datasets(args.dirs)
    if args.only_non_progressive:
        ds_paths = [p for p in ds_paths if p[1] == "compressed_prefixes"]

    if not ds_paths:
        print("No experiment datasets found. Searched:", file=sys.stderr)
        for d in args.dirs:
            print(f" - {d}", file=sys.stderr)
        return 1

    summaries: List[RunSummary] = []
    for ds_path, ds_type in tqdm(ds_paths, desc="Processing Runs"):
        try:
            rows = load_dataset_rows(ds_path)
        except Exception as e:
            print(f"Failed to load dataset at {ds_path}: {e}", file=sys.stderr)
            continue
        if ds_type == "compressed_prefixes":
            summary = aggregate_non_progressive(ds_path, rows)
        elif ds_type == "progressive_prefixes":
            summary = aggregate_progressive(ds_path, rows)
        elif ds_type == "prefix_tuning_prefixes":
            summary = aggregate_prefix_tuning(ds_path, rows)
        else:
            print(f"Unknown dataset type: {ds_type}", file=sys.stderr)
            continue
        if summary is None:
            continue
        summaries.append(summary)

    # Sort for readability
    def sort_key(s: RunSummary):
        return (
            0 if s.dataset_type == "compressed_prefixes" else 1,
            s.model_checkpoint,
            str(s.fix_position_ids or ""),
            str(s.loss_type or ""),
            str(s.embedding_init_method or ""),
            int(s.max_sequence_length or 0),
            int(s.number_of_compressed_tokens or 0),
        )

    summaries_sorted = sorted(summaries, key=sort_key)

    # Apply property filters
    def matches_filters(s: RunSummary) -> bool:
        if args.loss_type is not None and (s.loss_type or "").lower() != args.loss_type.lower():
            return False
        if args.hybrid_alpha is not None and (s.hybrid_alpha or "").lower() != args.hybrid_alpha.lower():
            return False
        # Check embedding_init_method filter (--embedding-init-method takes precedence over --init)
        embedding_init_filter = args.embedding_init_method if args.embedding_init_method is not None else args.init
        if embedding_init_filter is not None and (s.embedding_init_method or "").lower() != embedding_init_filter.lower():
            return False
        if args.seq_len is not None and (s.max_sequence_length is None or int(s.max_sequence_length) != int(args.seq_len)):
            return False
        if args.align_layers is not None and (
            s.num_alignment_layers is None or int(s.num_alignment_layers) != int(args.align_layers)
        ):
            return False
        if args.fix_position_ids is not None and (
            s.fix_position_ids is None or bool(s.fix_position_ids) != bool(args.fix_position_ids)
        ):
            return False
        if args.model is not None:
            model_val = (s.model_checkpoint or "").lower()
            if args.model.lower() not in model_val:
                return False
        if args.max_steps is not None and (
            s.max_optimization_steps_per_sample is None or int(s.max_optimization_steps_per_sample) != int(args.max_steps)
        ):
            return False
        if args.dtype is not None and (s.dtype or "").lower() != args.dtype.lower():
            return False
        return True

    summaries_sorted = [s for s in summaries_sorted if matches_filters(s)]
    latex = build_latex_table(
        summaries_sorted,
        include_progressive=args.include_progressive,
        selected_columns=args.columns,
        tablefmt=args.tablefmt,
    )

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
