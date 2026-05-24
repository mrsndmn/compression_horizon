import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scripts.visualize_attention_hijacking import (
    collate_stages_by_sample,
    compute_attention_mass_for_original_sequence,
    compute_attention_mass_for_stages,
    compute_average_attention_mass_per_layer,
    filter_records,
    load_dataset,
)
from tabulate import tabulate
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3Config

from compression_horizon.utils import hlines_to_booktabs, to_mean_std_cell


def save_attention_mass_cache(
    cache_data: Dict[str, Any],
    output_dir: str,
    sample_id: int,
):
    """Save attention mass cache to JSON file.

    Args:
        cache_data: Dictionary containing attention mass data
        output_dir: Directory to save cache file
        sample_id: Sample ID
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_filename = f"attention_mass_cache_sample_{sample_id}.json"
    cache_path = os.path.join(output_dir, cache_filename)
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)
    print(f"Saved attention mass cache to: {cache_path}")


def get_sample_ids_from_dataset(dataset_path: str) -> Tuple[List[int], str]:
    """Get sample IDs from dataset without loading full rows.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        Tuple of (sample_ids list, dataset_type)
    """
    try:
        ds, dataset_type = load_dataset(dataset_path)
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

    if dataset_type == "unknown":
        raise ValueError(f"Could not detect dataset type for {dataset_path}")

    # Get sample IDs efficiently
    if "sample_id" in ds.column_names:
        try:
            sample_ids = ds.unique("sample_id")
            sample_ids = sorted({int(sid) for sid in sample_ids if sid is not None})
        except Exception:
            # Fallback: iterate through dataset
            sample_ids = []
            for i in range(len(ds)):
                try:
                    sid = ds[i].get("sample_id")
                    if sid is not None:
                        sample_ids.append(int(sid))
                except Exception:
                    continue
            sample_ids = sorted(set(sample_ids))
    else:
        sample_ids = []

    return sample_ids, dataset_type


def check_all_cache_files_exist(dataset_path: str, sample_ids: List[int]) -> bool:
    """Check if cache files exist for all sample IDs.

    Args:
        dataset_path: Path to the dataset directory
        sample_ids: List of sample IDs to check

    Returns:
        True if cache files exist for all samples, False otherwise
    """
    if not sample_ids:
        return False

    output_dir = os.path.join(dataset_path, "attention_visualizations")
    for sample_id in sample_ids:
        cache_file = os.path.join(output_dir, f"attention_mass_cache_sample_{sample_id}.json")
        if not os.path.exists(cache_file):
            return False
    return True


def compute_checkpoint_attention_mass_data(
    dataset_path: str,
    model_checkpoint: Optional[str] = None,
    min_seq_length: int = 1,
    attention_block_size: int = 16,
    device: Optional[torch.device] = None,
    force: bool = False,
) -> bool:
    """Compute and save attention mass data for a checkpoint.

    Args:
        dataset_path: Path to the dataset directory
        model_checkpoint: Model checkpoint path (if None, tries to infer from dataset)
        min_seq_length: Minimum sequence length to consider
        attention_block_size: Block size for attention computation
        device: Device to run on (if None, uses CUDA if available)
        force: If True, recompute even if cache files exist

    Returns:
        True if computation succeeded, False otherwise
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if we can skip loading dataset (if all cache files exist and not forcing)
    if not force:
        try:
            sample_ids, dataset_type = get_sample_ids_from_dataset(dataset_path)
            if sample_ids and check_all_cache_files_exist(dataset_path, sample_ids):
                print(f"All cache files already exist for {len(sample_ids)} samples in {dataset_path}, skipping computation.")
                return True
        except Exception as e:
            print(f"Warning: Could not check existing cache files: {e}, proceeding with computation...")

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    try:
        ds, dataset_type = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    if dataset_type == "unknown":
        print(f"Warning: Could not detect dataset type for {dataset_path}")
        return False

    print(f"Detected dataset type: {dataset_type}")

    # Filter records
    rows = filter_records(ds, sample_id=None, dataset_type=dataset_type)
    if not rows:
        print(f"No records found in {dataset_path}")
        return False

    # Group by sample
    by_sid = collate_stages_by_sample(rows, dataset_type=dataset_type)

    # Determine model checkpoint
    if model_checkpoint is None:
        if rows:
            model_checkpoint = rows[0].get("model_checkpoint", "")
            if not model_checkpoint:
                print(f"Error: model_checkpoint not provided and cannot be inferred from dataset in {dataset_path}")
                return False
        else:
            print(f"Error: No rows found to infer model_checkpoint from in {dataset_path}")
            return False

    print(f"Using model checkpoint: {model_checkpoint}")

    # Load model and tokenizer
    print(f"Loading model on device: {device}")

    if model_checkpoint.startswith("unsloth/"):
        model_checkpoint = model_checkpoint.replace("unsloth/", "unsloth/")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            attn_implementation="eager",
        ).to(device)
    except TypeError:
        # Fallback for older transformers versions
        print("Warning: attn_implementation parameter not supported, loading model without it...")
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
        try:
            model.set_attn_implementation("eager")
        except (AttributeError, ValueError):
            print("Warning: Could not set attention implementation to 'eager'.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    config = model.config
    if isinstance(model.config, Gemma3Config):
        config = model.config.text_config

    num_layers = config.num_hidden_layers

    # Determine output directory
    output_dir = os.path.join(dataset_path, "attention_visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # Filter eligible samples
    eligible_by_sid: Dict[int, List[Dict[str, Any]]] = {}
    per_sample_max = []
    for _sid, stages in by_sid.items():
        if dataset_type == "prefix_tuning":
            stage_record = stages[0]
            text = stage_record.get("text", "")
            if not isinstance(text, str) or text.strip() == "":
                continue
            enc = tokenizer(text, truncation=True, padding=False, return_tensors="pt")
            max_len = enc["input_ids"].shape[1]
        else:
            max_len = max((int(s.get("stage_seq_len", -1)) for s in stages), default=-1)
        if max_len >= min_seq_length:
            eligible_by_sid[_sid] = stages
            per_sample_max.append(max_len)

    if not per_sample_max:
        print(f"No samples with max sequence length >= {min_seq_length} found in {dataset_path}")
        return False

    min_max_len = min(per_sample_max)
    target_seq_lengths_override = list(range(min_seq_length, min_max_len + 1))
    print(f"Processing {len(eligible_by_sid)} samples; using target_seq_len in [{min_seq_length}, {min_max_len}]")

    # Process each sample
    for sample_id, stages in tqdm(eligible_by_sid.items(), desc="Processing samples"):
        stage_count = len(stages)
        stage_label = "stages" if dataset_type == "progressive" else "entry"
        print(f"\nProcessing sample {sample_id} with {stage_count} {stage_label}...")

        # Check if cache already exists
        cache_file = os.path.join(output_dir, f"attention_mass_cache_sample_{sample_id}.json")
        if os.path.exists(cache_file):
            print(f"Cache file already exists for sample {sample_id}, skipping...")
            continue

        # Compute attention mass for compression embeddings
        results, attentions, text, num_compression_tokens = compute_attention_mass_for_stages(
            model=model,
            tokenizer=tokenizer,
            stages=stages,
            device=device,
            attention_block_size=attention_block_size,
            target_seq_lengths_override=target_seq_lengths_override,
            dataset_type=dataset_type,
        )

        if not results or text is None:
            print(f"Warning: Could not compute attention mass for sample {sample_id}")
            continue

        # Compute average attention mass per layer for compression
        avg_attention_mass_compression = compute_average_attention_mass_per_layer(
            results=results,
            num_layers=num_layers,
        )

        # Compute attention mass for original sequence (without compression)
        print(f"Computing attention mass for original sequence (sample {sample_id})...")
        avg_attention_mass_original = compute_attention_mass_for_original_sequence(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
            target_seq_lengths=target_seq_lengths_override,
            num_layers=num_layers,
        )

        # Save to cache
        cache_data = {
            "sample_id": sample_id,
            "num_layers": num_layers,
            "target_seq_lengths": target_seq_lengths_override,
            "avg_attention_mass_per_layer_compression": avg_attention_mass_compression,
            "avg_attention_mass_per_layer_original": avg_attention_mass_original,
        }
        save_attention_mass_cache(cache_data, output_dir, sample_id=sample_id)

    return True


def find_attention_mass_cache_files(checkpoint_dir: str) -> List[str]:
    """Find all attention mass cache files in a checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory (dataset path)

    Returns:
        List of cache file paths
    """
    cache_files = []
    # Check in attention_visualizations subdirectory
    attention_vis_dir = os.path.join(checkpoint_dir, "attention_visualizations")
    if os.path.isdir(attention_vis_dir):
        for filename in os.listdir(attention_vis_dir):
            if filename.startswith("attention_mass_cache_sample_") and filename.endswith(".json"):
                cache_files.append(os.path.join(attention_vis_dir, filename))
    # Also check directly in checkpoint_dir (in case cache files are there)
    if os.path.isdir(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("attention_mass_cache_sample_") and filename.endswith(".json"):
                cache_files.append(os.path.join(checkpoint_dir, filename))
    return sorted(cache_files)


def load_attention_mass_cache(cache_file: str) -> Optional[Dict[str, Any]]:
    """Load attention mass cache file.

    Args:
        cache_file: Path to cache file

    Returns:
        Cache data dictionary or None if loading fails
    """
    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load cache file {cache_file}: {e}")
        return None


def compute_checkpoint_attention_mass(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    """Compute average attention mass for a checkpoint by loading all cache files.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Dictionary with 'compression', 'original', and 'diff' average attention mass (mean/std/count),
        or None if no cache files found
    """
    cache_files = find_attention_mass_cache_files(checkpoint_dir)
    if not cache_files:
        return None

    all_compression_values = []
    all_original_values = []
    all_diff_values = []
    # Collect per-layer values for correlation computation
    all_compression_per_layer: List[List[float]] = []
    all_original_per_layer: List[List[float]] = []

    for cache_file in cache_files:
        cache_data = load_attention_mass_cache(cache_file)
        if cache_data is None:
            continue

        compression_per_layer = cache_data.get("avg_attention_mass_per_layer_compression")
        original_per_layer = cache_data.get("avg_attention_mass_per_layer_original")

        sample_avg_compression = None
        sample_avg_original = None

        if isinstance(compression_per_layer, list) and len(compression_per_layer) > 0:
            # Average across all layers for this sample
            sample_avg_compression = np.mean(compression_per_layer)
            all_compression_values.append(sample_avg_compression)
            # Store per-layer values for correlation
            all_compression_per_layer.append(compression_per_layer)

        if isinstance(original_per_layer, list) and len(original_per_layer) > 0:
            # Average across all layers for this sample
            sample_avg_original = np.mean(original_per_layer)
            all_original_values.append(sample_avg_original)
            # Store per-layer values for correlation
            all_original_per_layer.append(original_per_layer)

        # Compute difference if both values are available
        if sample_avg_compression is not None and sample_avg_original is not None:
            all_diff_values.append(sample_avg_compression - sample_avg_original)

    if not all_compression_values and not all_original_values:
        return None

    def summarize_values(values: List[float]) -> Optional[Dict[str, Any]]:
        if len(values) == 0:
            return None
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "count": int(len(values)),
        }

    # Compute correlation across layers
    correlation = None
    if all_compression_per_layer and all_original_per_layer:
        # Get num_layers from first cache file
        num_layers = len(all_compression_per_layer[0])
        # Average across samples for each layer
        avg_compression_per_layer = np.zeros(num_layers)
        avg_original_per_layer = np.zeros(num_layers)

        for compression_layers, original_layers in zip(all_compression_per_layer, all_original_per_layer):
            if len(compression_layers) == num_layers and len(original_layers) == num_layers:
                avg_compression_per_layer += np.array(compression_layers)
                avg_original_per_layer += np.array(original_layers)

        num_samples = len(all_compression_per_layer)
        if num_samples > 0:
            avg_compression_per_layer /= num_samples
            avg_original_per_layer /= num_samples

            # Compute Pearson correlation coefficient
            if (
                len(avg_compression_per_layer) > 1
                and np.std(avg_compression_per_layer) > 0
                and np.std(avg_original_per_layer) > 0
            ):
                correlation = float(np.corrcoef(avg_compression_per_layer, avg_original_per_layer)[0, 1])

    result = {
        "compression": summarize_values(all_compression_values),
        "original": summarize_values(all_original_values),
        "diff": summarize_values(all_diff_values),
    }
    if correlation is not None:
        result["correlation"] = correlation

    return result


def format_mean_std_cell(
    stat: Optional[Dict[str, Any]],
    precision: int,
    tablefmt: str,
) -> str:
    """Format mean ± std cell for table.

    Args:
        stat: Dictionary with 'mean' and 'std' keys
        precision: Number of decimal places
        tablefmt: Table format (for LaTeX detection)

    Returns:
        Formatted string
    """
    if not stat:
        return "nan"
    mean_val = stat.get("mean")
    std_val = stat.get("std")
    if mean_val is None or std_val is None:
        return "nan"
    return to_mean_std_cell(
        mean_val,
        std_val,
        use_latex=(tablefmt == "latex"),
        float_precision=precision,
    )


def format_attention_mass_table(
    checkpoint_names: List[str],
    statistics: List[Dict[str, Any]],
    midrule_indicies: Optional[List[int]] = None,
    tablefmt: str = "grid",
) -> str:
    """Build the attention mass statistics table as a string."""
    if len(checkpoint_names) == 0 or len(statistics) == 0:
        return ""

    table_data = []
    for i, (name, stats) in enumerate(zip(checkpoint_names, statistics)):
        table_name = name
        table_name = table_name.replace("pt_sl_1024_", "")
        table_name = table_name.replace("sl_4096_", "")
        table_name = table_name.replace("_lowproj", "")
        table_name = table_name.replace("Meta-", "")
        table_name = table_name.replace("_ds_pg19_loss_cosine", "")
        table_name = table_name.replace("_loss_cosine", "")
        table_name = re.sub(r"_hybrid_(\d+(\.?\d+)?)", r" {\\small $\\alpha=\1$}", table_name)
        table_name = re.sub(r"_align_(\d+)", r" {\\small $L=\1$}", table_name)
        table_name = re.sub(r"_lowdim_(\d+)", r" {\\small dim=\1}", table_name)
        table_name = re.sub(r"_lr_(\d+(\.?\d+)?)", r" {\\small lr=\1}", table_name)

        correlation = stats.get("correlation")
        correlation_str = f"{correlation:.4f}" if correlation is not None else "N/A"

        table_data.append(
            [
                table_name,
                format_mean_std_cell(stats.get("compression"), precision=2, tablefmt=tablefmt),
                format_mean_std_cell(stats.get("original"), precision=2, tablefmt=tablefmt),
                format_mean_std_cell(stats.get("diff"), precision=2, tablefmt=tablefmt),
                correlation_str,
            ]
        )

        if midrule_indicies is not None and i in midrule_indicies:
            table_data.append(["\\midrule REMOVE"])

    headers = [
        "Model",
        "Compression Token (%)",
        "BOS Token Original (%)",
        "Diff (%)",
        "Correlation",
    ]
    result = tabulate(table_data, headers=headers, tablefmt=tablefmt, numalign="right", stralign="left")

    result = result.replace("\\textbackslash{}", "\\")
    result = result.replace("\\$", "$")
    result = result.replace("\\{", "{")
    result = result.replace("\\}", "}")
    result = result.replace("_nobos", " \\bcancel{B}")
    result = result.replace("P-", "Pythia")
    result = result.replace("L3.2-", "Llama-3.2-")
    result = result.replace("L3.1-", "Llama-3.1-")

    result = re.sub(r"REMOVE.+", "", result)

    if tablefmt.startswith("latex"):
        result = hlines_to_booktabs(result)
    return result


def print_attention_mass_table(
    checkpoint_names: List[str],
    statistics: List[Dict[str, Any]],
    midrule_indicies: Optional[List[int]] = None,
    tablefmt: str = "grid",
):
    """Print attention mass statistics table using tabulate."""
    result = format_attention_mass_table(checkpoint_names, statistics, midrule_indicies, tablefmt)
    if not result:
        return
    print("\n" + "=" * 80)
    print("Attention Mass Statistics")
    print("=" * 80)
    print(result)
    print("=" * 80 + "\n")


def parse_names_mapping(names_str: Optional[str]) -> tuple[Dict[str, str], Optional[List[str]]]:
    """Parse names mapping from string.

    Supports two formats:
    1. Path-based: 'path1:name1,path2:name2' (returns dict, None)
    2. Positional list: 'name1,name2,name3' (returns empty dict, list of names)

    Returns:
        Tuple of (path_mapping_dict, positional_names_list)
    """
    if names_str is None:
        return {}, None

    # Check if it contains colons (path-based mapping)
    if ":" in names_str:
        mapping = {}
        for pair in names_str.split(","):
            if ":" in pair:
                key, value = pair.split(":", 1)
                mapping[key.strip()] = value.strip()
        return mapping, None
    else:
        # Positional list format
        names = [name.strip() for name in names_str.split(",") if name.strip()]
        return {}, names if names else None


def main():
    parser = argparse.ArgumentParser(
        description="Compute and display average attention mass statistics for compression embeddings and BOS token"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to checkpoint directories (dataset paths). "
        "Script will look for attention_visualizations/attention_mass_cache_sample_*.json files, "
        "or compute them if --compute is set.",
    )
    parser.add_argument(
        "--compute",
        action="store_true",
        help="Compute attention mass data if cache files don't exist (or recompute if --force is set)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cache files exist (requires --compute)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (if not provided, will try to infer from dataset). "
        "Can be a single path (used for all checkpoints) or comma-separated list matching --checkpoints order.",
    )
    parser.add_argument(
        "--min_seq_length",
        type=int,
        default=1,
        help="Minimum sequence length to consider when computing attention mass",
    )
    parser.add_argument(
        "--attention_block_size",
        type=int,
        default=16,
        help="Block size for averaging attention for long sequences",
    )
    parser.add_argument(
        "--names_mapping",
        type=str,
        default=None,
        help="Optional mapping of checkpoint paths to display names. "
        "Two formats supported: 1) Path-based: 'path1:name1,path2:name2' "
        "2) Positional list: 'name1,name2,name3' (corresponds to --checkpoints order)",
    )
    parser.add_argument(
        "--tablefmt",
        type=str,
        default="grid",
        help="Tabulate table format for printed statistics (e.g., grid, simple, github, latex). Default: grid.",
    )
    parser.add_argument(
        "--midrule_indicies", nargs="+", type=int, default=None, help="Indices where to insert midrule (for LaTeX)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If set, write the rendered table to <save-dir>/<save-name>.tex.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="attn_hijacking",
        help="Output slug (without extension) used together with --save-dir.",
    )

    args = parser.parse_args()

    if args.force and not args.compute:
        raise ValueError("--force requires --compute to be set")

    # Parse names mapping
    path_mapping, positional_names = parse_names_mapping(args.names_mapping)

    # Validate positional names length if provided
    if positional_names is not None and len(positional_names) != len(args.checkpoints):
        raise ValueError(
            f"Number of names in --names_mapping ({len(positional_names)}) "
            f"does not match number of checkpoints ({len(args.checkpoints)})"
        )

    # Validate checkpoint directories exist
    not_exists_checkpoints = []
    for checkpoint in args.checkpoints:
        if not os.path.isdir(checkpoint):
            not_exists_checkpoints.append(checkpoint)
    if not_exists_checkpoints:
        raise ValueError(f"Checkpoints do not exist: {not_exists_checkpoints}")

    # Parse model_checkpoint (can be single or comma-separated list)
    model_checkpoints = None
    if args.model_checkpoint:
        model_checkpoints = [m.strip() for m in args.model_checkpoint.split(",")]
        if len(model_checkpoints) == 1:
            # Single model checkpoint for all
            model_checkpoints = model_checkpoints * len(args.checkpoints)
        elif len(model_checkpoints) != len(args.checkpoints):
            raise ValueError(
                f"Number of model checkpoints ({len(model_checkpoints)}) "
                f"does not match number of checkpoints ({len(args.checkpoints)})"
            )

    # Compute attention mass data if requested
    if args.compute:
        print("=" * 80)
        print("Computing attention mass data")
        print("=" * 80)
        for idx, checkpoint_path in enumerate(tqdm(args.checkpoints, desc="Computing attention mass")):
            model_checkpoint = model_checkpoints[idx] if model_checkpoints else None
            if args.force:
                # Remove existing cache files
                cache_files = find_attention_mass_cache_files(checkpoint_path)
                for cache_file in cache_files:
                    try:
                        os.remove(cache_file)
                        print(f"Removed existing cache file: {cache_file}")
                    except OSError as e:
                        print(f"Warning: Failed to remove {cache_file}: {e}")

            success = compute_checkpoint_attention_mass_data(
                dataset_path=checkpoint_path,
                model_checkpoint=model_checkpoint,
                min_seq_length=args.min_seq_length,
                attention_block_size=args.attention_block_size,
                force=args.force,
            )
            if not success:
                print(f"Warning: Failed to compute attention mass data for {checkpoint_path}")
        print("=" * 80 + "\n")

    # Compute statistics for each checkpoint
    checkpoint_names = []
    statistics_list = []

    for idx, checkpoint_path in tqdm(enumerate(args.checkpoints), desc="Processing checkpoints", total=len(args.checkpoints)):
        stats = compute_checkpoint_attention_mass(checkpoint_path)
        if stats is None:
            print(f"Warning: No attention mass cache files found in {checkpoint_path}")
            continue

        statistics_list.append(stats)

        # Determine name for this checkpoint
        if positional_names is not None:
            # Use positional mapping
            checkpoint_names.append(positional_names[idx])
        elif checkpoint_path in path_mapping:
            # Use path-based mapping
            checkpoint_names.append(path_mapping[checkpoint_path])
        else:
            # Extract a short name from the path
            name = os.path.basename(os.path.dirname(checkpoint_path))
            if not name or name == ".":
                name = os.path.basename(checkpoint_path)
            checkpoint_names.append(name)

    if len(statistics_list) == 0:
        raise ValueError("No valid statistics computed from any checkpoint")

    # Print statistics table
    print_attention_mass_table(
        checkpoint_names,
        statistics_list,
        midrule_indicies=args.midrule_indicies,
        tablefmt=args.tablefmt,
    )

    if args.save_dir is not None:
        rendered = format_attention_mass_table(
            checkpoint_names,
            statistics_list,
            midrule_indicies=args.midrule_indicies,
            tablefmt=args.tablefmt,
        )
        if rendered:
            out_dir = Path(args.save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{args.save_name}.tex"
            out_path.write_text(rendered + "\n", encoding="utf-8")
            print(f"Saved 'tab:{args.save_name}' to {out_path}")


if __name__ == "__main__":
    main()
