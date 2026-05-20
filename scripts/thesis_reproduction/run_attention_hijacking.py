"""Compute per-sample attention-hijacking profiles over a saved compression Dataset.

Implements paper Section 5.5 (Table 3): for each sample we measure how much
attention every model layer routes onto the first prefix position when that
position is (a) the learned compression embedding, and (b) the BOS token.
Outputs a JSON file with per-sample profiles + aggregate Table-3-style stats.

Inputs:
    --source_dir: a thesis_reproduction artifacts dir produced by train.py
                  (must contain progressive_prefixes/ or compressed_prefixes/).
    --model_checkpoint: HF id of the LM whose attention is being measured.

Output:
    --output_dir/attention_hijacking.json with keys
        {samples: [...], summary: {...}, config: {...}}.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.analysis import (
    compute_sample_profiles,
    pearson_correlation,
    summarize_hijacking,
)
from compression_horizon.utils.launch import freeze_model_parameters, get_device, resolve_torch_dtype


def _resolve_attn_implementation() -> str:
    """Eager attention is required to get output_attentions; report once."""
    return "eager"


def _load_source_dataset(source_dir: str) -> Dataset:
    """Locate progressive_prefixes/ or compressed_prefixes/ inside source_dir."""
    for subdir in ("progressive_prefixes", "compressed_prefixes"):
        path = os.path.join(source_dir, subdir)
        if os.path.exists(path):
            return Dataset.load_from_disk(path)
    raise FileNotFoundError(f"No progressive_prefixes/ or compressed_prefixes/ found under {source_dir}")


def _select_one_row_per_sample(ds: Dataset) -> list[dict]:
    """For Progressive datasets, pick the final-converged stage per sample id.

    Mirrors thesis_reproduction/analyze.py::_aggregate_rows_per_sample so the
    hijacking analysis is performed on the same embeddings that the IG
    analysis was computed on.
    """
    rows = list(ds)
    if not rows:
        return rows
    if "stage_seq_len" not in rows[0]:
        return rows
    by_sample: dict[int, list[dict]] = {}
    for row in rows:
        by_sample.setdefault(int(row["sample_id"]), []).append(row)
    aggregated: list[dict] = []
    for sample_id in sorted(by_sample):
        sample_rows = by_sample[sample_id]
        converged = [r for r in sample_rows if r.get("final_convergence") == 1.0]
        candidates = converged if converged else sample_rows
        aggregated.append(max(candidates, key=lambda r: r["stage_seq_len"]))
    return aggregated


def _parse_target_lengths(spec: str | None) -> list[int] | None:
    if spec is None or spec.strip() == "":
        return None
    return [int(s) for s in spec.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention-hijacking analysis (paper Section 5.5).")
    parser.add_argument("--source_dir", required=True, help="Artifacts dir from train.py.")
    parser.add_argument("--model_checkpoint", required=True, help="HF model id used for the source run.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write attention_hijacking.json (default: source_dir).",
    )
    parser.add_argument("--dtype", default="bf16", help="bf16 / fp16 / fp32.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Cap the number of samples evaluated.",
    )
    parser.add_argument(
        "--target_prefix_lengths",
        default=None,
        help="Comma-separated suffix lengths used in eq. 8 (default: geometric 4..4096 capped by sample length).",
    )
    parser.add_argument("--num_compression_tokens", type=int, default=1)
    args = parser.parse_args()

    output_dir = args.output_dir or args.source_dir
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    torch_dtype = resolve_torch_dtype(args.dtype)
    print(f"Device: {device}; dtype: {torch_dtype}")

    attn_implementation = _resolve_attn_implementation()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint, dtype=torch_dtype, attn_implementation=attn_implementation
    )
    freeze_model_parameters(model)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = _load_source_dataset(args.source_dir)
    rows = _select_one_row_per_sample(dataset)
    if args.num_samples is not None:
        rows = rows[: args.num_samples]
    print(f"Evaluating attention hijacking on {len(rows)} samples")

    target_prefix_lengths = _parse_target_lengths(args.target_prefix_lengths)

    sample_records: list[dict] = []
    comp_profiles: list[list[float]] = []
    bos_profiles: list[list[float]] = []
    skipped: list[tuple[int, str]] = []
    for row in tqdm(rows, desc="hijacking"):
        embedding_tensor = torch.tensor(row["embedding"], dtype=torch.float32)
        # Tolerate [H] vs [num_compression_tokens, H]: progressive stores [num_compression_tokens, H].
        if embedding_tensor.dim() == 1:
            embedding_tensor = embedding_tensor.unsqueeze(0)
        num_compression_tokens = embedding_tensor.shape[0]
        if num_compression_tokens != args.num_compression_tokens:
            print(
                f"Sample {row.get('sample_id')}: embedding has {num_compression_tokens} tokens, "
                f"overriding --num_compression_tokens={args.num_compression_tokens}."
            )
        text = row["text"]
        try:
            comp_profile, bos_profile, used_lengths = compute_sample_profiles(
                model=model,
                tokenizer=tokenizer,
                compression_token_embedding=embedding_tensor.to(device),
                context=text,
                num_compression_tokens=num_compression_tokens,
                target_prefix_lengths=target_prefix_lengths,
                device=device,
            )
        except ValueError as exc:
            # Typically: sample too short for the default geometric schedule.
            # We log and skip rather than silently rescale, so the metric stays paper-canonical.
            skipped.append((int(row.get("sample_id", -1)), str(exc)))
            continue
        corr = pearson_correlation(comp_profile, bos_profile)
        sample_records.append(
            {
                "sample_id": int(row.get("sample_id", -1)),
                "stage_seq_len": int(row.get("stage_seq_len", row.get("num_input_tokens", 0))),
                "used_lengths": used_lengths,
                "compression_profile": comp_profile,
                "bos_profile": bos_profile,
                "compression_max_pct": max(comp_profile) * 100.0,
                "bos_max_pct": max(bos_profile) * 100.0,
                "correlation": corr,
            }
        )
        comp_profiles.append(comp_profile)
        bos_profiles.append(bos_profile)

    if skipped:
        print(f"\nSkipped {len(skipped)} sample(s) too short for the suffix schedule:")
        for sample_id, reason in skipped:
            print(f"  sample_id={sample_id}: {reason}")

    summary = summarize_hijacking(comp_profiles, bos_profiles)

    output = {
        "config": {
            "model_checkpoint": args.model_checkpoint,
            "source_dir": args.source_dir,
            "num_samples": len(sample_records),
            "num_skipped": len(skipped),
            "skipped_sample_ids": [sample_id for sample_id, _ in skipped],
            "target_prefix_lengths": target_prefix_lengths,
        },
        "summary": summary,
        "samples": sample_records,
    }
    output_path = Path(output_dir) / "attention_hijacking.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {output_path}")

    print()
    print("Summary:")
    print(f"  compression mass (%): {summary['compression_mass']['mean']:.2f} ± {summary['compression_mass']['std']:.2f}")
    print(f"  bos mass (%):         {summary['bos_mass']['mean']:.2f} ± {summary['bos_mass']['std']:.2f}")
    print(f"  correlation:          {summary['correlation']['mean']:.4f} ± {summary['correlation']['std']:.4f}")


if __name__ == "__main__":
    main()
