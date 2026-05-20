"""Causal attention-knockout probe (paper Section 4.4 reviewer extension).

For each Progressive sample, measures teacher-forced reconstruction accuracy of
the original prefix under three masking regimes that zero out attention from
all query positions onto the compression-token columns at the chosen layers:

    - per-layer KO          : mask layer {l}                  for l = 0..L-1
    - forward cumulative KO : mask layers {0..l}              for l = 0..L-1
    - reverse cumulative KO : mask layers {l..L-1}            for l = 0..L-1

Together with a no-knockout baseline these tell us which layers *causally*
depend on the compression token. Paper text (W2 response): early-layer KO
degrades reconstruction; late-layer KO does not. We average per-layer accuracy
across samples and report the asymmetry.

Output: ``--output_dir/attention_knockout.json``.
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

from compression_horizon.analysis.attention_intervention import (
    compute_reconstruction_accuracy_with_knockout,
    get_decoder_layers,
)
from compression_horizon.utils.launch import get_device, resolve_torch_dtype


def _load_progressive_dataset(source_dir: str) -> Dataset:
    """Locate progressive_prefixes/ or compressed_prefixes/ inside source_dir."""
    for subdir in ("progressive_prefixes", "compressed_prefixes"):
        path = os.path.join(source_dir, subdir)
        if os.path.exists(path):
            return Dataset.load_from_disk(path)
    raise FileNotFoundError(f"No progressive_prefixes/ or compressed_prefixes/ found under {source_dir}")


def _select_one_row_per_sample(ds: Dataset) -> list[dict]:
    """Mirror analyze.py: keep the final-converged stage per sample id."""
    rows = list(ds)
    if not rows or "stage_seq_len" not in rows[0]:
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


def _per_sample_regimes(
    *,
    model,
    tokenizer,
    embedding: torch.Tensor,
    text: str,
    num_compression_tokens: int,
    num_layers: int,
    device: torch.device,
    skip_cumulative: bool,
    skip_reverse_cumulative: bool,
) -> dict:
    """Compute baseline + per-regime per-layer reconstruction accuracy for one sample."""

    def _accuracy(knockout_layers: list[int]) -> float:
        return compute_reconstruction_accuracy_with_knockout(
            model=model,
            tokenizer=tokenizer,
            compression_token_embeddings=embedding,
            context=text,
            knockout_layers=knockout_layers,
            num_compression_tokens=num_compression_tokens,
            device=device,
        )

    baseline = _accuracy([])
    per_layer = [_accuracy([layer_idx]) for layer_idx in range(num_layers)]
    cumulative = [_accuracy(list(range(layer_idx + 1))) for layer_idx in range(num_layers)] if not skip_cumulative else None
    reverse_cumulative = (
        [_accuracy(list(range(layer_idx, num_layers))) for layer_idx in range(num_layers)]
        if not skip_reverse_cumulative
        else None
    )
    return {
        "baseline": baseline,
        "per_layer": per_layer,
        "cumulative": cumulative,
        "reverse_cumulative": reverse_cumulative,
    }


def _aggregate(per_sample: list[dict], num_layers: int) -> dict:
    """Average per-sample accuracies into mean ± std per layer index."""
    baselines = torch.tensor([s["baseline"] for s in per_sample], dtype=torch.float64)

    def _layer_stats(key: str) -> dict | None:
        values = [s[key] for s in per_sample if s.get(key) is not None]
        if not values:
            return None
        matrix = torch.tensor(values, dtype=torch.float64)
        means = matrix.mean(dim=0)
        stds = matrix.std(dim=0, unbiased=False)
        return {"mean": means.tolist(), "std": stds.tolist()}

    summary: dict = {
        "num_samples": len(per_sample),
        "num_layers": num_layers,
        "baseline": {
            "mean": float(baselines.mean().item()),
            "std": float(baselines.std(unbiased=False).item()),
        },
        "per_layer": _layer_stats("per_layer"),
    }
    cumulative = _layer_stats("cumulative")
    if cumulative is not None:
        summary["cumulative"] = cumulative
    reverse_cumulative = _layer_stats("reverse_cumulative")
    if reverse_cumulative is not None:
        summary["reverse_cumulative"] = reverse_cumulative
    return summary


def _resolve_attn_implementation() -> str:
    """Eager attention is required by AttentionKnockoutContext."""
    return "eager"


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention-knockout causality probe.")
    parser.add_argument("--source_dir", required=True, help="Artifacts dir from train.py.")
    parser.add_argument("--model_checkpoint", required=True, help="HF model id used for the source run.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write attention_knockout.json (default: source_dir).",
    )
    parser.add_argument("--dtype", default="bf16", help="bf16 / fp16 / fp32.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Cap the number of samples evaluated.",
    )
    parser.add_argument(
        "--skip_cumulative",
        action="store_true",
        help="Skip the forward-cumulative regime (each layer adds knockout of layers 0..l).",
    )
    parser.add_argument(
        "--skip_reverse_cumulative",
        action="store_true",
        help="Skip the reverse-cumulative regime (each layer adds knockout of layers l..L-1).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.source_dir
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    torch_dtype = resolve_torch_dtype(args.dtype)
    print(f"Device: {device}; dtype: {torch_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint,
        dtype=torch_dtype,
        attn_implementation=_resolve_attn_implementation(),
    )
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = len(get_decoder_layers(model))
    print(f"num decoder layers: {num_layers}")

    ds = _load_progressive_dataset(args.source_dir)
    rows = _select_one_row_per_sample(ds)
    if args.num_samples is not None:
        rows = rows[: args.num_samples]
    print(f"Evaluating attention knockout on {len(rows)} samples")

    sample_records: list[dict] = []
    for row in tqdm(rows, desc="knockout"):
        embedding_tensor = torch.tensor(row["embedding"], dtype=torch.float32).to(device)
        if embedding_tensor.dim() == 1:
            embedding_tensor = embedding_tensor.unsqueeze(0)
        num_compression_tokens = embedding_tensor.shape[0]

        regimes = _per_sample_regimes(
            model=model,
            tokenizer=tokenizer,
            embedding=embedding_tensor,
            text=row["text"],
            num_compression_tokens=num_compression_tokens,
            num_layers=num_layers,
            device=device,
            skip_cumulative=args.skip_cumulative,
            skip_reverse_cumulative=args.skip_reverse_cumulative,
        )
        sample_records.append(
            {
                "sample_id": int(row.get("sample_id", -1)),
                "stage_seq_len": int(row.get("stage_seq_len", row.get("num_input_tokens", 0))),
                **regimes,
            }
        )

    summary = _aggregate(sample_records, num_layers)

    output = {
        "config": {
            "model_checkpoint": args.model_checkpoint,
            "source_dir": args.source_dir,
            "num_samples": len(sample_records),
            "num_layers": num_layers,
            "skip_cumulative": args.skip_cumulative,
            "skip_reverse_cumulative": args.skip_reverse_cumulative,
        },
        "summary": summary,
        "samples": sample_records,
    }
    output_path = Path(output_dir) / "attention_knockout.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {output_path}")

    print()
    print("Summary:")
    print(f"  baseline reconstruction accuracy: {summary['baseline']['mean']:.4f} ± {summary['baseline']['std']:.4f}")
    if summary.get("per_layer") is not None:
        first = summary["per_layer"]["mean"][0]
        last = summary["per_layer"]["mean"][-1]
        print(f"  per-layer KO at layer 0           : {first:.4f}")
        print(f"  per-layer KO at layer {num_layers - 1:<11}: {last:.4f}")


if __name__ == "__main__":
    main()
