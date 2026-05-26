"""Compute Table 18 metrics for one compressed_prefixes directory.

For each saved sample we:
  1. Load the trained compression embedding and the saved input_ids.
  2. Greedy-decode N = len(input_ids) tokens from the compression embedding
     using ``compression_horizon.inference.generation.generate_from_compression``
     with EOS early-stop disabled (we need exactly N predictions for the
     position-wise mismatch counts).
  3. Compare against input_ids and report:
        final_conv        - mean teacher-forcing accuracy (from artifact)
        greedy_conv       - mean greedy-decoding accuracy
        mismatch@0/1/2    - fraction of samples whose greedy prediction differs
                            from ground truth at position 0 / 1 / 2

Output JSON is consumed by scripts/build_table_18.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.inference.generation import generate_from_compression
from compression_horizon.utils.launch import resolve_torch_dtype


def _evaluate_one(
    *,
    row: dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict:
    input_ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
    num_tokens = int(input_ids.shape[0])
    if num_tokens == 0:
        return {
            "sample_id": int(row.get("sample_id", -1)),
            "num_tokens": 0,
            "final_convergence": float(row.get("final_convergence", float("nan"))),
            "greedy_match_rate": float("nan"),
            "mismatch_at_position": [],
        }

    embedding = torch.tensor(row["embedding"], dtype=torch.float32, device=device)
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)  # [1, H]
    compression_tokens = embedding.unsqueeze(0).to(model_dtype)  # [1, C, H]

    _texts, generated_ids = generate_from_compression(
        model=model,
        tokenizer=tokenizer,
        compression_token_embeddings=compression_tokens,
        max_new_tokens=num_tokens,
        num_return_sequences=1,
        return_generated_ids=True,
    )
    generated = generated_ids[0].to(device)
    # generate_from_compression aborts on EOS, so it may return < num_tokens
    # predictions. Pad the tail with EOS (or pad_token) so position-wise
    # mismatch counts at indices 0/1/2 stay well-defined and any post-EOS
    # gap counts as a mismatch against the ground-truth tokens.
    n_pred = int(generated.shape[0])
    if n_pred < num_tokens:
        fill_id = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
        )
        pad = torch.full((num_tokens - n_pred,), int(fill_id), dtype=generated.dtype, device=device)
        generated = torch.cat([generated, pad], dim=0)
    matches = (generated == input_ids).to(torch.long)
    greedy_match_rate = float(matches.float().mean().item())
    mismatch_at_position = [int((1 - matches[k]).item()) for k in range(min(3, num_tokens))]

    return {
        "sample_id": int(row.get("sample_id", -1)),
        "num_tokens": num_tokens,
        "num_generated": n_pred,
        "final_convergence": float(row.get("final_convergence", float("nan"))),
        "greedy_match_rate": greedy_match_rate,
        "mismatch_at_position": mismatch_at_position,
        "predicted_first3": generated[: min(3, num_tokens)].cpu().tolist(),
        "target_first3": input_ids[: min(3, num_tokens)].cpu().tolist(),
    }


def _aggregate(per_sample: list[dict]) -> dict:
    valid = [r for r in per_sample if r["num_tokens"] > 0]
    n = len(valid)
    if n == 0:
        return {"num_samples": 0}

    final_conv = sum(r["final_convergence"] for r in valid) / n
    greedy_conv = sum(r["greedy_match_rate"] for r in valid) / n
    mismatch = {}
    for k in (0, 1, 2):
        miss = [r["mismatch_at_position"][k] for r in valid if len(r["mismatch_at_position"]) > k]
        mismatch[f"mismatch_at_{k}"] = (sum(miss) / len(miss)) if miss else None

    return {
        "num_samples": n,
        "final_convergence_mean": final_conv,
        "greedy_match_rate_mean": greedy_conv,
        **mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Table 18 metrics for a single experiment")
    parser.add_argument(
        "--compressed_prefixes_path",
        required=True,
        help="Path to the compressed_prefixes/ directory saved by FullCrammingTrainer.",
    )
    parser.add_argument(
        "--model_checkpoint",
        default=None,
        help="HF model name. If omitted, taken from the first dataset row.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model dtype for evaluation (default: bfloat16, matching training).",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Where to write the aggregated metrics JSON. Default: <parent>/table_18_eval.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))

    ds = Dataset.load_from_disk(args.compressed_prefixes_path)
    if len(ds) == 0:
        raise SystemExit(f"No rows in {args.compressed_prefixes_path}")

    model_checkpoint = args.model_checkpoint or ds[0].get("model_checkpoint")
    if not model_checkpoint:
        raise SystemExit("model_checkpoint not provided and missing from dataset rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_torch_dtype(args.dtype)
    print(f"Loading {model_checkpoint} ({model_dtype}) on {device}")
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, dtype=model_dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    per_sample: list[dict] = []
    for i in range(len(ds)):
        row = ds[i]
        if "input_ids" not in row or row["input_ids"] is None:
            raise SystemExit(
                f"Row {i} has no 'input_ids' field. Was training run BEFORE the camera-ready "
                "refactor that added input_ids to compressed_prefixes? Retrain the experiment."
            )
        result = _evaluate_one(row=row, model=model, tokenizer=tokenizer, device=device, model_dtype=model_dtype)
        print(
            f"[{i+1}/{len(ds)}] sample_id={result['sample_id']} "
            f"final={result['final_convergence']:.4f} greedy={result['greedy_match_rate']:.4f} "
            f"mismatch@0/1/2={result['mismatch_at_position']}"
        )
        per_sample.append(result)

    summary = _aggregate(per_sample)
    summary["compressed_prefixes_path"] = os.path.abspath(args.compressed_prefixes_path)
    summary["model_checkpoint"] = model_checkpoint
    summary["per_sample"] = per_sample

    out_path = args.output_json or str(Path(args.compressed_prefixes_path).parent / "table_18_eval.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote summary to: {out_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
