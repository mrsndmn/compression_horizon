"""Greedy-generation reconstruction accuracy for full / progressive cramming runs.

The cramming artifacts only store the *teacher-forcing* convergence
(``final_convergence``): the per-position argmax match rate when the model sees
the ground-truth prefix at every step. This script computes the *greedy*
analogue: starting from the trained compression embedding alone, autoregressively
generate ``num_input_tokens`` tokens (feeding back the model's own predictions)
and compare them position-by-position to the reference tokens. The per-sample
match rate is averaged over samples and cached next to the run so
``scripts/paper/tables/full_cramming_table.py`` can render it.

Usage:
    PYTHONPATH=./src:. python scripts/greedy_reconstruction_eval.py \
        --run-dir artifacts/experiments/<...full run...> --dataset-type full
    PYTHONPATH=./src:. python scripts/greedy_reconstruction_eval.py \
        --run-dir artifacts/experiments_progressive/<...progr run...> --dataset-type progr
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from compression_horizon.inference.generation import generate_from_compression

CACHE_FILENAME = "greedy_accuracy_cache.json"
CACHE_VERSION = 1

DTYPE_MAP = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Experiment dir containing compressed_prefixes/ or progressive_prefixes/.")
    p.add_argument("--dataset-type", choices=["full", "progr"], required=True)
    p.add_argument("--limit", type=int, default=None, help="Cap on number of samples (default: all).")
    p.add_argument("--offset", type=int, default=0, help="Skip the first N samples before applying --limit.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="Recompute even if a matching cache exists.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def reference_ids(tokenizer, text: str, num_input_tokens: int) -> list[int]:
    """Re-tokenize the (special-token-stripped) text back to the crammed ids.

    The artifacts store ``text = decode(input_ids, skip_special_tokens=True)`` but
    not the ids themselves, so we re-encode and pick the special-token setting
    whose length matches ``num_input_tokens`` (Llama re-adds BOS, Pythia does not).
    """
    with_special = tokenizer.encode(text, add_special_tokens=True)
    if len(with_special) == num_input_tokens:
        ids = with_special
    else:
        without_special = tokenizer.encode(text, add_special_tokens=False)
        ids = (
            without_special
            if abs(len(without_special) - num_input_tokens) <= abs(len(with_special) - num_input_tokens)
            else with_special
        )
    return ids[:num_input_tokens]


def select_rows(ds, dataset_type: str) -> list[dict]:
    """One row per sample: the row itself (full) or the final stage (progressive)."""
    if dataset_type == "full":
        return [ds[i] for i in range(len(ds))]
    by_sample: dict[int, dict] = {}
    for i in range(len(ds)):
        r = ds[i]
        sid = int(r["sample_id"])
        if sid not in by_sample or int(r["stage_index"]) > int(by_sample[sid]["stage_index"]):
            by_sample[sid] = r
    return [by_sample[sid] for sid in sorted(by_sample)]


def greedy_match_rate(model, tokenizer, embedding: torch.Tensor, ref_ids: list[int], device) -> float:
    """Generate len(ref_ids) tokens from the compression embedding; return match rate."""
    L = len(ref_ids)
    if L == 0:
        return float("nan")
    emb = embedding.to(device=device, dtype=model.dtype)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    inputs_embeds = emb.unsqueeze(0)  # [1, C, H]
    _, gen_tensor = generate_from_compression(model, tokenizer, inputs_embeds, max_new_tokens=L, return_generated_ids=True)
    gen_ids = gen_tensor[0, :L].tolist()
    ref = torch.tensor(ref_ids[: len(gen_ids)])
    gen = torch.tensor(gen_ids[: len(ref)])
    return (gen == ref).float().mean().item()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    subdir = "compressed_prefixes" if args.dataset_type == "full" else "progressive_prefixes"
    ds_path = os.path.join(args.run_dir, subdir)
    if not os.path.isdir(ds_path):
        raise SystemExit(f"dataset not found: {ds_path}")

    cache_path = os.path.join(args.run_dir, CACHE_FILENAME)
    if os.path.isfile(cache_path) and not args.overwrite:
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("version") == CACHE_VERSION and cached.get("dataset_type") == subdir and cached.get("seed") == args.seed:
            print(f"[skip] cache present: {cache_path} (mean={cached['greedy_match_mean']:.4f}, n={cached['n_samples']})")
            return

    ds = load_from_disk(ds_path)
    rows = select_rows(ds, args.dataset_type)
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]

    model_checkpoint = rows[0]["model_checkpoint"]
    dtype = DTYPE_MAP.get(str(rows[0].get("dtype", "")).lower(), torch.bfloat16)
    print(f"Loading {model_checkpoint} ({dtype}) on {args.device} for {len(rows)} samples (offset={args.offset})...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=dtype).to(args.device)
    model.eval()

    per_sample = []
    for r in tqdm(rows, desc="greedy", ncols=100):
        ref = reference_ids(tokenizer, r["text"], int(r["num_input_tokens"]))
        emb = torch.tensor(r["embedding"], dtype=torch.float32)
        match = greedy_match_rate(model, tokenizer, emb, ref, args.device)
        per_sample.append({"sample_id": int(r["sample_id"]), "match": match, "L": len(ref)})

    # Merge with existing cache when using offset (appending new samples).
    if args.offset and os.path.isfile(cache_path):
        with open(cache_path) as f:
            existing = json.load(f)
        seen_ids = {s["sample_id"] for s in existing.get("per_sample", [])}
        merged = existing.get("per_sample", []) + [s for s in per_sample if s["sample_id"] not in seen_ids]
        per_sample = merged

    matches = np.array([s["match"] for s in per_sample], dtype=np.float64)
    payload = {
        "version": CACHE_VERSION,
        "dataset_type": subdir,
        "model_checkpoint": model_checkpoint,
        "seed": args.seed,
        "n_samples": len(per_sample),
        "greedy_match_mean": float(np.mean(matches)),
        "greedy_match_std": float(np.std(matches)),
        "per_sample": per_sample,
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"\nSaved {cache_path}: mean={payload['greedy_match_mean']:.4f} std={payload['greedy_match_std']:.4f} n={payload['n_samples']}"
    )


if __name__ == "__main__":
    main()
