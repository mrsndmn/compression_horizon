#!/usr/bin/env python3
"""Standard next-token (causal-LM) finetuning for a (truncated) base checkpoint,
built on the HuggingFace ``Trainer`` with ``torch.compile`` and bf16.

Motivation
----------
The transformer-depth ablation (``scripts/checkpoints/make_first_last_layers_ckpt.py``)
produces models that keep only the first ``N`` and last ``N`` decoder layers, with
the middle block removed. The retained layers still carry their original
pretrained weights but were never trained to talk to each other across the
excised gap, so the stitched model is degraded. This script finetunes such a
checkpoint with a plain language-modeling objective on fineweb-edu to recover
capability, after which we re-measure progressive-cramming compression capacity.

Data
----
Sequences are *packed*, run_clm-style: documents are tokenized without padding,
concatenated, and chunked into fixed ``--max_sequence_length`` blocks (remainder
dropped). Packing avoids padding waste so a step of ``B`` sequences is ``B *
seq_len`` real training tokens. The packed dataset is cached on the shared
filesystem keyed by ``--dataset_cache_key`` (set identically for every truncated
checkpoint so the 8-GPU jobs share one cache rather than re-tokenizing).

Usage (single GPU / smoke)::

    python scripts/finetune_causal_lm.py \
        --model_checkpoint artifacts/checkpoints/SmolLM2-1.7B-firstlast2 \
        --dataset_name LarryLovestein/pg19_1k --limit_dataset_items 64 \
        --max_sequence_length 128 --per_device_train_batch_size 2 \
        --max_steps 2 --no_torch_compile --dtype float32 \
        --output_dir /tmp/ft_smoke

Usual launch is via ``scripts/jobs/run_jobs_finetune_truncated.py`` ->
``scripts/jobs/multigpu.sh`` (8x A100, ``accelerate launch``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from itertools import chain

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Repo helper (PYTHONPATH=./src is set by the job env and multigpu.sh).
from compression_horizon.utils.launch import resolve_torch_dtype

DEFAULT_CACHE_DIR = "artifacts/cache/packed_lm_datasets"


def build_packed_dataset(
    *,
    dataset_name: str,
    split: str,
    tokenizer,
    seq_len: int,
    limit_docs: int | None,
    cache_dir: str,
    cache_key: str,
    num_proc: int,
) -> Dataset:
    """Tokenize + pack documents into fixed ``seq_len`` blocks (run_clm-style).

    The result is cached under ``cache_dir`` keyed by all parameters that affect
    the token stream (including ``cache_key``, which the launcher pins to the
    base-model name so every truncated checkpoint reuses one cache).
    """
    params = {
        "dataset": dataset_name,
        "split": split,
        "seq_len": seq_len,
        "limit_docs": limit_docs,
        "cache_key": cache_key,
    }
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"packed_{key}")
    if os.path.exists(cache_path):
        print(f"Loading packed dataset from cache: {cache_path}")
        return Dataset.load_from_disk(cache_path).with_format("torch")

    print(f"Building packed dataset (dataset={dataset_name}, seq_len={seq_len}, limit_docs={limit_docs}) ...")
    kwargs: dict = {"num_proc": num_proc, "split": split}
    if dataset_name == "HuggingFaceFW/fineweb-edu":
        # Mirror scripts/activation_distillation.load_or_create_tokenized_dataset.
        del kwargs["num_proc"]
        del kwargs["split"]
        kwargs["data_files"] = [f"sample/10BT/{i:03}_00000.parquet" for i in range(14)]
    elif dataset_name in ("LarryLovestein/pg19_1k", "LarryLovestein/fanfics_1k"):
        kwargs["split"] = "train"

    raw = load_dataset(dataset_name, **kwargs)
    if dataset_name == "HuggingFaceFW/fineweb-edu":
        raw = raw["train"]

    if limit_docs is not None and limit_docs < len(raw):
        raw = raw.select(range(limit_docs))

    def _tokenize(examples):
        # No padding/truncation: we pack, so keep the full token stream.
        return {"input_ids": tokenizer(examples["text"])["input_ids"]}

    tokenized = raw.map(_tokenize, batched=True, num_proc=num_proc, remove_columns=raw.column_names)

    def _group(examples):
        concatenated = list(chain(*examples["input_ids"]))
        total_len = (len(concatenated) // seq_len) * seq_len
        blocks = [concatenated[i : i + seq_len] for i in range(0, total_len, seq_len)]
        return {"input_ids": blocks, "labels": [b[:] for b in blocks]}

    packed = tokenized.map(_group, batched=True, num_proc=num_proc)
    packed = packed.filter(lambda ex: len(ex["input_ids"]) == seq_len, num_proc=num_proc)

    os.makedirs(cache_dir, exist_ok=True)
    packed.save_to_disk(cache_path)
    print(f"Saved packed dataset ({len(packed)} blocks of {seq_len} tokens) to {cache_path}")
    return packed.with_format("torch")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_checkpoint", required=True, help="Path/name of the checkpoint to finetune.")
    p.add_argument("--output_dir", required=True, help="Where to save the finetuned checkpoint.")
    p.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_split", default="train")
    p.add_argument(
        "--dataset_cache_key",
        default=None,
        help="Cache namespace for the packed dataset; pin to the base-model name so sibling "
        "jobs share one cache. Defaults to the model_checkpoint basename.",
    )
    p.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--max_sequence_length", type=int, default=1024)
    p.add_argument(
        "--limit_dataset_items",
        type=int,
        default=400000,
        help="Cap on raw documents tokenized (bounds packing time/cache size). "
        "Trainer repeats data if --max_steps exceeds one epoch.",
    )
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=0, help="If >0, save a checkpoint every N steps.")
    p.add_argument("--dataloader_num_workers", type=int, default=8)
    p.add_argument("--dtype", default="bf16", choices=["auto", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--no_torch_compile", action="store_true", help="Disable torch.compile (auto-off on CPU).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cuda = torch.cuda.is_available()
    torch_dtype = resolve_torch_dtype(args.dtype)
    # bf16/compile only make sense on GPU; gate so CPU smoke tests run cleanly.
    use_bf16 = cuda and args.dtype in ("bf16", "bfloat16", "auto")
    use_compile = cuda and not args.no_torch_compile
    cache_key = args.dataset_cache_key or os.path.basename(args.model_checkpoint.rstrip("/"))

    print(f"Finetune {args.model_checkpoint} -> {args.output_dir}")
    print(f"  cuda={cuda} bf16={use_bf16} torch_compile={use_compile} dtype={args.dtype}")
    print(
        f"  max_steps={args.max_steps} per_device_bs={args.per_device_train_batch_size} "
        f"grad_accum={args.gradient_accumulation_steps} seq_len={args.max_sequence_length}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy=("steps" if args.save_steps > 0 else "no"),
        save_steps=(args.save_steps if args.save_steps > 0 else 500),
        save_total_limit=1,
        bf16=use_bf16,
        torch_compile=use_compile,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=False,
        report_to=["tensorboard"],
        seed=args.seed,
    )

    # rank 0 builds + caches the packed dataset; other ranks load from cache.
    with training_args.main_process_first(desc="packed dataset"):
        train_dataset = build_packed_dataset(
            dataset_name=args.dataset_name,
            split=args.dataset_split,
            tokenizer=tokenizer,
            seq_len=args.max_sequence_length,
            limit_docs=args.limit_dataset_items,
            cache_dir=args.cache_dir,
            cache_key=cache_key,
            num_proc=args.dataloader_num_workers,
        )
    print(f"  train blocks: {len(train_dataset)}")

    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, dtype=torch_dtype)
    model.config.use_cache = False  # required for training (and torch.compile stability)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save the final finetuned checkpoint (Trainer writes only on the main process).
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        print(f"DONE. Saved finetuned checkpoint to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
