"""Downstream multiple-choice evaluation under Full Cramming (paper Section 5.6, Tables 5 & 10).

Replaces the legacy scripts/hellaswag_compress_evaluate.py and
scripts/arc_compress_evaluate.py with a single benchmark-agnostic entry-point
that leverages the new analysis primitives:

    - cramming via ``FullCrammingTrainer`` (paper-faithful AdamW defaults,
      cosine_with_min_lr scheduler, alignment / low-dim hooks available);
    - 8 PPL variants from ``analysis.downstream_eval`` (Table 10 of the paper);
    - optional attention-knockout intervention sweep via
      ``analysis.attention_intervention``;
    - convergence (== teacher-forced reconstruction accuracy at cramming-time)
      saved per sample so an "only-fully-converged" subset can be reported.

Output: ``--output_dir/downstream_eval.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from compression_horizon.analysis import (
    PPL_VARIANT_KEYS,
    compute_ppl_baseline_batch,
    compute_ppl_compression_batch,
    predict_best_continuation,
    summarize_downstream,
)
from compression_horizon.analysis.attention_intervention import (
    build_intervention_result,
    build_intervention_summary,
    evaluate_sample_interventions,
    get_decoder_layers,
    print_intervention_summary,
)
from compression_horizon.train import FullCrammingTrainer
from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.utils.launch import (
    freeze_model_parameters,
    get_device,
    resolve_torch_dtype,
    set_launch_seed,
)
from compression_horizon.utils.tokens import count_text_characters, count_text_tokens

# ---------------------------------------------------------------------------
# Benchmark loaders (HellaSwag + two ARC subsets)
# ---------------------------------------------------------------------------


def _load_hellaswag(num_samples: int) -> list[dict]:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.select(range(min(num_samples, len(ds))))
    return [
        {
            "id": row["ind"],
            "prefix": row["ctx"],
            "endings": list(row["endings"]),
            "label": int(row["label"]),
            "answer_key": str(row["label"]),
        }
        for row in ds
    ]


def _arc_answer_key_to_index(answer_key: str, labels: list[str]) -> int:
    if answer_key in labels:
        return labels.index(answer_key)
    if answer_key.isdigit():
        return int(answer_key) - 1
    return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(answer_key.upper(), 0)


def _load_arc(num_samples: int, subset: str) -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", subset, split="validation")
    ds = ds.select(range(min(num_samples, len(ds))))
    instances = []
    for row in ds:
        choices = row["choices"]
        label = _arc_answer_key_to_index(row["answerKey"], choices["label"])
        instances.append(
            {
                "id": row["id"],
                "prefix": row["question"],
                "endings": list(choices["text"]),
                "label": label,
                "answer_key": row["answerKey"],
            }
        )
    return instances


def _load_benchmark(name: str, num_samples: int) -> list[dict]:
    if name == "hellaswag":
        return _load_hellaswag(num_samples)
    if name == "arc-easy":
        return _load_arc(num_samples, "ARC-Easy")
    if name == "arc-challenge":
        return _load_arc(num_samples, "ARC-Challenge")
    raise ValueError(f"Unknown benchmark: {name}")


# ---------------------------------------------------------------------------
# Cramming via FullCrammingTrainer
# ---------------------------------------------------------------------------


def _resolve_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _build_tokenized_dataset(prefixes: list[str], tokenizer, max_sequence_length: int, add_special_tokens: bool) -> Dataset:
    """Tokenize the list of prefix texts into a HF Dataset with input_ids + attention_mask."""
    raw = Dataset.from_dict({"text": prefixes})

    def tokenize_fn(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length,
            add_special_tokens=add_special_tokens,
        )

    tokenized = raw.map(tokenize_fn, remove_columns=raw.column_names)
    return tokenized.with_format("torch")


def _build_cramming_args(args, cramming_dir: str, batch_size: int) -> MyTrainingArguments:
    """Programmatically build MyTrainingArguments for Full Cramming on benchmark prefixes."""
    base = MyTrainingArguments(output_dir=cramming_dir)
    return replace(
        base,
        output_dir=cramming_dir,
        logging_dir=cramming_dir,
        model_checkpoint=args.model_checkpoint,
        dataset_name=args.benchmark,
        max_sequence_length=args.max_sequence_length,
        max_optimization_steps_per_sample=args.max_optimization_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=args.learning_rate,
        embedding_init_method=args.embedding_init_method,
        loss_type=args.loss_type,
        hybrid_alpha=args.hybrid_alpha,
        num_alignment_layers=args.num_alignment_layers,
        inverted_alignment=args.inverted_alignment,
        low_dim_projection=False,
        low_dim_train=False,
        progressive_train=False,
        warmup_steps=100,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.9,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-3},
        dtype=args.dtype,
        random_seed=args.random_seed,
        report_to=[],
        save_strategy="no",
        logging_strategy="no",
        ddp_find_unused_parameters=False,
        load_best_model_at_end=False,
        no_bos_token=not args.bos_token,
        # Paper-faithful: don't drop the last (incomplete) batch. ARC-Challenge
        # has 299 / 64 = 4 full batches + 43 leftover; with the default
        # (drop_last=True) those 43 would silently disappear from cramming.
        dataloader_drop_last=False,
    )


def _run_cramming(model, tokenizer, instances, args, cramming_dir, add_special_tokens):
    """Run FullCrammingTrainer; return tensor [N, num_compression_tokens, hidden]."""
    prefixes = [inst["prefix"] for inst in instances]
    dataset = _build_tokenized_dataset(
        prefixes,
        tokenizer,
        args.max_sequence_length,
        add_special_tokens=add_special_tokens,
    )
    batch_size = args.cram_batch_size if args.cram_batch_size is not None else len(prefixes)
    cram_args = _build_cramming_args(args, cramming_dir, batch_size=batch_size)
    print(
        f"Cramming {len(prefixes)} prefixes for {cram_args.max_optimization_steps_per_sample} steps "
        f"(lr={cram_args.learning_rate}, loss={cram_args.loss_type})"
    )
    trainer = FullCrammingTrainer(
        model=model,
        processing_class=tokenizer,
        args=cram_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    embeddings_path = os.path.join(cramming_dir, "compression_embeddings.pt")
    return torch.load(embeddings_path, map_location="cpu")


# ---------------------------------------------------------------------------
# Per-sample reconstruction accuracy (convergence at eval time)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reconstruction_convergence(
    model,
    tokenizer,
    compression_embedding,
    prefix,
    num_compression_tokens,
    device,
    add_special_tokens,
):
    """Teacher-forced reconstruction accuracy on the original prefix."""
    enc = tokenizer(
        prefix,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_emb = model.get_input_embeddings()(input_ids)
    comp = compression_embedding.unsqueeze(0).to(token_emb.dtype).to(device)
    united_emb = torch.cat([comp, token_emb], dim=1)
    united_mask = torch.cat(
        [
            torch.ones((1, num_compression_tokens), dtype=attention_mask.dtype, device=device),
            attention_mask,
        ],
        dim=1,
    )
    outputs = model(inputs_embeds=united_emb, attention_mask=united_mask)
    seq_len = int(attention_mask.sum().item())
    pred = outputs.logits[0, num_compression_tokens - 1 : num_compression_tokens + seq_len - 1].argmax(dim=-1)
    target = input_ids[0, :seq_len]
    return float((pred == target).float().mean().item())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Downstream MC eval under Full Cramming (paper §5.6, Tables 5 & 10).")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["hellaswag", "arc-easy", "arc-challenge"],
    )
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--max_optimization_steps", type=int, default=5000)
    parser.add_argument(
        "--cram_batch_size",
        type=int,
        default=None,
        help="Override per-device cramming batch size (default: all samples in one batch). "
        "Reduce if you OOM on a smaller GPU.",
    )
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--embedding_init_method", default="random0.02")
    parser.add_argument("--num_compression_tokens", type=int, default=1)
    parser.add_argument(
        "--loss_type",
        default="cross_entropy",
        choices=["cross_entropy", "l2", "l1", "cosine"],
    )
    parser.add_argument("--hybrid_alpha", type=float, default=None)
    parser.add_argument("--num_alignment_layers", type=int, default=0)
    parser.add_argument("--inverted_alignment", action="store_true", default=False)
    parser.add_argument(
        "--no-bos-token",
        "--no_bos_token",
        dest="bos_token",
        action="store_false",
        default=True,
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--only_full_convergence",
        action="store_true",
        default=False,
        help="When summarising, restrict compression variants to samples with reconstruction accuracy == 1.0.",
    )
    parser.add_argument("--rerun_cramming", action="store_true", default=False)
    parser.add_argument(
        "--intervention",
        action="store_true",
        default=False,
        help="Also run the attention-knockout intervention sweep (per-layer / cumulative / reverse) per sample.",
    )
    parser.add_argument("--skip_per_layer", action="store_true", default=False)
    parser.add_argument("--skip_cumulative", action="store_true", default=False)
    parser.add_argument("--skip_reverse_cumulative", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cramming_dir = os.path.join(args.output_dir, "cramming")
    os.makedirs(cramming_dir, exist_ok=True)

    set_launch_seed(args.random_seed)
    device = get_device()
    torch_dtype = resolve_torch_dtype(args.dtype)
    attn_impl = _resolve_attn_implementation()
    print(f"Device: {device}; dtype: {torch_dtype}; attn: {attn_impl}")

    # Model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, dtype=torch_dtype, attn_implementation=attn_impl)
    freeze_model_parameters(model)
    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_bos_supported = hasattr(tokenizer, "add_bos_token")
    if not args.bos_token:
        if add_bos_supported:
            tokenizer.add_bos_token = False
            add_special_tokens = True
        else:
            add_special_tokens = False
    else:
        add_special_tokens = True

    num_model_layers = len(get_decoder_layers(model)) if args.intervention else None
    if args.intervention:
        print(f"Intervention sweep enabled; model has {num_model_layers} layers")

    # Benchmark
    instances = _load_benchmark(args.benchmark, args.num_samples)
    print(f"Loaded {len(instances)} instances from {args.benchmark}")

    # Cramming (cached). We prefer the concatenated .pt tensor; if its row count
    # does not match (older runs only saved the last batch), fall back to
    # reconstructing the full tensor from the per-sample compressed_prefixes/ HF
    # Dataset so we don't have to re-cram.
    embeddings_path = os.path.join(cramming_dir, "compression_embeddings.pt")
    dataset_path = os.path.join(cramming_dir, "compressed_prefixes")

    def _load_embeddings_from_dataset(path: str) -> torch.Tensor:
        ds = Dataset.load_from_disk(path)
        rows = sorted(list(ds), key=lambda r: int(r["sample_id"]))
        tensors = [torch.tensor(r["embedding"], dtype=torch.float32) for r in rows]
        if tensors[0].dim() == 1:
            tensors = [t.unsqueeze(0) for t in tensors]
        return torch.stack(tensors, dim=0)

    embeddings: torch.Tensor | None = None
    if args.rerun_cramming or not (os.path.exists(embeddings_path) or os.path.exists(dataset_path)):
        embeddings = _run_cramming(model, tokenizer, instances, args, cramming_dir, add_special_tokens)
    else:
        if os.path.exists(embeddings_path):
            cached = torch.load(embeddings_path, map_location="cpu")
            if cached.shape[0] == len(instances):
                print(f"Loading cached compression embeddings from {embeddings_path}")
                embeddings = cached
            else:
                print(
                    f"Cached {embeddings_path} has {cached.shape[0]} rows "
                    f"(expected {len(instances)}); falling back to {dataset_path}/."
                )
        if embeddings is None and os.path.exists(dataset_path):
            print(f"Reconstructing full embedding tensor from {dataset_path}/")
            embeddings = _load_embeddings_from_dataset(dataset_path)
            # Re-save the concatenated tensor so subsequent runs use the fast path.
            torch.save(embeddings, embeddings_path)
        if embeddings is None:
            embeddings = _run_cramming(model, tokenizer, instances, args, cramming_dir, add_special_tokens)
    if embeddings.shape[0] != len(instances):
        raise ValueError(f"Embedding tensor has {embeddings.shape[0]} rows but expected {len(instances)} instances")

    # Per-sample eval (8 variants + optional intervention)
    records: list[dict] = []
    intervention_data_list: list[dict | None] = []
    for i, inst in enumerate(tqdm(instances, desc="eval")):
        prefix = inst["prefix"]
        endings = inst["endings"]
        label = inst["label"]
        embedding_i = embeddings[i].to(device).to(torch_dtype)
        if embedding_i.dim() == 1:
            embedding_i = embedding_i.unsqueeze(0)
        num_compression_tokens = embedding_i.shape[0]

        # convergence on the prefix the embedding was trained for
        convergence = _reconstruction_convergence(
            model,
            tokenizer,
            embedding_i,
            prefix,
            num_compression_tokens,
            device,
            add_special_tokens,
        )
        is_fully_converged = convergence >= 1.0

        # baselines (variants 1, 2)
        baseline_full_ppls, baseline_ending_ppls = compute_ppl_baseline_batch(
            model, tokenizer, prefix, endings, device, add_special_tokens
        )

        # compression with prefix (variants 3, 4, 5) — note trailing space (legacy parity)
        comp_full_ppls, comp_edge_ppls, comp_end_ppls = compute_ppl_compression_batch(
            model,
            tokenizer,
            embedding_i,
            prefix + " ",
            endings,
            device,
            add_special_tokens,
        )

        # compression replacing prefix (variants 6, 7, 8)
        only_full_ppls, only_edge_ppls, only_end_ppls = compute_ppl_compression_batch(
            model,
            tokenizer,
            embedding_i,
            "",
            endings,
            device,
            add_special_tokens,
        )

        # token / char counts of the correct ending (used for token/char-normalised accuracies)
        token_count = None
        char_count = None
        if 0 <= label < len(endings):
            full_text = f"{prefix} {endings[label]}"
            token_count = count_text_tokens(tokenizer, full_text, add_special_tokens=add_special_tokens)
            char_count = count_text_characters(full_text)

        variant_ppls = {
            "baseline": baseline_full_ppls,
            "baseline_endings": baseline_ending_ppls,
            "compression": comp_full_ppls,
            "compression_edge": comp_edge_ppls,
            "compression_endings": comp_end_ppls,
            "compression_only": only_full_ppls,
            "compression_only_edge": only_edge_ppls,
            "compression_only_endings": only_end_ppls,
        }
        record = {
            "sample_id": i,
            "instance_id": inst.get("id"),
            "prefix": prefix,
            "endings": endings,
            "label": label,
            "answer_key": inst.get("answer_key"),
            "convergence": convergence,
            "is_fully_converged": is_fully_converged,
            "lengths": {"tokens": token_count, "characters": char_count},
        }
        for variant_name, ppls in variant_ppls.items():
            pred = predict_best_continuation(ppls)
            record[variant_name] = {
                "ppls": ppls,
                "predicted_label": pred,
                "is_correct": pred == label,
            }

        if args.intervention:
            try:
                intervention_data = evaluate_sample_interventions(
                    model=model,
                    tokenizer=tokenizer,
                    compression_embedding=embedding_i,
                    context=prefix,
                    endings=endings,
                    num_compression_tokens=num_compression_tokens,
                    num_model_layers=num_model_layers,
                    device=device,
                    add_special_tokens=add_special_tokens,
                    skip_per_layer=args.skip_per_layer,
                    skip_cumulative=args.skip_cumulative,
                    skip_reverse_cumulative=args.skip_reverse_cumulative,
                )
            except Exception as exc:
                print(f"Intervention sweep failed for sample {i}: {exc}")
                intervention_data = None
            intervention_data_list.append(intervention_data)
            if intervention_data is not None:
                record.update(build_intervention_result(intervention_data, label, num_model_layers))
        records.append(record)

    # Compute BOTH views from the same records — one run gives you Table 5
    # (all samples) and Table 10 (perfectly-reconstructed subset) at once.
    summary_all = summarize_downstream(records, only_full_convergence=False)
    summary_perfect = summarize_downstream(records, only_full_convergence=True)
    summary = summary_perfect if args.only_full_convergence else summary_all

    output: dict = {
        "config": {
            "benchmark": args.benchmark,
            "model_checkpoint": args.model_checkpoint,
            "num_samples": len(instances),
            "max_sequence_length": args.max_sequence_length,
            "max_optimization_steps": args.max_optimization_steps,
            "learning_rate": args.learning_rate,
            "embedding_init_method": args.embedding_init_method,
            "loss_type": args.loss_type,
            "hybrid_alpha": args.hybrid_alpha,
            "num_alignment_layers": args.num_alignment_layers,
            "inverted_alignment": args.inverted_alignment,
            "num_compression_tokens": args.num_compression_tokens,
            "bos_token": args.bos_token,
            "only_full_convergence": args.only_full_convergence,
            "intervention": args.intervention,
            "random_seed": args.random_seed,
        },
        # `summary` is the CLI-chosen view (kept for backwards compatibility).
        "summary": summary,
        # Both views are always present; analyzers / expected.json pick which one.
        "summary_all_samples": summary_all,
        "summary_perfectly_reconstructed": summary_perfect,
        "samples": records,
    }
    if args.intervention:
        output["intervention_summary"] = build_intervention_summary(
            records,
            num_model_layers,
            skip_per_layer=args.skip_per_layer,
            skip_cumulative=args.skip_cumulative,
            skip_reverse_cumulative=args.skip_reverse_cumulative,
        )
        output["num_model_layers"] = num_model_layers

    output_path = Path(args.output_dir) / "downstream_eval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {output_path}")

    # Pretty-print BOTH summary views.
    def _print_summary_block(title: str, s: dict) -> None:
        print()
        print("=" * 80)
        print(f"{title} — {args.benchmark}")
        print("=" * 80)
        print(f"Samples in this view   : {s['num_samples_total']}")
        print(f"Fully converged samples: {s['num_full_convergence']}")
        print()
        header = f"  {'#':>2}  {'variant':<28}  {'acc':>8}  {'token-norm':>10}  {'char-norm':>10}  {'n_pred':>7}"
        print(header)
        print(f"  {'-'*2}  {'-'*28}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*7}")
        for idx, variant in enumerate(PPL_VARIANT_KEYS, start=1):
            stats = s[variant]
            print(
                f"  {idx:>2}  {variant:<28}  {stats['accuracy']:>8.4f}  "
                f"{stats['token_normalized_accuracy']:>10.4f}  "
                f"{stats['char_normalized_accuracy']:>10.4f}  "
                f"{stats['total_predictions']:>7d}"
            )

    _print_summary_block("Table 5 view (all samples)", summary_all)
    _print_summary_block("Table 10 view (perfectly-reconstructed subset)", summary_perfect)

    if args.intervention:
        print()
        print_intervention_summary(
            output["intervention_summary"],
            num_model_layers,
            summary["baseline"]["accuracy"],
        )


if __name__ == "__main__":
    # Silence noisy log_softmax tracebacks when bf16 overflows infrequently.
    if hasattr(F, "scaled_dot_product_attention"):
        torch.backends.cuda.enable_flash_sdp(True)
    main()
