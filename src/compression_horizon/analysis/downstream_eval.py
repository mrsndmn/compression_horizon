"""Downstream multiple-choice evaluation under compression (paper Section 5.6, Tables 5 & 10)."""

from __future__ import annotations

import math
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from compression_horizon.analysis.perplexity import (
    estimate_token_perplexity,
    estimate_token_perplexity_full_labels,
)

# Note: `build_united_input` is imported lazily inside the function below to
# avoid a circular import path
# (analysis → train.inputs → train → trainers/full_cramming → analysis).

PPL_VARIANT_KEYS: tuple[str, ...] = (
    "baseline",
    "baseline_endings",
    "compression",
    "compression_edge",
    "compression_endings",
    "compression_only",
    "compression_only_edge",
    "compression_only_endings",
)

# Variants that are computed with compression embedding prepended. The other
# two (baseline*) use the original prefix only and live on the full-sample
# denominator (they are always reported on every sample).
_COMPRESSION_VARIANTS: tuple[str, ...] = (
    "compression",
    "compression_edge",
    "compression_endings",
    "compression_only",
    "compression_only_edge",
    "compression_only_endings",
)


@torch.no_grad()
def compute_ppl_baseline_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    context: str,
    endings: list[str],
    device: torch.device,
    add_special_tokens: bool = True,
) -> tuple[list[float], list[float]]:
    """Return ``(full_ppls, endings_only_ppls)`` for the four candidate endings."""
    model = model.to(device)
    model.eval()

    if not context:
        return [], []

    full_texts = [f"{context} {ending}" for ending in endings]
    encoded = tokenizer(
        full_texts,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    context_ids = tokenizer(f"{context} ", add_special_tokens=add_special_tokens)["input_ids"]
    context_len = len(context_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    full_ppls: list[float] = []
    endings_ppls: list[float] = []
    for i in range(len(full_texts)):
        seq_len = int(attention_mask[i].sum().item())
        sample_logits = outputs.logits[i : i + 1, :seq_len]  # [1, sequence, vocabulary]
        sample_input_ids = input_ids[i : i + 1, :seq_len]  # [1, sequence]

        ppl = estimate_token_perplexity(sample_logits, sample_input_ids)
        full_ppls.append(ppl if not math.isnan(ppl) else float("inf"))

        ending_logits = sample_logits[:, context_len - 1 :, :]
        ending_ids = sample_input_ids[:, context_len - 1 :]
        ending_ppl = estimate_token_perplexity(ending_logits, ending_ids)
        endings_ppls.append(ending_ppl if not math.isnan(ending_ppl) else float("inf"))
    return full_ppls, endings_ppls


@torch.no_grad()
def compute_ppl_compression_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    compression_token_embeddings: torch.Tensor,
    context: str,
    endings: list[str],
    device: torch.device,
    add_special_tokens: bool = True,
) -> tuple[list[float], list[float], list[float]]:
    """Return ``(full_ppls, edge_ppls, endings_ppls)`` under compression.

    Caller-owned separator: ``context`` is concatenated with each ``ending``
    verbatim (no space inserted). Pass ``context + " "`` for the natural-text
    case and ``context=""`` for compression-only (no prefix re-text). This
    keeps a single function usable for both Section 5.6 regimes without a
    leading space appearing in compression-only mode.
    """
    model = model.to(device)
    model.eval()

    full_texts = [f"{context}{ending}" for ending in endings]
    encoded = tokenizer(
        full_texts,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    if context:
        context_ids = tokenizer(context, add_special_tokens=add_special_tokens)["input_ids"]
        context_len = len(context_ids)
    else:
        context_len = 0

    token_embeddings = model.get_input_embeddings()(input_ids)
    num_compression_tokens = compression_token_embeddings.shape[0]
    batch_size = len(full_texts)

    # The same compression embedding is shared across all 4 candidate endings.
    # Broadcast it across the batch, then concat with the (already padded) text
    # token embeddings — equivalent to per-sample concat + re-pad to max-len,
    # because padding lives at the tail of token_embeddings already.
    batched_compression = compression_token_embeddings.unsqueeze(0).expand(batch_size, -1, -1).to(token_embeddings.dtype)
    compression_attention_mask = torch.ones((batch_size, num_compression_tokens), dtype=attention_mask.dtype, device=device)
    from compression_horizon.train.inputs import build_united_input  # deferred (circular)

    batch_token_embeddings, batch_attention_mask = build_united_input(
        batched_compression, compression_attention_mask, token_embeddings, attention_mask
    )

    outputs = model(inputs_embeds=batch_token_embeddings, attention_mask=batch_attention_mask)

    full_ppls: list[float] = []
    edge_ppls: list[float] = []
    endings_ppls: list[float] = []
    for i in range(len(full_texts)):
        seq_len = int(attention_mask[i].sum().item())

        # variant 3 / 6: full, exclude comp→first-prefix-token logit
        sample_logits = outputs.logits[i : i + 1, num_compression_tokens : num_compression_tokens + seq_len]
        sample_input_ids = input_ids[i : i + 1, :seq_len]
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids)
        full_ppls.append(ppl if not math.isnan(ppl) else float("inf"))

        # variant 4 / 7: full + edge (use the comp→first-prefix-token logit too)
        edge_logits = outputs.logits[i : i + 1, num_compression_tokens - 1 : num_compression_tokens + seq_len]
        edge_ppl = estimate_token_perplexity_full_labels(edge_logits, sample_input_ids)
        edge_ppls.append(edge_ppl if not math.isnan(edge_ppl) else float("inf"))

        # variant 5 / 8: endings only
        ending_start = num_compression_tokens + max(0, context_len - 1)
        ending_logits = outputs.logits[i : i + 1, ending_start : num_compression_tokens + seq_len, :]
        ending_ids = sample_input_ids[:, max(0, context_len - 1) : seq_len]
        ending_ppl = estimate_token_perplexity(ending_logits, ending_ids)
        endings_ppls.append(ending_ppl if not math.isnan(ending_ppl) else float("inf"))

    return full_ppls, edge_ppls, endings_ppls


def predict_best_continuation(ppls: list[float]) -> int:
    """Argmin of per-ending PPL → predicted label."""
    return int(torch.tensor(ppls).argmin().item())


def aggregate_variant_accuracy(
    records: list[dict],
    variant: str,
    *,
    only_full_convergence: bool,
) -> dict:
    """Compute counts and accuracies for one PPL variant across saved records.

    Each record must contain ``label``, ``convergence``, ``lengths`` (with
    ``tokens`` and ``characters``), and ``{variant}`` dict with
    ``is_correct``. Baselines are always counted on all samples; compression
    variants respect ``only_full_convergence``.
    """
    is_compression = variant in _COMPRESSION_VARIANTS

    correct_predictions = 0
    total_predictions = 0
    correct_tokens = 0
    total_tokens = 0
    correct_chars = 0
    total_chars = 0

    for r in records:
        is_converged = float(r.get("convergence", 0.0)) >= 1.0
        include = not (is_compression and only_full_convergence) or is_converged
        if not include:
            continue
        entry = r.get(variant)
        if entry is None:
            continue
        total_predictions += 1
        token_count = (r.get("lengths") or {}).get("tokens") or 0
        char_count = (r.get("lengths") or {}).get("characters") or 0
        total_tokens += token_count
        total_chars += char_count
        if entry["is_correct"]:
            correct_predictions += 1
            correct_tokens += token_count
            correct_chars += char_count

    return {
        "accuracy": (correct_predictions / total_predictions if total_predictions else 0.0),
        "token_normalized_accuracy": (correct_tokens / total_tokens if total_tokens else 0.0),
        "char_normalized_accuracy": correct_chars / total_chars if total_chars else 0.0,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions,
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "correct_characters": correct_chars,
        "total_characters": total_chars,
    }


def summarize_downstream(
    records: list[dict],
    *,
    only_full_convergence: bool = False,
) -> dict:
    """Aggregate all 8 variants of the per-instance MC outcomes.

    Returns a dict keyed by variant name (Table-10 ordering) with accuracy
    statistics; plus ``num_samples_total`` and ``num_full_convergence`` summary
    counters.
    """
    summary: dict = {
        "num_samples_total": len(records),
        "num_full_convergence": sum(1 for r in records if float(r.get("convergence", 0.0)) >= 1.0),
        "only_full_convergence": only_full_convergence,
    }
    for variant in PPL_VARIANT_KEYS:
        summary[variant] = aggregate_variant_accuracy(records, variant, only_full_convergence=only_full_convergence)
    return summary


@torch.no_grad()
def compute_continuation_nll(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prefix: str,
    continuation: str,
    compression_embedding: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> float:
    """Per-token-averaged NLL of ``continuation`` given ``prefix`` (legacy single-pair API).

    Kept for unit-test compatibility; the 8-variant pipeline uses the
    ``compute_ppl_*_batch`` helpers above instead.
    """
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    prefix_ids = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
    continuation_ids = tokenizer(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    if continuation_ids.shape[1] == 0:
        return float("nan")

    full_ids = torch.cat([prefix_ids, continuation_ids], dim=1)
    token_embeddings = model.get_input_embeddings()(full_ids)
    if compression_embedding is not None:
        compression_token_embeddings = compression_embedding.unsqueeze(0).to(token_embeddings.dtype).to(device)
        token_embeddings = torch.cat([compression_token_embeddings, token_embeddings], dim=1)
        num_compression_tokens = compression_embedding.shape[0]
    else:
        num_compression_tokens = 0

    attention_mask = torch.ones(token_embeddings.shape[:2], dtype=torch.long, device=device)
    outputs = model(inputs_embeds=token_embeddings, attention_mask=attention_mask)

    prefix_len = prefix_ids.shape[1]
    continuation_len = continuation_ids.shape[1]
    start = num_compression_tokens + prefix_len - 1
    end = num_compression_tokens + prefix_len + continuation_len - 1
    continuation_logits = outputs.logits[0, start:end]
    log_probs = torch.log_softmax(continuation_logits.float(), dim=-1)
    nll_per_token = -log_probs.gather(1, continuation_ids[0].unsqueeze(1)).squeeze(1)
    return float(nll_per_token.mean().item())
