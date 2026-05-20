"""Attention-hijacking diagnostics."""

from __future__ import annotations

import math
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from compression_horizon.analysis.attention_intervention import EagerAttentionContext
from compression_horizon.utils.launch import get_device

# Note: `build_united_input` is imported lazily inside the function below to
# avoid a circular import path
# (analysis → train.inputs → train → trainers/full_cramming → analysis).


def _default_target_prefix_lengths(max_length: int) -> list[int]:
    """Geometric schedule of suffix lengths capped at the sample length."""
    candidates = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    return [s for s in candidates if s <= max_length]


@torch.no_grad()
def compute_attention_mass_profile(
    model: PreTrainedModel,
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    target_prefix_lengths: list[int],
) -> list[float]:
    """Return per-layer m̄_l (eq. 8) given pre-built embeddings.

    ``token_embeddings``: [1, T, H] with the *first* position being the token we
    are measuring attention onto (compression token OR BOS).
    ``attention_mask``: [1, T].
    ``target_prefix_lengths``: list of suffix lengths ``s`` (s >= 2). Values
    above T are skipped.
    """
    if token_embeddings.shape[0] != 1:
        raise ValueError("compute_attention_mass_profile expects a single-sample batch!")
    total_len = int(attention_mask.sum().item())
    valid_s = [s for s in target_prefix_lengths if 2 <= s <= total_len]
    if not valid_s:
        raise ValueError(f"No valid suffix lengths in {target_prefix_lengths} for sequence of length {total_len}!")

    with EagerAttentionContext(model):
        outputs = model(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    # Each attention tensor: [1, head, sequence, sequence]; average over heads -> [1, sequence, sequence].
    layers_attn = [layer.mean(dim=1)[0] for layer in outputs.attentions]
    num_layers = len(layers_attn)

    profile: list[float] = []
    for layer_attn in layers_attn:
        # column 0 = key position of the token under inspection
        mass_per_s = []
        for s in valid_s:
            queries = layer_attn[1:s, 0]  # q = 1..s-1
            mass_per_s.append(queries.mean().item())
        profile.append(float(sum(mass_per_s) / len(mass_per_s)))

    assert len(profile) == num_layers
    return profile


@torch.no_grad()
def compute_sample_profiles(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    compression_token_embedding: torch.Tensor,
    context: str,
    num_compression_tokens: int = 1,
    target_prefix_lengths: Optional[list[int]] = None,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> tuple[list[float], list[float], list[int]]:
    """Compute (m̄^comp, m̄^bos, used_lengths) profiles for a single sample."""
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    encoded = tokenizer(
        context,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    embed_fn = model.get_input_embeddings()
    token_embeddings = embed_fn(input_ids)  # [1, sequence, hidden]

    # ---- Compression run: prepend ``num_compression_tokens`` learned embeddings.
    compression_token_embeddings = compression_token_embedding.unsqueeze(0)
    compression_attention_mask = torch.ones((1, num_compression_tokens), dtype=attention_mask.dtype, device=device)
    from compression_horizon.train.inputs import build_united_input  # deferred (circular)

    united_token_embeddings, united_attention_mask = build_united_input(
        compression_token_embeddings,
        compression_attention_mask,
        token_embeddings,
        attention_mask,
    )
    total_len_comp = int(united_attention_mask.sum().item())
    lengths = target_prefix_lengths or _default_target_prefix_lengths(total_len_comp)
    used_lengths = [s for s in lengths if 2 <= s <= total_len_comp]

    comp_profile = compute_attention_mass_profile(model, united_token_embeddings, united_attention_mask, used_lengths)

    # ---- BOS run: prepend a single BOS-token embedding (paper baseline).
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if bos_id is None:
        raise ValueError("Tokenizer has neither bos_token_id nor eos_token_id; cannot build BOS profile")
    bos_embed = embed_fn(torch.tensor([[bos_id]], device=device)).to(token_embeddings.dtype)  # [1, 1, hidden]
    bos_inputs = torch.cat([bos_embed, token_embeddings], dim=1)
    bos_mask = torch.cat(
        [torch.ones((1, 1), dtype=attention_mask.dtype, device=device), attention_mask],
        dim=1,
    )
    total_len_bos = int(bos_mask.sum().item())
    bos_lengths = [s for s in lengths if 2 <= s <= total_len_bos]
    bos_profile = compute_attention_mass_profile(model, bos_inputs, bos_mask, bos_lengths)

    return comp_profile, bos_profile, used_lengths


def pearson_correlation(profile_a: list[float], profile_b: list[float]) -> float:
    """Pearson correlation between two equal-length per-layer profiles (NaN if degenerate)."""
    if len(profile_a) != len(profile_b):
        raise ValueError(f"Profile length mismatch: {len(profile_a)} vs {len(profile_b)}!")
    n = len(profile_a)
    if n < 2:
        return float("nan")
    a = torch.tensor(profile_a, dtype=torch.float64)
    b = torch.tensor(profile_b, dtype=torch.float64)
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = math.sqrt(float((a_centered**2).sum()) * float((b_centered**2).sum()))
    if denominator == 0.0:
        return float("nan")
    return float((a_centered * b_centered).sum() / denominator)


def summarize_hijacking(
    compression_profiles: list[list[float]],
    bos_profiles: list[list[float]],
) -> dict:
    """Aggregate per-sample profiles into Table-3-style summary statistics."""
    if len(compression_profiles) != len(bos_profiles):
        raise ValueError(f"#compression_profiles ({len(compression_profiles)}) " f"!= #bos_profiles ({len(bos_profiles)})")
    if not compression_profiles:
        return {
            "compression_mass": {"mean": 0.0, "std": 0.0},
            "bos_mass": {"mean": 0.0, "std": 0.0},
            "correlation": {"mean": 0.0, "std": 0.0},
            "avg_compression_profile": [],
            "avg_bos_profile": [],
            "num_samples": 0,
        }

    num_layers = len(compression_profiles[0])
    comp = torch.tensor(compression_profiles, dtype=torch.float64)  # [N, L]
    bos = torch.tensor(bos_profiles, dtype=torch.float64)

    # Per-sample max attention mass over layers, reported as percentages.
    comp_max = comp.max(dim=1).values * 100.0
    bos_max = bos.max(dim=1).values * 100.0

    correlations = torch.tensor(
        [pearson_correlation(c, b) for c, b in zip(compression_profiles, bos_profiles)],
        dtype=torch.float64,
    )
    finite_corr = correlations[torch.isfinite(correlations)]

    return {
        "compression_mass": {
            "mean": float(comp_max.mean()),
            "std": float(comp_max.std(unbiased=False)),
        },
        "bos_mass": {
            "mean": float(bos_max.mean()),
            "std": float(bos_max.std(unbiased=False)),
        },
        "correlation": {
            "mean": float(finite_corr.mean()) if finite_corr.numel() else float("nan"),
            "std": (float(finite_corr.std(unbiased=False)) if finite_corr.numel() else float("nan")),
        },
        "avg_compression_profile": [float(comp[:, layer_idx].mean()) for layer_idx in range(num_layers)],
        "avg_bos_profile": [float(bos[:, layer_idx].mean()) for layer_idx in range(num_layers)],
        "num_samples": int(comp.shape[0]),
    }
