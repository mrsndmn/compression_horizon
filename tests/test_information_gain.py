"""Numerical-parity tests for compression_horizon.analysis.information_gain.

These tests are CPU-only and use a tiny ``GPT2LMHeadModel`` constructed from
config (no model download). The goal is **byte-identical** numerical parity
with the original inline IG computation that lived in
``FullCrammingTrainer._train_full_cramming`` and
``ProgressiveCrammingTrainer._train_progressive`` before the Stage 1.1 refactor.

The reference implementation in :func:`_reference_inline_ig` is a
copy-paste-then-pruned snapshot of that inline code. If the new
:func:`compute_information_gain` ever drifts from the reference numerically,
this test will catch it.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from compression_horizon.analysis.information_gain import compute_information_gain

# ---------------------------------------------------------------------------
# Reference implementation: copy of the original inline IG block, simplified
# to match a single call site (no PCA branch — that just changes how the
# compression embedding is produced upstream, not the IG computation itself).
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reference_inline_ig(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_embeddings: torch.Tensor,
    final_compression_tokens_for_ig: torch.Tensor,
    compression_attention_mask: torch.Tensor,
    num_compression_tokens: int,
) -> list[float]:
    """Verbatim re-creation of the pre-refactor inline IG loop."""
    batch_size = input_ids.size(0)
    per_sample_info_gain: list[float] = []
    for j in range(batch_size):
        sample_input_ids = input_ids[j : j + 1]
        sample_attention_mask = attention_mask[j : j + 1]

        sample_outputs_lm = model(input_ids=sample_input_ids, attention_mask=sample_attention_mask)
        sample_logits_lm = sample_outputs_lm.logits
        sample_shift_logits_lm = sample_logits_lm[:, :-1, :].contiguous()
        sample_shift_labels_lm = sample_input_ids[:, 1:].contiguous()
        sample_shift_mask_lm = sample_attention_mask[:, 1:].contiguous()

        sample_shift_logits_lm_flat = sample_shift_logits_lm.view(-1, sample_shift_logits_lm.size(-1))
        sample_shift_labels_lm_flat = sample_shift_labels_lm.view(-1)
        sample_shift_mask_lm_flat = sample_shift_mask_lm.view(-1)

        sample_valid_mask_lm = sample_shift_mask_lm_flat.bool()
        if sample_valid_mask_lm.sum() > 0:
            sample_ce_lm_sum = F.cross_entropy(
                sample_shift_logits_lm_flat[sample_valid_mask_lm],
                sample_shift_labels_lm_flat[sample_valid_mask_lm],
                reduction="sum",
            )
            sample_H_LM_bits = sample_ce_lm_sum.item() / math.log(2)
        else:
            sample_H_LM_bits = 0.0

        sample_inputs_embeds = token_embeddings[j : j + 1]
        sample_compression_tokens = final_compression_tokens_for_ig[j : j + 1]
        sample_model_tokens_with_compression = torch.cat(
            [
                sample_compression_tokens.to(sample_inputs_embeds.device).to(sample_inputs_embeds.dtype),
                sample_inputs_embeds,
            ],
            dim=1,
        )
        sample_compression_attention_mask = compression_attention_mask[j : j + 1]
        sample_attention_mask_with_compression = torch.cat([sample_compression_attention_mask, sample_attention_mask], dim=1)

        sample_outputs_mem = model(
            inputs_embeds=sample_model_tokens_with_compression,
            attention_mask=sample_attention_mask_with_compression,
        )
        sample_logits_mem = sample_outputs_mem.logits
        sample_aligned_logits_mem = sample_logits_mem[:, num_compression_tokens:, :]
        sample_shift_logits_mem = sample_aligned_logits_mem[:, :-1, :].contiguous()
        sample_shift_labels_mem = sample_input_ids[:, 1:].contiguous()
        sample_shift_mask_mem = sample_attention_mask[:, 1:].contiguous()

        sample_shift_logits_mem_flat = sample_shift_logits_mem.view(-1, sample_shift_logits_mem.size(-1))
        sample_shift_labels_mem_flat = sample_shift_labels_mem.view(-1)
        sample_shift_mask_mem_flat = sample_shift_mask_mem.view(-1)

        sample_valid_mask_mem = sample_shift_mask_mem_flat.bool()
        if sample_valid_mask_mem.sum() > 0:
            sample_ce_mem_sum = F.cross_entropy(
                sample_shift_logits_mem_flat[sample_valid_mask_mem],
                sample_shift_labels_mem_flat[sample_valid_mask_mem],
                reduction="sum",
            )
            sample_H_LM_mem_bits = sample_ce_mem_sum.item() / math.log(2)
        else:
            sample_H_LM_mem_bits = 0.0

        per_sample_info_gain.append(sample_H_LM_bits - sample_H_LM_mem_bits)
    return per_sample_info_gain


# ---------------------------------------------------------------------------
# Test fixtures.
# ---------------------------------------------------------------------------


def _tiny_gpt2(seed: int = 0) -> GPT2LMHeadModel:
    """Tiny deterministic GPT2 (no download). 64 vocab, 32 hidden, 2 layers."""
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=64,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _make_inputs(
    model: GPT2LMHeadModel, *, batch_size: int, seq_len: int, num_compression_tokens: int = 1, mask_pattern: str = "all_ones"
):
    """Build a deterministic batch of (input_ids, attention_mask, token_embeddings, compression_token_embeddings, compression_attention_mask)."""
    torch.manual_seed(123)
    vocab = model.config.vocab_size
    hidden = model.config.n_embd

    input_ids = torch.randint(1, vocab, (batch_size, seq_len), dtype=torch.long)
    if mask_pattern == "all_ones":
        attention_mask = torch.ones_like(input_ids)
    elif mask_pattern == "right_padded":
        # Right-pad: first sample full, second sample masks last 2 positions.
        attention_mask = torch.ones_like(input_ids)
        if batch_size >= 2 and seq_len >= 3:
            attention_mask[1, -2:] = 0
    else:
        raise ValueError(mask_pattern)

    token_embeddings = model.get_input_embeddings()(input_ids).to(torch.float32)
    compression_token_embeddings = torch.randn(batch_size, num_compression_tokens, hidden, dtype=torch.float32) * 0.1
    compression_attention_mask = torch.ones(batch_size, num_compression_tokens, dtype=attention_mask.dtype)
    return (
        input_ids,
        attention_mask,
        token_embeddings,
        compression_token_embeddings,
        compression_attention_mask,
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size, seq_len, K, mask_pattern",
    [
        (1, 8, 1, "all_ones"),
        (3, 16, 1, "all_ones"),
        (2, 12, 1, "right_padded"),
        (1, 8, 4, "all_ones"),  # K > 1 to exercise multi-token compression prefix
        (2, 16, 2, "right_padded"),
    ],
)
def test_information_gain_matches_inline_reference(batch_size, seq_len, K, mask_pattern):
    """compute_information_gain must produce numerically identical values to the pre-refactor inline code."""
    model = _tiny_gpt2(seed=0)
    (
        input_ids,
        attention_mask,
        token_embeddings,
        compression_token_embeddings,
        compression_attention_mask,
    ) = _make_inputs(
        model,
        batch_size=batch_size,
        seq_len=seq_len,
        num_compression_tokens=K,
        mask_pattern=mask_pattern,
    )

    ref = _reference_inline_ig(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_embeddings=token_embeddings,
        final_compression_tokens_for_ig=compression_token_embeddings,
        compression_attention_mask=compression_attention_mask,
        num_compression_tokens=K,
    )
    new = compute_information_gain(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_embeddings=token_embeddings,
        compression_token_embeddings=compression_token_embeddings,
        compression_attention_mask=compression_attention_mask,
    )

    assert isinstance(new, list)
    assert len(new) == batch_size
    assert len(ref) == batch_size

    # The two implementations call the model with byte-identical inputs in the
    # same order, so we expect exact equality (no atol/rtol slack).
    for j, (r, n) in enumerate(zip(ref, new)):
        assert isinstance(n, float), f"sample {j}: expected Python float, got {type(n)}"
        assert n == r, f"sample {j}: information gain mismatch — " f"new={n!r}, reference={r!r}"


def test_information_gain_returns_zero_when_mask_excludes_everything():
    """Edge case: if every position past index 0 is masked, IG must be exactly 0."""
    model = _tiny_gpt2(seed=0)
    batch_size, seq_len, K = 2, 4, 1
    input_ids, attention_mask, token_embeddings, compression_token_embeddings, compression_attention_mask = _make_inputs(
        model, batch_size=batch_size, seq_len=seq_len, num_compression_tokens=K
    )
    # Mask out everything except position 0; after the next-token shift,
    # mask[:, 1:] is all-zero and there are no positions to score.
    attention_mask = torch.zeros_like(attention_mask)
    attention_mask[:, 0] = 1

    out = compute_information_gain(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_embeddings=token_embeddings,
        compression_token_embeddings=compression_token_embeddings,
        compression_attention_mask=compression_attention_mask,
    )
    assert out == [0.0, 0.0]


def test_information_gain_is_finite_for_random_compression():
    """Smoke check: random small-norm compression produces finite, real-valued IG."""
    model = _tiny_gpt2(seed=0)
    input_ids, attention_mask, token_embeddings, compression_token_embeddings, compression_attention_mask = _make_inputs(
        model, batch_size=2, seq_len=12, num_compression_tokens=1
    )
    out = compute_information_gain(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_embeddings=token_embeddings,
        compression_token_embeddings=compression_token_embeddings,
        compression_attention_mask=compression_attention_mask,
    )
    assert len(out) == 2
    for v in out:
        assert math.isfinite(v)
