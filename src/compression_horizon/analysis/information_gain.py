from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _sequence_cross_entropy_bits(
    logits: torch.Tensor,  # [batch, sequence, vocabulary]
    labels: torch.Tensor,  # [batch, sequence]
    mask: torch.Tensor,  # [batch, sequence]
    *,
    align_offset: int = 0,
) -> float:
    """Total masked next-token cross-entropy, in bits."""
    aligned_logits = logits[:, align_offset:, :]
    shifted_logits = aligned_logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_mask = mask[:, 1:].contiguous()

    flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    flat_labels = shifted_labels.view(-1)
    flat_mask = shifted_mask.view(-1).bool()

    if flat_mask.sum() == 0:
        return 0.0

    ce_sum = F.cross_entropy(
        flat_logits[flat_mask],
        flat_labels[flat_mask],
        reduction="sum",
    )
    return ce_sum.item() / math.log(2)


@torch.no_grad()
def compute_information_gain(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,  # [batch, sequence]
    attention_mask: torch.Tensor,  # [batch, sequence]
    token_embeddings: torch.Tensor,  # [batch, sequence, hidden]
    compression_token_embeddings: torch.Tensor,  # [batch, compression, hidden]
    compression_attention_mask: torch.Tensor,  # [batch, compression]
) -> list[float]:
    """Per-sample Information Gain in bits (paper Eq. 9)."""
    batch_size = input_ids.size(0)
    num_compression_tokens = compression_token_embeddings.size(1)
    information_gains: list[float] = []

    for j in range(batch_size):
        sample_input_ids = input_ids[j : j + 1]  # [1, sequence]
        sample_attention_mask = attention_mask[j : j + 1]  # [1, sequence]
        sample_token_embeddings = token_embeddings[j : j + 1]  # [1, sequence, hidden]
        sample_compression_token_embeddings = (  # [1, compression, hidden]
            compression_token_embeddings[j : j + 1].to(sample_token_embeddings.device).to(sample_token_embeddings.dtype)
        )
        sample_compression_attention_mask = compression_attention_mask[j : j + 1]  # [1, compression]

        # Original cross entropy
        outputs = model(input_ids=sample_input_ids, attention_mask=sample_attention_mask)
        h_lm_bits = _sequence_cross_entropy_bits(outputs.logits, sample_input_ids, sample_attention_mask, align_offset=0)

        # Compression cross entropy
        united_token_embeddings = torch.cat(
            (sample_compression_token_embeddings, sample_token_embeddings),
            dim=1,
        )  # [1, compression + sequence, hidden]
        united_attention_mask = torch.cat(
            (sample_compression_attention_mask, sample_attention_mask),
            dim=1,
        )  # [1, compression + sequence]

        outputs = model(inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask)
        h_comp_lm_bits = _sequence_cross_entropy_bits(
            outputs.logits,
            sample_input_ids,
            sample_attention_mask,
            align_offset=num_compression_tokens,
        )

        information_gains.append(h_lm_bits - h_comp_lm_bits)

    return information_gains
