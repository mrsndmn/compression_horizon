from __future__ import annotations

import torch
import torch.nn.functional as F


def get_alignment_layer_indices(total_layers: int, num_alignment_layers: int, inverted_alignment: bool) -> range:
    """Indices of hidden-state layers to align (total_layers includes the embedding layer)."""
    if num_alignment_layers > 0:
        num_layers = max(0, min(num_alignment_layers, total_layers))
        if inverted_alignment:
            return range(total_layers - num_layers, total_layers)
        return range(num_layers)
    return range(total_layers)


def next_token_cross_entropy_loss_with_prefix(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_compression_tokens: int,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Next-token cross-entropy when logits include compression-prefix tokens."""
    if num_compression_tokens < 1:
        raise ValueError(f"num_compression_tokens must be >= 1, got {num_compression_tokens}!")

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return F.cross_entropy(
        logits[:, num_compression_tokens - 1 : -1].flatten(0, 1),  # [batch * sequence, vocabulary]
        labels.flatten(),  # [batch * sequence]
        reduction=reduction,
    )


def next_token_cross_entropy_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard next-token CE loss (no prefix tokens in logits)."""
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return F.cross_entropy(
        logits[:, :-1].flatten(0, 1),  # [batch * sequence, vocabulary]
        labels[:, 1:].flatten(),  # [batch * sequence]
        reduction=reduction,
    )


def activation_alignment_loss_with_prefix(
    *,
    compression_hidden_states: tuple[torch.Tensor, ...],
    target_hidden_states: tuple[torch.Tensor, ...],
    num_compression_tokens: int,
    alignment_layer_indices: range,
    loss_type: str,
) -> torch.Tensor:
    """Compute activation alignment loss between compressed and target hidden states."""
    if loss_type not in {"l2", "l1", "cosine"}:
        raise ValueError(f"Unsupported loss_type: {loss_type}!")
    if num_compression_tokens < 0:
        raise ValueError(f"num_compression_tokens must be >= 0, got {num_compression_tokens}!")

    total = torch.zeros((), device=target_hidden_states[0].device, dtype=target_hidden_states[0].dtype)
    for i in alignment_layer_indices:
        compression_hidden_states_layer = compression_hidden_states[i][:, num_compression_tokens:]  # [batch, sequence, hidden]
        target_hidden_states_layer = target_hidden_states[i]  # [batch, sequence, hidden]
        if loss_type == "l2":
            layer_loss = (
                F.mse_loss(
                    compression_hidden_states_layer,
                    target_hidden_states_layer,
                    reduction="none",
                )
                .sum(dim=-1)
                .sqrt()
                .mean()
            )
        elif loss_type == "l1":
            layer_loss = (
                F.l1_loss(
                    compression_hidden_states_layer,
                    target_hidden_states_layer,
                    reduction="none",
                )
                .sum(dim=-1)
                .mean()
            )
        else:
            cosine = F.cosine_similarity(compression_hidden_states_layer, target_hidden_states_layer, dim=-1)
            layer_loss = (1.0 - cosine).mean()
        total = total + layer_loss
    return total


def compute_hybrid_cross_entropy_and_alignment_loss(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_compression_tokens: int,
    target_hidden_states: tuple[torch.Tensor, ...] | None = None,
    compression_hidden_states: tuple[torch.Tensor, ...] | None = None,
    num_alignment_layers: int,
    inverted_alignment: bool,
    loss_type: str,
    hybrid_alpha: float | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute CE loss and optional activation alignment loss (hybrid)."""
    ce_loss = next_token_cross_entropy_loss_with_prefix(
        logits,
        input_ids,
        attention_mask,
        num_compression_tokens,
        reduction="mean",
    )

    loss_type = (loss_type or "").lower()
    if hybrid_alpha is None or loss_type == "cross_entropy":
        return ce_loss, None

    if target_hidden_states is None or compression_hidden_states is None:
        raise ValueError("Hidden states are required when hybrid_alpha is set!")
    alignment_layer_indices = get_alignment_layer_indices(
        total_layers=len(target_hidden_states),
        num_alignment_layers=num_alignment_layers,
        inverted_alignment=inverted_alignment,
    )
    alignment_loss = activation_alignment_loss_with_prefix(
        compression_hidden_states=compression_hidden_states,
        target_hidden_states=target_hidden_states,
        num_compression_tokens=num_compression_tokens,
        alignment_layer_indices=alignment_layer_indices,
        loss_type=loss_type,
    )
    return ce_loss + float(hybrid_alpha) * alignment_loss, alignment_loss


def compute_hybrid_cross_entropy_and_alignment_loss_no_prefix(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_hidden_states: tuple[torch.Tensor, ...] | None = None,
    compression_hidden_states: tuple[torch.Tensor, ...] | None = None,
    num_alignment_layers: int,
    inverted_alignment: bool,
    loss_type: str,
    hybrid_alpha: float | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute CE loss and optional activation alignment loss (no prefix tokens in logits)."""
    ce_loss = next_token_cross_entropy_loss(
        logits,
        input_ids,
        attention_mask,
        reduction="mean",
    )

    loss_type = (loss_type or "").lower()
    if hybrid_alpha is None or loss_type == "cross_entropy":
        return ce_loss, None

    if target_hidden_states is None or compression_hidden_states is None:
        raise ValueError("Hidden states are required when hybrid_alpha is set!")
    alignment_layer_indices = get_alignment_layer_indices(
        total_layers=len(target_hidden_states),
        num_alignment_layers=num_alignment_layers,
        inverted_alignment=inverted_alignment,
    )
    alignment_loss = activation_alignment_loss_with_prefix(
        compression_hidden_states=compression_hidden_states,
        target_hidden_states=target_hidden_states,
        num_compression_tokens=0,
        alignment_layer_indices=alignment_layer_indices,
        loss_type=loss_type,
    )
    return ce_loss + float(hybrid_alpha) * alignment_loss, alignment_loss


@torch.no_grad()
def token_argmax_match_rate_with_prefix(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_compression_tokens: int,
) -> torch.Tensor:
    """Per-sample token-level argmax match rate when logits include compression-prefix tokens."""
    if num_compression_tokens < 1:
        raise ValueError(f"num_compression_tokens must be >= 1, got {num_compression_tokens}!")

    prediction_ids = logits[:, num_compression_tokens - 1 : -1].argmax(dim=-1)  # [batch, sequence]
    # Mask out padding positions: otherwise a model that learns to predict the pad
    # token on padded input positions inflates the numerator and the ratio can
    # exceed 1.0. We divide by the count of valid tokens, so the numerator must
    # also be restricted to valid tokens.
    matches = ((prediction_ids == input_ids) & (attention_mask == 1)).sum(dim=-1)
    ratio = matches / attention_mask.sum(dim=-1).clamp_min(1)
    return ratio


@torch.no_grad()
def token_argmax_match_rate(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample token-level argmax match rate (no prefix tokens in logits)."""
    prediction_ids = logits[:, :-1].argmax(dim=-1)  # [batch, sequence]
    matches = ((prediction_ids == input_ids[:, 1:]) & (attention_mask[:, 1:] == 1)).sum(dim=-1)
    ratio = matches / attention_mask[:, 1:].sum(dim=-1).clamp_min(1)
    return ratio
