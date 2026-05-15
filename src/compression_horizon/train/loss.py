from __future__ import annotations

import torch
import torch.nn.functional as F


def get_alignment_layer_indices(total_layers: int, num_alignment_layers: int, inverted_alignment: bool) -> range:
    """Return which hidden-state layers to align.

    `total_layers` includes the embedding layer (index 0) and all decoder layers.
    """
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
    num_prefix_tokens: int,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute next-token cross entropy when logits include prefix tokens.

    This project prepends learnable compression tokens to the input embeddings.
    When computing the LM loss against the original (non-prefixed) `input_ids`,
    we align logits by slicing from `num_prefix_tokens - 1` and dropping the last
    time step, mirroring the existing training/eval logic.

    Args:
        logits: Model logits of shape [B, num_prefix_tokens + T, V].
        input_ids: Original token ids (no prefix), shape [B, T].
        attention_mask: Mask for `input_ids`, shape [B, T].
        num_prefix_tokens: Number of prepended prefix tokens.
        reduction: Passed to `torch.nn.functional.cross_entropy`.

    Returns:
        Scalar loss (or unreduced loss if `reduction="none"`).
    """
    if num_prefix_tokens < 1:
        raise ValueError(f"num_prefix_tokens must be >= 1, got {num_prefix_tokens}")

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    aligned_logits = logits[:, num_prefix_tokens - 1 : -1]
    return F.cross_entropy(
        aligned_logits.flatten(0, 1),
        labels.flatten(),
        reduction=reduction,
    )


def next_token_fused_linear_ce_loss_with_prefix(
    lm_head_weight: torch.Tensor,
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_prefix_tokens: int,
) -> torch.Tensor:
    """Fused linear + cross-entropy variant of :func:`next_token_cross_entropy_loss_with_prefix`.

    Avoids materializing the [B, T, V] logits tensor by chunking ``lm_head @ hidden^T``
    fused with cross-entropy inside Liger's kernel (`LigerFusedLinearCrossEntropyLoss`).
    Lower peak memory and fewer launched kernels than the eager hidden→logits→CE chain.

    Prefix alignment matches the eager helper: hidden states at positions
    ``[num_prefix_tokens - 1 : num_prefix_tokens - 1 + T]`` predict ``input_ids[:, :T]``,
    and positions with ``attention_mask == 0`` get label ``-100`` so they are dropped.

    Args:
        lm_head_weight: [V, H] LM head weight (typically ``model.lm_head.weight``).
        last_hidden_state: [B, num_prefix_tokens + T, H] hidden states from the base model.
        input_ids: [B, T] target token ids (no prefix).
        attention_mask: [B, T] mask over ``input_ids``.
        num_prefix_tokens: number of prepended compression tokens.

    Returns:
        Scalar loss (mean over unmasked tokens).
    """
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    if num_prefix_tokens < 1:
        raise ValueError(f"num_prefix_tokens must be >= 1, got {num_prefix_tokens}")

    seq_len = input_ids.shape[1]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    shift_labels = labels.reshape(-1)

    aligned_hidden = last_hidden_state[:, num_prefix_tokens - 1 : num_prefix_tokens - 1 + seq_len, :]
    hidden_2d = aligned_hidden.reshape(-1, aligned_hidden.shape[-1])

    num_items = int((shift_labels != -100).sum().item())
    if num_items == 0:
        return last_hidden_state.new_zeros((), requires_grad=True)

    liger_lce = LigerFusedLinearCrossEntropyLoss(reduction="sum")
    loss_sum = liger_lce(lm_head_weight, hidden_2d, shift_labels)
    return loss_sum / num_items


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
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    return F.cross_entropy(
        shifted_logits.flatten(0, 1),
        shifted_labels.flatten(),
        reduction=reduction,
    )


def activation_alignment_loss_with_prefix(
    *,
    compression_hidden_states: tuple[torch.Tensor, ...],
    target_hidden_states: tuple[torch.Tensor, ...],
    num_prefix_tokens: int,
    alignment_layer_indices: range,
    loss_type: str,
) -> torch.Tensor:
    """Compute activation alignment loss between compressed and target hidden states."""
    loss_type = (loss_type or "").lower()
    if loss_type not in {"l2", "l1", "cosine"}:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    if num_prefix_tokens < 0:
        raise ValueError(f"num_prefix_tokens must be >= 0, got {num_prefix_tokens}")

    total = torch.zeros((), device=target_hidden_states[0].device, dtype=target_hidden_states[0].dtype)
    for i in alignment_layer_indices:
        compression_h = compression_hidden_states[i][:, num_prefix_tokens:]  # [B, T, H]
        target_h = target_hidden_states[i]  # [B, T, H]
        if loss_type == "l2":
            layer_loss = (
                F.mse_loss(
                    compression_h,
                    target_h,
                    reduction="none",
                )
                .sum(dim=-1)
                .sqrt()
                .mean()
            )
        elif loss_type == "l1":
            layer_loss = (
                F.l1_loss(
                    compression_h,
                    target_h,
                    reduction="none",
                )
                .sum(dim=-1)
                .mean()
            )
        else:
            cosine = F.cosine_similarity(compression_h, target_h, dim=-1)
            layer_loss = (1.0 - cosine).mean()
        total = total + layer_loss
    return total


def compute_hybrid_cross_entropy_and_alignment_loss(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_prefix_tokens: int,
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
        num_prefix_tokens,
        reduction="mean",
    )

    lt = (loss_type or "").lower()
    if hybrid_alpha is None or lt == "cross_entropy":
        return ce_loss, None

    if target_hidden_states is None or compression_hidden_states is None:
        raise ValueError("target_hidden_states and compression_hidden_states are required when hybrid_alpha is set")

    alignment_layer_indices = get_alignment_layer_indices(
        total_layers=len(target_hidden_states),
        num_alignment_layers=num_alignment_layers,
        inverted_alignment=inverted_alignment,
    )
    align_loss = activation_alignment_loss_with_prefix(
        compression_hidden_states=compression_hidden_states,
        target_hidden_states=target_hidden_states,
        num_prefix_tokens=num_prefix_tokens,
        alignment_layer_indices=alignment_layer_indices,
        loss_type=lt,
    )
    return ce_loss + float(hybrid_alpha) * align_loss, align_loss


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

    lt = (loss_type or "").lower()
    if hybrid_alpha is None or lt == "cross_entropy":
        return ce_loss, None

    if target_hidden_states is None or compression_hidden_states is None:
        raise ValueError("target_hidden_states and compression_hidden_states are required when hybrid_alpha is set")

    alignment_layer_indices = get_alignment_layer_indices(
        total_layers=len(target_hidden_states),
        num_alignment_layers=num_alignment_layers,
        inverted_alignment=inverted_alignment,
    )
    align_loss = activation_alignment_loss_with_prefix(
        compression_hidden_states=compression_hidden_states,
        target_hidden_states=target_hidden_states,
        num_prefix_tokens=0,
        alignment_layer_indices=alignment_layer_indices,
        loss_type=lt,
    )
    return ce_loss + float(hybrid_alpha) * align_loss, align_loss


@torch.no_grad()
def token_argmax_match_rate_with_prefix(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_prefix_tokens: int,
) -> torch.Tensor:
    """Compute per-sample token-level argmax match rate with prefixed logits.

    Returns:
        Tensor of shape [B] with match rate in [0, 1] (undefined where mask sums to 0).
    """
    if num_prefix_tokens < 1:
        raise ValueError(f"num_prefix_tokens must be >= 1, got {num_prefix_tokens}")

    preds = logits[:, num_prefix_tokens - 1 : -1].argmax(dim=-1)
    matches = (preds == input_ids).sum(dim=-1)
    denom = attention_mask.sum(dim=-1)
    return matches / denom


@torch.no_grad()
def token_argmax_match_rate(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample token-level argmax match rate (no prefix tokens in logits).

    Uses next-token alignment: logits[t] predicts input_ids[t+1].
    """
    preds = logits[:, :-1].argmax(dim=-1)
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:]
    matches = ((preds == labels) & (mask == 1)).sum(dim=-1)
    denom = mask.sum(dim=-1).clamp_min(1)
    return matches / denom
