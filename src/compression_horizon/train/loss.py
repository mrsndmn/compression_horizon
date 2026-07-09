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
    prefix_len: int = 0,
    reduction: str = "mean",
    leading_token_loss_weight: float = 1.0,
    leading_token_loss_count: int = 0,
    loss_margin: float = 0.0,
    ce_temperature: float = 1.0,
    ce_temperature_compensation: str = "none",
) -> torch.Tensor:
    """Next-token cross-entropy when logits include compression (and optional uncompressed prefix) tokens.

    The united sequence fed to the model is ``[mem (num_compression_tokens)] [prefix (prefix_len)]
    [continuation]``. ``input_ids``/``attention_mask`` describe the continuation only (the loss
    target); the ``prefix_len`` real prefix positions are skipped so loss is computed only on the
    tokens that follow the prefix. With ``prefix_len=0`` this is identical to the original.

    ``loss_margin`` (>0) reweights per-token CE by each token's margin deficit
    ``clamp_min(0, loss_margin - (logit[true] - runner_up))`` (weights detached): tokens already
    past the margin contribute ~0 loss, deficient ones are up-weighted, so optimization focuses on
    the hard tokens. Takes precedence over the leading-token weighting. 0.0 = plain CE (unchanged).

    ``ce_temperature`` (T) divides the logits by T before the CE softmax: T>1 softens the predicted
    distribution the loss is measured against, T<1 sharpens it. ``ce_temperature_compensation``
    selects the gradient convention: ``"none"`` (raw, gradient ~1/T) or ``"t2"`` (Hinton, loss
    multiplied by T^2 so gradient magnitude stays ~constant). T=1.0 is byte-identical to plain CE.
    Temperature reshapes only the softmax the CE is taken over; the argmax convergence check
    (computed elsewhere) is invariant to it.
    """
    if num_compression_tokens < 1:
        raise ValueError(f"num_compression_tokens must be >= 1, got {num_compression_tokens}!")
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be >= 0, got {prefix_len}!")

    temperature = float(ce_temperature)
    if temperature <= 0.0:
        raise ValueError(f"ce_temperature must be > 0, got {temperature}!")
    apply_temperature = temperature != 1.0
    compensate_t2 = apply_temperature and str(ce_temperature_compensation).lower() in ("t2", "tsq", "hinton")

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    offset = num_compression_tokens + prefix_len
    pred_logits = logits[:, offset - 1 : -1]  # [batch, sequence, vocabulary] -- next-token logits
    # Temperature scales only the logits entering the CE softmax; at T=1.0 the original tensor flows
    # through unchanged (byte-identical). The argmax convergence check (elsewhere) is unaffected.
    ce_logits = pred_logits / temperature if apply_temperature else pred_logits

    if loss_margin > 0.0:
        per_token_loss = F.cross_entropy(
            ce_logits.flatten(0, 1),
            labels.flatten(),
            reduction="none",
            ignore_index=-100,
        ).view_as(
            labels
        )  # [batch, sequence]
        with torch.no_grad():
            # Margin deficit is measured on the un-temperatured logits: it is a real logit-gap /
            # decode-robustness quantity, not part of the softmax the temperature reshapes.
            true_logit = pred_logits.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # [batch, sequence]
            top2 = pred_logits.topk(2, dim=-1).values  # [batch, sequence, 2]
            is_true_top1 = pred_logits.argmax(dim=-1) == input_ids
            runner_up = torch.where(is_true_top1, top2[..., 1], top2[..., 0])
            # Weight = margin deficit: 0 once the token clears loss_margin, larger the further short.
            weights = (loss_margin - (true_logit - runner_up)).clamp_min(0.0)
        weights = weights.masked_fill(labels == -100, 0.0)
        # Mean over the deficient tokens keeps full-strength gradient on the hard ones; all-satisfied
        # -> numerator 0 -> loss 0 (the convergence signal).
        loss = (per_token_loss * weights).sum() / weights.sum().clamp_min(1e-6)
    elif leading_token_loss_count <= 0 or leading_token_loss_weight == 1.0:
        loss = F.cross_entropy(
            ce_logits.flatten(0, 1),  # [batch * sequence, vocabulary]
            labels.flatten(),  # [batch * sequence]
            reduction=reduction,
        )
    else:
        per_token_loss = F.cross_entropy(
            ce_logits.flatten(0, 1),  # [batch * sequence, vocabulary]
            labels.flatten(),  # [batch * sequence]
            reduction="none",
            ignore_index=-100,
        ).view_as(
            labels
        )  # [batch, sequence]

        weights = torch.ones_like(per_token_loss)
        weights[:, : min(leading_token_loss_count, labels.size(1))] = float(leading_token_loss_weight)
        weights = weights.masked_fill(labels == -100, 0.0)
        weighted_sum = (per_token_loss * weights).sum()
        if reduction == "sum":
            loss = weighted_sum
        else:
            num_valid = attention_mask.sum().to(per_token_loss.dtype).clamp_min(1.0)
            loss = weighted_sum / num_valid
    # Hinton distillation convention: T^2 restores the gradient magnitude that raw logits/T scales
    # down by ~1/T, isolating the distribution-shape effect at fixed learning rate. Raw omits it.
    if compensate_t2:
        loss = loss * (temperature * temperature)
    return loss


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
    prefix_len: int = 0,
) -> torch.Tensor:
    """Compute activation alignment loss between compressed and target hidden states.

    ``prefix_len`` uncompressed prefix positions (after the compression tokens) are skipped so the
    alignment is computed only over the continuation, matching ``target_hidden_states``.
    """
    if loss_type not in {"l2", "l1", "cosine"}:
        raise ValueError(f"Unsupported loss_type: {loss_type}!")
    if num_compression_tokens < 0:
        raise ValueError(f"num_compression_tokens must be >= 0, got {num_compression_tokens}!")
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be >= 0, got {prefix_len}!")

    offset = num_compression_tokens + prefix_len
    total = torch.zeros((), device=target_hidden_states[0].device, dtype=target_hidden_states[0].dtype)
    for i in alignment_layer_indices:
        compression_hidden_states_layer = compression_hidden_states[i][:, offset:]  # [batch, sequence, hidden]
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
    leading_token_loss_weight: float = 1.0,
    leading_token_loss_count: int = 0,
    prefix_len: int = 0,
    loss_margin: float = 0.0,
    ce_temperature: float = 1.0,
    ce_temperature_compensation: str = "none",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute CE loss and optional activation alignment loss (hybrid).

    ``prefix_len`` skips the uncompressed prefix positions so both terms are computed only over the
    continuation (the tokens that follow the prefix). ``loss_margin`` enables per-token margin-aware
    CE reweighting; ``ce_temperature``/``ce_temperature_compensation`` scale the CE softmax (see
    ``next_token_cross_entropy_loss_with_prefix``). Temperature applies to the CE term only, not the
    activation-alignment term.
    """
    ce_loss = next_token_cross_entropy_loss_with_prefix(
        logits,
        input_ids,
        attention_mask,
        num_compression_tokens,
        prefix_len=prefix_len,
        reduction="mean",
        leading_token_loss_weight=leading_token_loss_weight,
        leading_token_loss_count=leading_token_loss_count,
        loss_margin=loss_margin,
        ce_temperature=ce_temperature,
        ce_temperature_compensation=ce_temperature_compensation,
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
        prefix_len=prefix_len,
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
    *,
    prefix_len: int = 0,
    margin: float = 0.0,
) -> torch.Tensor:
    """Per-sample token-level match rate when logits include compression (+ optional prefix) tokens.

    ``prefix_len`` skips the uncompressed prefix positions so the match rate is measured only over
    the continuation (the tokens that follow the prefix).

    ``margin`` (epsilon) requires the true token's logit to lead the runner-up by at least this much
    to count as a match: ``logit[true] - max_{j != true} logit[j] >= margin``. ``margin == 0.0``
    reduces to bare-argmax matching (legacy behaviour, kept bit-identical). A positive margin yields
    a stricter convergence target whose solutions are robust to forward-shape / kernel perturbations.
    """
    if num_compression_tokens < 1:
        raise ValueError(f"num_compression_tokens must be >= 1, got {num_compression_tokens}!")
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be >= 0, got {prefix_len}!")

    offset = num_compression_tokens + prefix_len
    pred_logits = logits[:, offset - 1 : -1]  # [batch, sequence, vocab]
    if margin <= 0.0:
        token_ok = pred_logits.argmax(dim=-1) == input_ids  # [batch, sequence]
    else:
        true_logit = pred_logits.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # [batch, sequence]
        top2 = pred_logits.topk(2, dim=-1).values  # [batch, sequence, 2]
        is_true_top1 = pred_logits.argmax(dim=-1) == input_ids
        # Runner-up = 2nd-highest logit when the true token is the argmax, else the highest
        # (then true_logit - runner_up < 0, so the token correctly fails the margin).
        runner_up = torch.where(is_true_top1, top2[..., 1], top2[..., 0])
        token_ok = (true_logit - runner_up) >= margin
    # Mask out padding positions: otherwise a model that learns to predict the pad
    # token on padded input positions inflates the numerator and the ratio can
    # exceed 1.0. We divide by the count of valid tokens, so the numerator must
    # also be restricted to valid tokens.
    matches = (token_ok & (attention_mask == 1)).sum(dim=-1)
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
