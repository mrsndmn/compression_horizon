from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_LN2 = math.log(2.0)


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
    """
    if num_compression_tokens < 1:
        raise ValueError(f"num_compression_tokens must be >= 1, got {num_compression_tokens}!")
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be >= 0, got {prefix_len}!")

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    offset = num_compression_tokens + prefix_len
    if loss_margin > 0.0:
        pred_logits = logits[:, offset - 1 : -1]  # [batch, sequence, vocabulary]
        per_token_loss = F.cross_entropy(
            pred_logits.flatten(0, 1),
            labels.flatten(),
            reduction="none",
            ignore_index=-100,
        ).view_as(
            labels
        )  # [batch, sequence]
        with torch.no_grad():
            true_logit = pred_logits.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # [batch, sequence]
            top2 = pred_logits.topk(2, dim=-1).values  # [batch, sequence, 2]
            is_true_top1 = pred_logits.argmax(dim=-1) == input_ids
            runner_up = torch.where(is_true_top1, top2[..., 1], top2[..., 0])
            # Weight = margin deficit: 0 once the token clears loss_margin, larger the further short.
            weights = (loss_margin - (true_logit - runner_up)).clamp_min(0.0)
        weights = weights.masked_fill(labels == -100, 0.0)
        # Mean over the deficient tokens keeps full-strength gradient on the hard ones; all-satisfied
        # -> numerator 0 -> loss 0 (the convergence signal).
        return (per_token_loss * weights).sum() / weights.sum().clamp_min(1e-6)
    if leading_token_loss_count <= 0 or leading_token_loss_weight == 1.0:
        loss = F.cross_entropy(
            logits[:, offset - 1 : -1].flatten(0, 1),  # [batch * sequence, vocabulary]
            labels.flatten(),  # [batch * sequence]
            reduction=reduction,
        )
    else:
        per_token_loss = F.cross_entropy(
            logits[:, offset - 1 : -1].flatten(0, 1),  # [batch * sequence, vocabulary]
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute CE loss and optional activation alignment loss (hybrid).

    ``prefix_len`` skips the uncompressed prefix positions so both terms are computed only over the
    continuation (the tokens that follow the prefix). ``loss_margin`` enables per-token margin-aware
    CE reweighting (see ``next_token_cross_entropy_loss_with_prefix``).
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


def _per_token_margin(pred_logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token logit margin ``logit[true] - max_{j != true} logit[j]`` (>= 0 iff argmax is true).

    ``pred_logits`` is the prediction window ``[batch, sequence, vocab]`` already aligned to
    ``input_ids`` (position i predicts ``input_ids[:, i]``).
    """
    true_logit = pred_logits.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # [batch, sequence]
    top2 = pred_logits.topk(2, dim=-1).values  # [batch, sequence, 2]
    is_true_top1 = pred_logits.argmax(dim=-1) == input_ids
    runner_up = torch.where(is_true_top1, top2[..., 1], top2[..., 0])
    return true_logit - runner_up


def budget_rebalance_loss_with_prefix(
    logits: torch.Tensor,  # [batch, compression + prefix + sequence, vocab]
    input_ids: torch.Tensor,  # [batch, sequence] (continuation, loss target)
    attention_mask: torch.Tensor,  # [batch, sequence]
    num_compression_tokens: int,
    *,
    base_ce_nats: torch.Tensor,  # [batch, sequence] cached H_base (nats), stage-constant
    base_valid_mask: torch.Tensor,  # [batch, sequence] bool: positions with defined base surprisal
    prefix_len: int = 0,
    mode: str = "cap",
    epsilon: float = 1.0,
    reclaim_weight: float = 1.0,
    water_level_bits: torch.Tensor | None = None,  # [batch] per-sample cap c = B / L (cap mode)
    budget_bits: torch.Tensor | None = None,  # [batch] per-sample IG budget B (dual mode, + logging)
    dual_lambda: torch.Tensor | None = None,  # [batch] per-sample Lagrange multiplier >= 0 (dual mode)
    softcount_tau: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Information-gain budget-rebalancing loss (paper Appendix, "budget rebalancing").

    Redistributes the (roughly constant) per-sample information-gain budget so that MORE tokens
    clear the fixed convergence margin ``epsilon``. Works on true per-token bits-saved
    ``delta_bits_i = (H_base_i - H_comp_i) / ln2`` where ``H_comp_i`` is the per-token CE under the
    memory embedding (differentiable) and ``H_base_i`` is the frozen-LM surprisal without it
    (``base_ce_nats``, cached per stage since it does not depend on the embedding).

    ``mode='cap'`` (C): a margin-deficit CE *floor* (identical mechanism to ``loss_margin``/+LM)
    pushes tokens with ``margin < epsilon`` up, PLUS a *reclaim* term that pulls tokens with
    ``margin > epsilon`` whose bits-saved exceed the adaptive water level ``water_level_bits`` back
    down (``relu(delta_bits - c)``). The reclaim mask (``margin > epsilon``, detached) guarantees a
    deficient token is never pushed further below the floor.

    The floor (and the dual soft count) depend only on the model's own logits, so they run over
    every real (non-padding) token; the reclaim term and the bits budget consume ``delta_bits`` and
    so are restricted to tokens with a defined base surprisal (``base_valid_mask``). This split
    matters at a single-token / no-prefix stage, whose base mask is entirely False: the floor still
    drives reconstruction there (base_valid_mask empty only zeroes reclaim), so progressive cramming
    can leave stage 0.

    ``mode='dual'`` (D): maximize a soft count ``sigmoid((margin - epsilon)/tau)`` of tokens past
    epsilon subject to a total-bits budget ``sum_i delta_bits_i <= budget_bits`` via the per-sample
    Lagrangian ``-soft_count + lambda * (bits - B)`` (``lambda`` detached; the caller runs dual
    ascent on the returned ``budget_violation``). KKT optimum is water-filling.

    Returns ``(loss, diagnostics)``; ``diagnostics`` holds detached per-sample ``[batch]`` tensors
    (``budget_violation``, ``total_bits``, ``min_margin``, ``margin_var``, ``delta_bits_max``,
    ``delta_bits_cov``, ``soft_count``) for dual updates and the fungibility diagnostic.
    """
    if mode not in ("cap", "dual"):
        raise ValueError(f"budget_rebalance mode must be 'cap' or 'dual', got {mode!r}!")
    if num_compression_tokens < 1:
        raise ValueError(f"num_compression_tokens must be >= 1, got {num_compression_tokens}!")

    batch_size, seq_len = input_ids.shape
    offset = num_compression_tokens + prefix_len
    pred_logits = logits[:, offset - 1 : -1]  # [batch, sequence, vocab]
    vocab = pred_logits.size(-1)

    # Two masks. The reconstruction terms (the margin-deficit CE floor and the soft count) depend
    # only on the model's OWN logits, so they run over every real (non-padding) token -- including
    # positions where the base LM has no defined surprisal (e.g. token 0 of a no-prefix stage). Only
    # the reclaim term and the bits budget consume ``delta_bits``, so those use the stricter
    # base-surprisal mask. Tying the floor to ``base_valid_mask`` was a bug: a single-token /
    # no-prefix stage has an all-False base mask, which zeroed the entire loss (no gradient) and
    # stalled progressive cramming at stage 0 forever.
    pad_valid = attention_mask == 1  # [batch, sequence] real (non-padding) tokens
    bits_valid = pad_valid & base_valid_mask.bool()  # tokens with defined base surprisal (delta_bits)
    pad_valid_f = pad_valid.to(pred_logits.dtype)
    bits_valid_f = bits_valid.to(pred_logits.dtype)

    comp_ce = F.cross_entropy(
        pred_logits.reshape(-1, vocab),
        input_ids.reshape(-1),
        reduction="none",
    ).view(
        batch_size, seq_len
    )  # per-token H_comp (nats), differentiable
    margin = _per_token_margin(pred_logits, input_ids)  # [batch, sequence]
    delta_bits = (base_ce_nats - comp_ce) / _LN2  # [batch, sequence]

    with torch.no_grad():
        pad_count = pad_valid_f.sum(dim=1).clamp_min(1.0)  # [batch] real tokens
        bits_count = bits_valid_f.sum(dim=1).clamp_min(1.0)  # [batch] base-surprisal-defined tokens
        m_mean = (margin * pad_valid_f).sum(dim=1) / pad_count
        margin_var = (((margin - m_mean.unsqueeze(1)) ** 2) * pad_valid_f).sum(dim=1) / pad_count
        min_margin = margin.masked_fill(~pad_valid, float("inf")).amin(dim=1)
        total_bits = (delta_bits * bits_valid_f).sum(dim=1)  # [batch]
        pos_delta = delta_bits.clamp_min(0.0) * bits_valid_f
        delta_max = pos_delta.amax(dim=1)  # [batch]
        d_mean = pos_delta.sum(dim=1) / bits_count
        d_std = (((pos_delta - d_mean.unsqueeze(1)) ** 2) * bits_valid_f).sum(dim=1).div(bits_count).sqrt()
        delta_cov = d_std / d_mean.clamp_min(1e-6)  # coefficient of variation (inequality proxy)

    if mode == "cap":
        if water_level_bits is None:
            raise ValueError("water_level_bits is required for budget_rebalance mode='cap'!")
        with torch.no_grad():
            deficit_w = (epsilon - margin).clamp_min(0.0).masked_fill(~pad_valid, 0.0)  # +LM floor weights
            reclaim_mask = (bits_valid & (margin > epsilon)).to(pred_logits.dtype)
        floor = (comp_ce * deficit_w).sum() / deficit_w.sum().clamp_min(1e-6)
        c = water_level_bits.to(delta_bits.dtype).view(batch_size, 1)  # [batch, 1]
        excess = (delta_bits - c).clamp_min(0.0) * reclaim_mask  # differentiable through comp_ce
        reclaim = excess.sum() / reclaim_mask.sum().clamp_min(1.0)
        loss = floor + reclaim_weight * reclaim
        soft_count = (margin > epsilon).to(pred_logits.dtype).mul(pad_valid_f).sum(dim=1)
        budget_violation = (total_bits - budget_bits) if budget_bits is not None else total_bits
    else:  # dual
        if budget_bits is None or dual_lambda is None:
            raise ValueError("budget_bits and dual_lambda are required for budget_rebalance mode='dual'!")
        soft = torch.sigmoid((margin - epsilon) / softcount_tau) * pad_valid_f  # [batch, sequence]
        soft_count = soft.sum(dim=1)  # [batch], differentiable
        bits_per_sample = (delta_bits * bits_valid_f).sum(dim=1)  # [batch], differentiable
        violation = bits_per_sample - budget_bits.to(bits_per_sample.dtype)  # [batch]
        lam = dual_lambda.to(bits_per_sample.dtype).clamp_min(0.0)  # detached (input is a buffer)
        loss = (-soft_count + lam * violation).mean()
        budget_violation = violation.detach()

    diagnostics = {
        "budget_violation": budget_violation.detach(),  # [batch]
        "total_bits": total_bits,  # [batch]
        "min_margin": min_margin,  # [batch]
        "margin_var": margin_var,  # [batch]
        "delta_bits_max": delta_max,  # [batch]
        "delta_bits_cov": delta_cov,  # [batch]
        "soft_count": soft_count.detach(),  # [batch]
    }
    return loss, diagnostics
