"""Unit tests for the information-gain budget-rebalancing loss.

The loss (``budget_rebalance_loss_with_prefix``) consumes logits + a cached per-token base
surprisal directly, so its mechanism is testable on synthetic tensors with no model. We assert the
two defining gradient signs:

* **floor** (margin < epsilon): pushes the true-token logit UP (reduces CE) -> grad is negative.
* **reclaim** (margin > epsilon and delta_bits > water level): pulls the true-token logit DOWN
  (reduces bits-saved, reclaiming budget) -> grad is positive. This is the behaviour +LM lacks.

A separate test covers the cached ``compute_per_token_base_surprisal_nats`` helper (shape + the
no-prefix token-0 invalid mask) with a tiny CPU GPT2.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from compression_horizon.analysis.information_gain import compute_per_token_base_surprisal_nats
from compression_horizon.train.loss import budget_rebalance_loss_with_prefix

_LN2 = math.log(2.0)


def _build_synthetic_logits() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One sample, 3 continuation tokens (true id = 0), num_compression_tokens = 1, no prefix.

    Token 0: margin 5.0 (>> epsilon) with a large bits-saving -> reclaim candidate.
    Token 1: margin 0.2 (<  epsilon)                          -> floor candidate.
    Token 2: margin ~1.0 (== epsilon)                          -> neither.
    """
    # logits length = num_compression_tokens (1) + sequence (3) = 4; pred window = logits[:, 0:3].
    logits = torch.zeros(1, 4, 4)
    logits[0, 0, 0] = 5.0  # token 0 true logit (big margin)
    logits[0, 1, 0] = 0.2  # token 1 true logit (deficient)
    logits[0, 2, 0] = 1.0  # token 2 true logit (~epsilon)
    logits[0, 3, 0] = 0.0  # unused trailing position
    logits = logits.clone().requires_grad_(True)
    input_ids = torch.zeros(1, 3, dtype=torch.long)
    attention_mask = torch.ones(1, 3, dtype=torch.long)
    return logits, input_ids, attention_mask


def test_cap_mode_gradient_signs():
    """Reclaim pushes an over-budget token's true logit down; floor pushes a deficient one up."""
    logits, input_ids, attention_mask = _build_synthetic_logits()
    base_ce_nats = torch.full((1, 3), 3.0)  # generous base surprisal -> positive bits-saved
    base_valid_mask = torch.ones(1, 3, dtype=torch.bool)

    loss, diag = budget_rebalance_loss_with_prefix(
        logits,
        input_ids,
        attention_mask,
        num_compression_tokens=1,
        base_ce_nats=base_ce_nats,
        base_valid_mask=base_valid_mask,
        prefix_len=0,
        mode="cap",
        epsilon=1.0,
        reclaim_weight=1.0,
        water_level_bits=torch.tensor([1.0]),  # cap each token at 1 bit
        budget_bits=torch.tensor([3.0]),
    )
    assert torch.isfinite(loss)
    loss.backward()

    # Token 0 is over-margined AND over-budget -> reclaim raises loss with its true logit -> grad > 0.
    assert logits.grad[0, 0, 0].item() > 0.0
    # Token 1 is deficient -> floor lowers loss by raising its true logit -> grad < 0.
    assert logits.grad[0, 1, 0].item() < 0.0
    # Diagnostics are per-sample [batch] tensors.
    assert diag["total_bits"].shape == (1,)
    assert diag["min_margin"].shape == (1,)


def test_cap_reclaim_disabled_when_below_water_level():
    """A high water level (no token exceeds it) => no reclaim => only the floor term contributes."""
    logits, input_ids, attention_mask = _build_synthetic_logits()
    base_ce_nats = torch.full((1, 3), 3.0)
    base_valid_mask = torch.ones(1, 3, dtype=torch.bool)

    loss, _ = budget_rebalance_loss_with_prefix(
        logits,
        input_ids,
        attention_mask,
        num_compression_tokens=1,
        base_ce_nats=base_ce_nats,
        base_valid_mask=base_valid_mask,
        mode="cap",
        epsilon=1.0,
        reclaim_weight=1.0,
        water_level_bits=torch.tensor([100.0]),  # unreachable cap
        budget_bits=torch.tensor([3.0]),
    )
    loss.backward()
    # With reclaim inert, the over-margined token 0 gets no gradient (floor weight is 0 there too).
    assert abs(logits.grad[0, 0, 0].item()) < 1e-6
    # The deficient token 1 still receives the floor's upward push.
    assert logits.grad[0, 1, 0].item() < 0.0


def test_dual_mode_runs_and_reports_violation():
    """Dual mode returns a finite loss and a budget violation equal to total_bits - budget."""
    logits, input_ids, attention_mask = _build_synthetic_logits()
    base_ce_nats = torch.full((1, 3), 3.0)
    base_valid_mask = torch.ones(1, 3, dtype=torch.bool)
    budget = torch.tensor([1.0])

    loss, diag = budget_rebalance_loss_with_prefix(
        logits,
        input_ids,
        attention_mask,
        num_compression_tokens=1,
        base_ce_nats=base_ce_nats,
        base_valid_mask=base_valid_mask,
        mode="dual",
        epsilon=1.0,
        budget_bits=budget,
        dual_lambda=torch.tensor([0.5]),
        softcount_tau=0.5,
    )
    assert torch.isfinite(loss)
    loss.backward()  # soft-count + budget terms are both differentiable
    assert logits.grad is not None
    # budget_violation == total_bits - budget (both per-sample).
    assert torch.allclose(diag["budget_violation"], diag["total_bits"] - budget, atol=1e-5)
    # soft_count in [0, sequence_len].
    assert 0.0 <= diag["soft_count"].item() <= 3.0


def test_base_surprisal_helper_no_prefix_masks_first_token():
    """Cached H_base: correct shape, token 0 invalid without a prefix, matches manual CE elsewhere."""
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=50, n_positions=32, n_embd=16, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config).eval()

    input_ids = torch.randint(0, 50, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.long)

    base_ce, valid = compute_per_token_base_surprisal_nats(model=model, input_ids=input_ids, attention_mask=attention_mask)
    assert base_ce.shape == (2, 5)
    assert valid.shape == (2, 5)
    # No prefix => token 0 has no predictor and is masked out.
    assert not valid[:, 0].any()
    assert valid[:, 1:].all()

    # Position i (>=1) should equal the plain LM next-token CE from logits[i-1].
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    manual = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(2, 4)
    assert torch.allclose(base_ce[:, 1:], manual, atol=1e-4)
