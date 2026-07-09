"""Tests for the cross-entropy temperature knob in progressive cramming.

Covers ``ce_temperature`` / ``ce_temperature_compensation`` in
``next_token_cross_entropy_loss_with_prefix``:

  * T=1.0 (raw and t2) is byte-identical to the un-temperatured call,
  * raw temperature = plain CE on logits/T (value parity),
  * t2 compensation multiplies the loss by T^2,
  * the t2 gradient is exactly T^2 x the raw gradient (raw scales the CE gradient
    by ~1/T at fixed learning rate; t2 restores its magnitude),
  * the argmax convergence check is temperature-invariant (scaling logits by 1/T
    does not move the convergence bar),
  * invalid (non-positive) temperature raises.

All CPU-only, pure-function tests (no model download).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

from compression_horizon.train.loss import (
    next_token_cross_entropy_loss_with_prefix,
    token_argmax_match_rate_with_prefix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _fixture(seed: int = 0, *, batch: int = 2, C: int = 1, L: int = 6, vocab: int = 11, dtype=torch.float64):
    torch.manual_seed(seed)
    logits = torch.randn(batch, C + L, vocab, dtype=dtype)
    ids = torch.randint(0, vocab, (batch, L))
    mask = torch.ones(batch, L, dtype=torch.long)
    return logits, ids, mask, C, vocab


# --------------------------------------------------------------------------- #
# T=1.0 identity (byte-identical to the un-temperatured path).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("compensation", ["none", "t2"])
def test_temperature_one_is_byte_identical(compensation):
    """T=1.0 (either compensation) must equal the call without the temperature kwargs."""
    logits, ids, mask, C, _ = _fixture(1)
    base = next_token_cross_entropy_loss_with_prefix(logits, ids, mask, C)
    got = next_token_cross_entropy_loss_with_prefix(
        logits, ids, mask, C, ce_temperature=1.0, ce_temperature_compensation=compensation
    )
    assert got == base  # exact equality, not allclose


def test_temperature_one_byte_identical_loss_margin_path():
    """The loss_margin (>0) path must also be unchanged at T=1.0."""
    logits, ids, mask, C, _ = _fixture(2)
    base = next_token_cross_entropy_loss_with_prefix(logits, ids, mask, C, loss_margin=1.5)
    got = next_token_cross_entropy_loss_with_prefix(
        logits, ids, mask, C, loss_margin=1.5, ce_temperature=1.0, ce_temperature_compensation="t2"
    )
    assert got == base


# --------------------------------------------------------------------------- #
# Value parity: raw = CE(logits/T); t2 = T^2 * CE(logits/T).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", [0.1, 0.5, 1.5, 2.0])
def test_raw_temperature_equals_manual_scaled_ce(T):
    logits, ids, mask, C, vocab = _fixture(3)
    got = next_token_cross_entropy_loss_with_prefix(logits, ids, mask, C, ce_temperature=T, ce_temperature_compensation="none")
    expected = F.cross_entropy((logits[:, C - 1 : -1] / T).reshape(-1, vocab), ids.reshape(-1))
    assert torch.allclose(got, expected, atol=1e-10)


@pytest.mark.parametrize("T", [0.1, 0.5, 1.5, 2.0])
def test_t2_compensation_multiplies_by_T_squared(T):
    logits, ids, mask, C, _ = _fixture(3)
    raw = next_token_cross_entropy_loss_with_prefix(logits, ids, mask, C, ce_temperature=T, ce_temperature_compensation="none")
    t2 = next_token_cross_entropy_loss_with_prefix(logits, ids, mask, C, ce_temperature=T, ce_temperature_compensation="t2")
    assert torch.allclose(t2, (T * T) * raw, atol=1e-10)
    # Compensation must actually change the loss away from 1.0 temperature.
    assert not torch.allclose(t2, raw)


# --------------------------------------------------------------------------- #
# Gradient relationship: grad(t2) == T^2 * grad(raw).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", [0.25, 0.5, 1.5, 3.0])
def test_t2_gradient_is_T_squared_times_raw_gradient(T):
    """Raw logits/T scales the CE gradient by ~1/T; the t2 form restores it (x T^2 exactly)."""
    logits, ids, mask, C, _ = _fixture(4)

    lr = logits.clone().requires_grad_(True)
    next_token_cross_entropy_loss_with_prefix(lr, ids, mask, C, ce_temperature=T, ce_temperature_compensation="none").backward()

    lt = logits.clone().requires_grad_(True)
    next_token_cross_entropy_loss_with_prefix(lt, ids, mask, C, ce_temperature=T, ce_temperature_compensation="t2").backward()

    assert torch.allclose(lt.grad, (T * T) * lr.grad, atol=1e-10)


# --------------------------------------------------------------------------- #
# Convergence is temperature-invariant (argmax bar cannot be moved by scaling).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", [0.1, 0.5, 1.5, 2.0])
def test_argmax_convergence_is_temperature_invariant(T):
    logits, ids, mask, C, _ = _fixture(5)
    base = token_argmax_match_rate_with_prefix(logits, ids, mask, C)
    scaled = token_argmax_match_rate_with_prefix(logits / T, ids, mask, C)
    assert torch.equal(base, scaled)


# --------------------------------------------------------------------------- #
# Guardrail.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_T", [0.0, -1.0])
def test_non_positive_temperature_raises(bad_T):
    logits, ids, mask, C, _ = _fixture(6)
    with pytest.raises(ValueError):
        next_token_cross_entropy_loss_with_prefix(logits, ids, mask, C, ce_temperature=bad_T)
