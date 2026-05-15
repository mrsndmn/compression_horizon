"""CPU-only unit tests for the bucketed progressive curriculum.

These tests exercise the load-bearing pieces of the new bucketed path
without spinning up a real model:

* The ``_build_step_attention_mask`` helper (AC 4, AC 7 — drives the
  cumulative cross-entropy mask).
* Sticky OR semantics of the per-position convergence array (AC 5).
* Cumulative cross-entropy numerical correctness — feeding the mask the
  helper produces through ``next_token_cross_entropy_loss_with_prefix``
  yields ~0 loss when positions ``0..f`` are predicted perfectly.
* The CE-only guard at ``_train_progressive_bucketed_for_batch`` entry —
  hybrid alignment under bucketed path raises ``NotImplementedError``.

Run with::

    PYTHONPATH=./src pytest tests/test_progressive_bucketed.py -q
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

# Make `src/` importable when running pytest from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compression_horizon.train.loss import next_token_cross_entropy_loss_with_prefix
from compression_horizon.train.progressive_cramming_trainer import (
    ProgressiveCrammingTrainer,
    _build_step_attention_mask,
)

# ---------------------------------------------------------------------------
# _build_step_attention_mask
# ---------------------------------------------------------------------------


def test_step_attention_mask_frontier_zero_ac4():
    """AC 4 — at frontier=0, bucket=64, exactly 1 unmasked position per sample.

    All samples are fully-valid (no padding), so the mask must be exactly
    ``[1, 0, 0, ..., 0]`` for every row.
    """
    batch_size = 2
    bucket_size = 64
    valid_mask_full = torch.ones(batch_size, bucket_size, dtype=torch.long)
    mask = _build_step_attention_mask(
        frontier=0,
        bucket_size=bucket_size,
        valid_mask_full=valid_mask_full,
        dtype=torch.long,
    )
    assert mask.shape == (batch_size, bucket_size)
    assert mask.dtype == torch.long
    # exactly one active position per sample
    assert int(mask.sum(dim=1).max().item()) == 1
    assert int(mask.sum(dim=1).min().item()) == 1
    # position 0 active; rest masked
    assert torch.equal(mask[:, 0], torch.ones(batch_size, dtype=torch.long))
    assert int(mask[:, 1:].sum().item()) == 0
    # output must be contiguous (stride-guard defense for Dynamo)
    assert mask.is_contiguous()


def test_step_attention_mask_respects_valid_mask():
    """A sample with valid_len < frontier+1 contributes zero past its real length."""
    bucket_size = 8
    # Sample 0 fully valid; sample 1 only 3 real tokens (positions 0,1,2).
    valid_mask_full = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    mask = _build_step_attention_mask(
        frontier=4,
        bucket_size=bucket_size,
        valid_mask_full=valid_mask_full,
        dtype=torch.long,
    )
    # Sample 0: positions 0..4 active = 5
    assert int(mask[0].sum().item()) == 5
    # Sample 1: positions 0..2 active = 3 (capped by valid mask)
    assert int(mask[1].sum().item()) == 3
    assert torch.equal(mask[1, 3:], torch.zeros(5, dtype=torch.long))


def test_step_attention_mask_frontier_advance_grows_active_set():
    """As frontier advances 0 -> 1 -> 2, exactly one more position activates."""
    batch_size = 1
    bucket_size = 8
    valid_mask_full = torch.ones(batch_size, bucket_size, dtype=torch.long)
    counts = [
        int(
            _build_step_attention_mask(
                frontier=f,
                bucket_size=bucket_size,
                valid_mask_full=valid_mask_full,
                dtype=torch.long,
            )
            .sum()
            .item()
        )
        for f in range(bucket_size)
    ]
    assert counts == list(range(1, bucket_size + 1))


# ---------------------------------------------------------------------------
# Sticky-OR per-position convergence
# ---------------------------------------------------------------------------


def test_per_position_converged_sticky_or_ac5():
    """AC 5 — once converged, a position stays converged.

    Feed step_match=[1,0,...] then [0,0,...]; assert position 0 remains True.
    """
    batch_size = 1
    bucket_size = 4
    per_position_converged = torch.zeros(batch_size, bucket_size, dtype=torch.bool)
    step_match_1 = torch.tensor([[True, False, False, False]])
    step_match_2 = torch.tensor([[False, False, False, False]])

    per_position_converged = per_position_converged | step_match_1
    per_position_converged = per_position_converged | step_match_2

    assert bool(per_position_converged[0, 0].item()) is True
    assert int(per_position_converged.sum().item()) == 1


# ---------------------------------------------------------------------------
# Cumulative CE numerical correctness on a tiny bucket
# ---------------------------------------------------------------------------


def test_cumulative_ce_perfect_prediction_yields_near_zero():
    """Positions 0..f predicted perfectly -> CE should be near zero.

    Construct logits where for each unmasked position ``i`` the argmax
    matches ``input_ids[:, i]`` with very high confidence; masked positions
    have garbage logits. ``next_token_cross_entropy_loss_with_prefix`` must
    ignore them (labels=-100 there) and return a near-zero loss.
    """
    batch_size = 1
    bucket_size = 4
    vocab_size = 8
    num_compression_tokens = 1
    frontier = 2  # positions 0,1,2 active

    input_ids = torch.tensor([[3, 5, 2, 7]], dtype=torch.long)

    # logits shape: [B, num_compression_tokens + bucket_size, V]
    logits = torch.full(
        (batch_size, num_compression_tokens + bucket_size, vocab_size),
        -10.0,
    )
    # The CE helper aligns: aligned_logits = logits[:, num_prefix-1 : -1]
    # so logits[:, num_compression_tokens - 1 + i] predicts input_ids[:, i].
    for i in range(bucket_size):
        logits[0, num_compression_tokens - 1 + i, int(input_ids[0, i].item())] = 50.0

    valid_mask_full = torch.ones(batch_size, bucket_size, dtype=torch.long)
    step_attention_mask = _build_step_attention_mask(
        frontier=frontier,
        bucket_size=bucket_size,
        valid_mask_full=valid_mask_full,
        dtype=torch.long,
    )

    loss = next_token_cross_entropy_loss_with_prefix(
        logits,
        input_ids,
        step_attention_mask,
        num_prefix_tokens=num_compression_tokens,
    )
    assert loss.item() < 1e-3, f"expected near-zero loss, got {loss.item()}"


def test_cumulative_ce_only_counts_active_positions():
    """If positions past ``f`` predict garbage, CE must not blow up.

    Position 0 is perfect, positions 1..3 predict the *wrong* token with
    high confidence — but they are masked out (frontier=0), so CE must
    only reflect position 0 (near-zero).
    """
    batch_size = 1
    bucket_size = 4
    vocab_size = 8
    num_compression_tokens = 1

    input_ids = torch.tensor([[3, 5, 2, 7]], dtype=torch.long)
    logits = torch.full(
        (batch_size, num_compression_tokens + bucket_size, vocab_size),
        -10.0,
    )
    # Position 0 perfect
    logits[0, num_compression_tokens - 1 + 0, int(input_ids[0, 0].item())] = 50.0
    # Positions 1..3 confidently *wrong*
    for i in range(1, bucket_size):
        wrong = (int(input_ids[0, i].item()) + 1) % vocab_size
        logits[0, num_compression_tokens - 1 + i, wrong] = 50.0

    valid_mask_full = torch.ones(batch_size, bucket_size, dtype=torch.long)
    step_attention_mask = _build_step_attention_mask(
        frontier=0,
        bucket_size=bucket_size,
        valid_mask_full=valid_mask_full,
        dtype=torch.long,
    )

    loss = next_token_cross_entropy_loss_with_prefix(
        logits,
        input_ids,
        step_attention_mask,
        num_prefix_tokens=num_compression_tokens,
    )
    assert loss.item() < 1e-3, f"masked positions leaked into loss: {loss.item()}"


# ---------------------------------------------------------------------------
# CE-only guard
# ---------------------------------------------------------------------------


class _FakeProcessingClass:
    """Stand-in tokenizer; the guard path never calls into this."""

    def decode(self, ids, skip_special_tokens=True):  # pragma: no cover - guard returns before use
        return ""


def _make_fake_trainer_for_guard(*, loss_type: str = "l2", hybrid_alpha=None):
    """Build a ``ProgressiveCrammingTrainer`` instance bypassing __init__.

    We only need ``self.args`` set for the guard branch to fire.
    """
    args = SimpleNamespace(
        loss_type=loss_type,
        hybrid_alpha=hybrid_alpha,
        progressive_bucket_size=64,
        max_sequence_length=128,
        progressive_reset_lr_scheduler_on_non_convergence=False,
        max_optimization_steps_per_token=4,
        low_dim_projection=False,
        low_dim_proj_train=False,
        learning_rate=1e-2,
        output_dir=None,
    )
    trainer = ProgressiveCrammingTrainer.__new__(ProgressiveCrammingTrainer)
    trainer.args = args
    trainer.processing_class = _FakeProcessingClass()
    return trainer


def _guard_kwargs(device="cpu"):
    """Minimal kwargs to reach the CE-only guard (which raises immediately)."""
    return dict(
        model=None,
        full_input_ids=torch.zeros(1, 1, dtype=torch.long),
        full_model_token_embeddings=torch.zeros(1, 1, 4),
        full_attention_mask=torch.ones(1, 1, dtype=torch.long),
        num_compression_tokens=1,
        compression_tokens_attention_mask=torch.ones(1, 1, dtype=torch.long),
        per_sample_params=None,
        per_sample_optimizers=None,
        per_sample_schedulers=None,
        per_sample_pca_coefficients=None,
        pca_components_device=None,
        pca_mean_device=None,
        init_method="random",
        low_dim_prjoection=None,
        low_dim_optim=None,
        low_dim_scheduler=None,
        per_sample_steps_taken=[0],
        skipped_mask=[False],
        collected_rows=[],
        sample_id_counter=0,
        initialization_embeddings=torch.zeros(1, 1, 4),
        batch_size=1,
        hidden_size=4,
        max_len=1,
        threshold=1.0,
        max_stages_cap=0,
        device=torch.device(device),
    )


def test_ce_only_guard_rejects_non_cross_entropy_loss_type():
    """``loss_type != 'cross_entropy'`` must raise NotImplementedError at entry."""
    trainer = _make_fake_trainer_for_guard(loss_type="l2", hybrid_alpha=None)
    with pytest.raises(NotImplementedError, match="cross_entropy"):
        trainer._train_progressive_bucketed_for_batch(**_guard_kwargs())


def test_ce_only_guard_rejects_hybrid_alpha():
    """Non-None ``hybrid_alpha`` must raise NotImplementedError at entry."""
    trainer = _make_fake_trainer_for_guard(loss_type="cross_entropy", hybrid_alpha=0.5)
    with pytest.raises(NotImplementedError, match="cross_entropy"):
        trainer._train_progressive_bucketed_for_batch(**_guard_kwargs())
