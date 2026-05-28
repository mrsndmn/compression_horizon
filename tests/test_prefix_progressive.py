"""Tests for fixed-prefix progressive cramming.

Covers the prefix-aware pieces added for the fixed-prefix ablation:
  * loss / convergence offset = num_compression_tokens + prefix_len (prefix_len=0 == original),
  * build_united_input inserting a [mem][prefix][continuation] block,
  * compute_information_gain measured over the continuation on top of a real prefix,
  * compute_prefix_surprisal_bits_per_token,
  * an end-to-end ProgressiveCrammingTrainer smoke test with a prefix (CUDA only).

The pure-function tests are CPU-only and use a tiny GPT2 built from config (no download).
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from compression_horizon.analysis.information_gain import (
    _sequence_cross_entropy_bits,
    compute_information_gain,
    compute_prefix_surprisal_bits_per_token,
)
from compression_horizon.train.inputs import build_united_input
from compression_horizon.train.loss import (
    next_token_cross_entropy_loss_with_prefix,
    token_argmax_match_rate_with_prefix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _tiny_gpt2(seed: int = 0) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=2, bos_token_id=0, eos_token_id=0)
    model = GPT2LMHeadModel(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# --------------------------------------------------------------------------- #
# Loss / convergence offset parity.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("C, P, L", [(1, 0, 6), (1, 3, 5), (2, 4, 4), (1, 7, 3)])
def test_ce_loss_prefix_offset_matches_manual_slice(C, P, L):
    """CE over the continuation must use the logits window [C+P-1 : -1]; P=0 reproduces the original."""
    torch.manual_seed(0)
    vocab = 11
    logits = torch.randn(2, C + P + L, vocab)
    cont_ids = torch.randint(0, vocab, (2, L))
    cont_mask = torch.ones(2, L, dtype=torch.long)

    got = next_token_cross_entropy_loss_with_prefix(logits, cont_ids, cont_mask, C, prefix_len=P)
    expected = F.cross_entropy(logits[:, C + P - 1 : -1].reshape(-1, vocab), cont_ids.reshape(-1))
    assert torch.allclose(got, expected, atol=1e-6)


def test_ce_loss_prefix_zero_equals_no_prefix_default():
    """prefix_len=0 must be byte-identical to the call without the kwarg (no behavioural change)."""
    torch.manual_seed(1)
    vocab, C, L = 13, 1, 7
    logits = torch.randn(3, C + L, vocab)
    ids = torch.randint(0, vocab, (3, L))
    mask = torch.ones(3, L, dtype=torch.long)
    assert next_token_cross_entropy_loss_with_prefix(
        logits, ids, mask, C, prefix_len=0
    ) == next_token_cross_entropy_loss_with_prefix(logits, ids, mask, C)


@pytest.mark.parametrize("C, P, L", [(1, 0, 6), (1, 3, 5), (2, 4, 4)])
def test_argmax_match_rate_prefix_offset(C, P, L):
    """Match-rate must be measured over the continuation window; P=0 reproduces the original."""
    torch.manual_seed(2)
    vocab = 17
    logits = torch.randn(2, C + P + L, vocab)
    cont_ids = torch.randint(0, vocab, (2, L))
    cont_mask = torch.ones(2, L, dtype=torch.long)

    got = token_argmax_match_rate_with_prefix(logits, cont_ids, cont_mask, C, prefix_len=P)
    preds = logits[:, C + P - 1 : -1].argmax(dim=-1)
    expected = ((preds == cont_ids) & (cont_mask == 1)).sum(dim=-1) / cont_mask.sum(dim=-1).clamp_min(1)
    assert torch.allclose(got, expected)


# --------------------------------------------------------------------------- #
# build_united_input layout.
# --------------------------------------------------------------------------- #
def test_build_united_input_prefix_layout_and_order():
    """Prefix block must sit between compression tokens and the sequence; masks line up."""
    B, C, P, L, H = 2, 1, 3, 4, 5
    mem = torch.full((B, C, H), 1.0)
    mem_mask = torch.ones(B, C, dtype=torch.long)
    prefix = torch.full((B, P, H), 2.0)
    prefix_mask = torch.ones(B, P, dtype=torch.long)
    seq = torch.full((B, L, H), 3.0)
    seq_mask = torch.ones(B, L, dtype=torch.long)

    emb, mask = build_united_input(
        mem, mem_mask, seq, seq_mask, prefix_token_embeddings=prefix, prefix_attention_mask=prefix_mask
    )
    assert emb.shape == (B, C + P + L, H)
    assert mask.shape == (B, C + P + L)
    assert torch.all(emb[:, :C] == 1.0)
    assert torch.all(emb[:, C : C + P] == 2.0)
    assert torch.all(emb[:, C + P :] == 3.0)


def test_build_united_input_no_prefix_identical():
    """With no prefix the result equals the original [mem][seq] concatenation."""
    B, C, L, H = 2, 1, 4, 5
    mem = torch.randn(B, C, H)
    mem_mask = torch.ones(B, C, dtype=torch.long)
    seq = torch.randn(B, L, H)
    seq_mask = torch.ones(B, L, dtype=torch.long)
    emb, mask = build_united_input(mem, mem_mask, seq, seq_mask)
    assert torch.equal(emb, torch.cat([mem, seq], dim=1))
    assert torch.equal(mask, torch.cat([mem_mask, seq_mask], dim=1))


def test_build_united_input_prefix_requires_mask():
    with pytest.raises(ValueError):
        build_united_input(
            torch.zeros(1, 1, 4),
            torch.ones(1, 1),
            torch.zeros(1, 2, 4),
            torch.ones(1, 2),
            prefix_token_embeddings=torch.zeros(1, 3, 4),
        )


# --------------------------------------------------------------------------- #
# Prefix-aware information gain.
# --------------------------------------------------------------------------- #
def test_information_gain_with_prefix_matches_manual_reference():
    """IG with a prefix == H_lm(c|prefix) - H_comp_lm(c|mem,prefix), measured over the continuation."""
    model = _tiny_gpt2(seed=0)
    torch.manual_seed(7)
    B, C, P, L, H, vocab = 2, 1, 3, 5, model.config.n_embd, model.config.vocab_size

    cont_ids = torch.randint(1, vocab, (B, L))
    cont_mask = torch.ones(B, L, dtype=torch.long)
    cont_emb = model.get_input_embeddings()(cont_ids).to(torch.float32)
    prefix_ids = torch.randint(1, vocab, (B, P))
    prefix_mask = torch.ones(B, P, dtype=torch.long)
    prefix_emb = model.get_input_embeddings()(prefix_ids).to(torch.float32)
    mem = torch.randn(B, C, H) * 0.1
    mem_mask = torch.ones(B, C, dtype=torch.long)

    got = compute_information_gain(
        model=model,
        input_ids=cont_ids,
        attention_mask=cont_mask,
        token_embeddings=cont_emb,
        compression_token_embeddings=mem,
        compression_attention_mask=mem_mask,
        prefix_token_embeddings=prefix_emb,
        prefix_attention_mask=prefix_mask,
    )

    expected = []
    for j in range(B):
        base_logits = model(
            inputs_embeds=torch.cat([prefix_emb[j : j + 1], cont_emb[j : j + 1]], dim=1),
            attention_mask=torch.cat([prefix_mask[j : j + 1], cont_mask[j : j + 1]], dim=1),
        ).logits
        h_base = _sequence_cross_entropy_bits(base_logits, cont_ids[j : j + 1], cont_mask[j : j + 1], align_offset=P)
        comp_logits = model(
            inputs_embeds=torch.cat([mem[j : j + 1], prefix_emb[j : j + 1], cont_emb[j : j + 1]], dim=1),
            attention_mask=torch.cat([mem_mask[j : j + 1], prefix_mask[j : j + 1], cont_mask[j : j + 1]], dim=1),
        ).logits
        h_comp = _sequence_cross_entropy_bits(comp_logits, cont_ids[j : j + 1], cont_mask[j : j + 1], align_offset=C + P)
        expected.append(h_base - h_comp)

    assert len(got) == B
    for g, e in zip(got, expected):
        assert g == e


def test_information_gain_no_prefix_unchanged():
    """Passing no prefix must equal the original (no-prefix) computation."""
    model = _tiny_gpt2(seed=0)
    torch.manual_seed(7)
    B, C, L, H, vocab = 2, 1, 6, model.config.n_embd, model.config.vocab_size
    ids = torch.randint(1, vocab, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    emb = model.get_input_embeddings()(ids).to(torch.float32)
    mem = torch.randn(B, C, H) * 0.1
    mem_mask = torch.ones(B, C, dtype=torch.long)
    out = compute_information_gain(
        model=model,
        input_ids=ids,
        attention_mask=mask,
        token_embeddings=emb,
        compression_token_embeddings=mem,
        compression_attention_mask=mem_mask,
    )
    assert len(out) == B
    for v in out:
        assert math.isfinite(v)


# --------------------------------------------------------------------------- #
# Prefix surprisal.
# --------------------------------------------------------------------------- #
def test_prefix_surprisal_bits_per_token_matches_manual():
    model = _tiny_gpt2(seed=0)
    torch.manual_seed(9)
    B, P, vocab = 2, 5, model.config.vocab_size
    prefix_ids = torch.randint(1, vocab, (B, P))
    prefix_mask = torch.ones(B, P, dtype=torch.long)

    got = compute_prefix_surprisal_bits_per_token(model=model, prefix_input_ids=prefix_ids, prefix_attention_mask=prefix_mask)
    assert len(got) == B
    for j in range(B):
        logits = model(input_ids=prefix_ids[j : j + 1], attention_mask=prefix_mask[j : j + 1]).logits
        total = _sequence_cross_entropy_bits(logits, prefix_ids[j : j + 1], prefix_mask[j : j + 1], align_offset=0)
        scored = int(prefix_mask[j : j + 1][:, 1:].sum().item())
        assert got[j] == pytest.approx(total / scored)


def test_prefix_surprisal_zero_when_no_scorable_tokens():
    model = _tiny_gpt2(seed=0)
    prefix_ids = torch.randint(1, model.config.vocab_size, (2, 1))  # length-1 prefix => nothing to score
    prefix_mask = torch.ones(2, 1, dtype=torch.long)
    assert compute_prefix_surprisal_bits_per_token(
        model=model, prefix_input_ids=prefix_ids, prefix_attention_mask=prefix_mask
    ) == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# End-to-end trainer smoke test (CUDA).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_progressive_cramming_prefix_smoke():
    """Run ProgressiveCrammingTrainer with a fixed prefix; rows must carry prefix metadata."""
    from datasets import load_from_disk
    from tests.trainer_helpers import TinyDataset, _collate_batch, _make_args
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from compression_horizon.train import ProgressiveCrammingTrainer

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    import tempfile

    with tempfile.TemporaryDirectory() as out_dir:
        args = _make_args(
            progressive_train=True,
            progressive_min_seq_len=2,
            progressive_step=1,
            progressive_max_stages=2,
            max_optimization_steps_per_sample=1,
            max_optimization_steps_per_token=1,
            number_of_mem_tokens=1,
            progressive_prefix_len=4,
            output_dir=out_dir,
            logging_dir=None,
        )
        dataset = TinyDataset(num_samples=2, seq_len=12, vocab_size=16)  # 12 = 4 prefix + room to cram
        trainer = ProgressiveCrammingTrainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=dataset,
            eval_dataset=None,
            data_collator=_collate_batch,
        )
        out = trainer.train()
        assert out is not None
        rows = load_from_disk(out)
        assert len(rows) > 0
        for row in rows:
            assert row["progressive_prefix_len"] == 4
            assert row["prefix_surprisal_bits_per_token"] is not None
            assert math.isfinite(row["prefix_surprisal_bits_per_token"])
            # The crammed continuation excludes the prefix: stage seq_len <= (12 - 4).
            assert row["stage_seq_len"] <= 8
