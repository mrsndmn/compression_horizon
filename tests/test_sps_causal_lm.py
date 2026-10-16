"""Behavioral checks for native Standard/SPS cramming, independent of SPS checkout."""

import math

import pytest
import torch
from torch.nn import functional as F

from compression_horizon.models.sps_causal_lm import SPSCausalLM, attention_visibility, load_sps_checkpoint


def tiny_args():
    return dict(
        block_size=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        hidden_size=16,
        intermediate_size=48,
        norm_eps=1e-6,
        dropout=0.0,
        bias=False,
        eos_token_id=29,
        pad_token_id=31,
        predict_token_id=30,
        window_size=0,
    )


def test_sps_zero_window_visibility():
    actual = attention_visibility(torch.ones(1, 3), sps=True)[0, 0]
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(actual, expected)
    # Memory is input slot zero, visible at every later input and predict query.
    assert actual[:, 0].all()


@pytest.mark.parametrize("sps", [False, True])
def test_embeds_ids_padding_causality_and_gradients(sps):
    torch.manual_seed(123)
    model = SPSCausalLM(tiny_args(), sps=sps).eval().requires_grad_(False)
    ids = torch.tensor([[1, 2, 3, 4]])
    embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    original = model(input_ids=ids).logits
    output = model(inputs_embeds=embeds, output_hidden_states=True)
    torch.testing.assert_close(original, output.logits)
    assert len(output.hidden_states) == 3
    changed = ids.clone()
    changed[:, 2:] = 7
    torch.testing.assert_close(original[:, :2], model(input_ids=changed).logits[:, :2])
    padded = torch.tensor([[1, 2, 3, 4, 31, 31], [31, 31, 1, 2, 3, 4]])
    mask = padded.ne(31)
    result = model(input_ids=padded, attention_mask=mask).logits
    torch.testing.assert_close(original, result[:1, :4])
    torch.testing.assert_close(original, result[1:, 2:])
    # Later prediction has a differentiable path to the learned memory input.
    F.cross_entropy(output.logits[:, -1], torch.tensor([5])).backward()
    assert torch.isfinite(embeds.grad).all()
    assert embeds.grad[0, 0].norm() > 0
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("sps", [False, True])
def test_attention_matches_independent_explicit_softmax(sps):
    """Independent reference includes explicit complex RoPE and nested visibility."""
    torch.manual_seed(42)
    model = SPSCausalLM(tiny_args(), sps=sps).eval()
    ids = torch.tensor([[1, 2, 3]])
    x = model.get_input_embeddings()(ids)
    if sps:
        predict = model.get_input_embeddings()(torch.full_like(ids, 30))
        x = torch.stack((x, predict), dim=2).flatten(1, 2)
    n = x.shape[1]
    pos = torch.arange(3).repeat_interleave(2) if sps else torch.arange(3)
    angles = pos[:, None] * (10000.0 ** (-torch.arange(0, 8, 2).float() / 8))
    rotation = torch.polar(torch.ones_like(angles), angles)
    allowed = torch.tensor([[k <= q and (not sps or k % 2 == 0 or k // 2 == q // 2) for k in range(n)] for q in range(n)])
    for block in model.transformer.h:
        h = block.attention_norm(x)
        q, k, v = block.attn.c_attn(h).chunk(3, dim=-1)

        def rope(h):
            z = torch.view_as_complex(h.reshape(1, n, 2, 4, 2).contiguous())
            return torch.view_as_real(z * rotation[None, :, None]).flatten(-2).transpose(1, 2)

        q, k = rope(q), rope(k)
        v = v.reshape(1, n, 2, 8).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(8)
        probs = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        a = (probs @ v).transpose(1, 2).reshape(1, n, 16)
        x = x + block.attn.c_proj(a)
        x = x + block.mlp(block.mlp_norm(x))
    x = model.transformer.output_norm(x)
    if sps:
        x = x[:, 1::2]
    torch.testing.assert_close(model(input_ids=ids).logits, model.lm_head(x), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("sps", [False, True])
def test_strict_native_loading_and_rope_dtype(tmp_path, sps):
    model = SPSCausalLM(tiny_args(), sps=sps)
    target = "modeling.models.sps.SPSModel" if sps else "modeling.models.full_attention_model.Model"
    ckpt = dict(model_args=tiny_args(), model=model.state_dict(), config={"model": {"_target_": target}})
    path = tmp_path / "model.pt"
    torch.save(ckpt, path)
    loaded = load_sps_checkpoint(path, torch_dtype=torch.bfloat16)
    assert loaded.freqs_cis.is_complex()
    assert loaded.freqs_cis[1].imag.abs().sum() > 0
    assert loaded.get_input_embeddings().weight is loaded.lm_head.weight
    assert loaded.dtype == torch.bfloat16
    torch.testing.assert_close(loaded.lm_head.weight, model.lm_head.weight.to(torch.bfloat16))
    del ckpt["model"]["transformer.h.0.attn.c_proj.weight"]
    torch.save(ckpt, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_sps_checkpoint(path)


@pytest.mark.parametrize("sps", [False, True])
def test_progressive_trainer_native_smoke(tmp_path, sps):
    from tests.trainer_helpers import TinyDataset, _collate_batch, _make_args

    from compression_horizon.models.sps_causal_lm import load_sps_tokenizer
    from compression_horizon.train import ProgressiveCrammingTrainer

    model = SPSCausalLM(tiny_args(), sps=sps).eval()
    args = _make_args(
        output_dir=str(tmp_path),
        progressive_train=True,
        progressive_min_seq_len=2,
        progressive_step=1,
        progressive_max_stages=2,
        max_optimization_steps_per_sample=2,
        max_optimization_steps_per_token=1,
        number_of_mem_tokens=1,
        logging_dir=None,
    )
    trainer = ProgressiveCrammingTrainer(
        model=model,
        processing_class=load_sps_tokenizer(),
        args=args,
        train_dataset=TinyDataset(num_samples=2, seq_len=8, vocab_size=16),
        data_collator=_collate_batch,
    )
    out = trainer.train()
    assert out and (tmp_path / "progressive_prefixes").exists()


@pytest.mark.parametrize("sps", [False, True])
def test_information_gain_without_memory_is_zero_across_eos(sps):
    from compression_horizon.analysis.information_gain import compute_information_gain

    torch.manual_seed(42)
    model = SPSCausalLM(tiny_args(), sps=sps).eval()
    ids = torch.tensor([[1, 29, 2, 3]])
    embeddings = model.get_input_embeddings()(ids)
    mask = torch.ones_like(ids)
    # Make sure EOS inference really distinguishes these contexts: a zero-IG
    # assertion would be vacuous if both forward paths accidentally matched.
    assert not torch.allclose(model(input_ids=ids).logits, model(inputs_embeds=embeddings).logits)
    gain = compute_information_gain(
        model=model,
        input_ids=ids,
        attention_mask=mask,
        token_embeddings=embeddings,
        compression_token_embeddings=embeddings[:, :0],
        compression_attention_mask=mask[:, :0],
    )
    assert gain == [0.0]
