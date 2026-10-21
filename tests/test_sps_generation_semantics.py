"""SPS w=0 training/generation parity with ordinary-only persistent history."""

from types import SimpleNamespace

import pytest
import torch
from scripts.validation.validate_sps_generation import ordinary_history_next_logits
from tests.test_sps_causal_lm import tiny_args
from torch import nn

from compression_horizon.inference.generation import generate_from_compression
from compression_horizon.models.sps_causal_lm import SPSCausalLM


def test_teacher_forcing_logits_and_memory_gradients_match_ordinary_history():
    torch.manual_seed(73)
    model = SPSCausalLM(tiny_args(), sps=True).eval().requires_grad_(False)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    memory = torch.randn(2, 1, 16, requires_grad=True)
    context = torch.cat((memory, model.get_input_embeddings()(ids)), dim=1)
    actual = model(inputs_embeds=context).logits
    reference = torch.stack([ordinary_history_next_logits(model, context[:, :n]) for n in range(1, 5)], dim=1)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    # Every scored target uses exactly the same memory -> logits path in both layouts.
    weights = torch.randn_like(actual[:, :-1])
    grad_actual = torch.autograd.grad((actual[:, :-1] * weights).sum(), memory, retain_graph=True)[0]
    grad_reference = torch.autograd.grad((reference[:, :-1] * weights).sum(), memory)[0]
    torch.testing.assert_close(grad_actual, grad_reference, atol=2e-6, rtol=2e-5)


def test_historical_prediction_states_cannot_change_next_logits():
    torch.manual_seed(19)
    model = SPSCausalLM(tiny_args(), sps=True).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(input_ids=ids).logits[:, -1]

    def corrupt_old_predictions(module, args):
        x, freqs = args
        altered = x.clone()
        altered[:, 1:-1:2] = torch.randn_like(altered[:, 1:-1:2]) * 100
        return altered, freqs

    handle = model.transformer.h[0].register_forward_pre_hook(corrupt_old_predictions)
    try:
        actual = model(input_ids=ids).logits[:, -1]
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


class SmallTokenizer:
    eos_token_id = None
    eos_token = None
    pad_token = "<pad>"

    def __len__(self):
        return 30

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(x)) for x in ids)

    def batch_decode(self, ids, skip_special_tokens=True):
        return [self.decode(row) for row in ids]


class FixedLogitLM(nn.Module):
    """Force unused vocabulary/pad/predict logits above the best normal token."""

    def __init__(self, native=True):
        super().__init__()
        self.embeddings = nn.Embedding(34, 4)
        self.config = SimpleNamespace(sps=True, predict_token_id=30, pad_token_id=31) if native else SimpleNamespace()
        self.contexts = []

    @property
    def device(self):
        return self.embeddings.weight.device

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, inputs_embeds, attention_mask):
        self.contexts.append(inputs_embeds.clone())
        logits = torch.zeros(*inputs_embeds.shape[:2], 34)
        logits[..., 2] = 10
        logits[..., 30] = 100
        logits[..., 31] = 200
        logits[..., 33] = 300
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("path", ["greedy", "perplexity_generation"])
def test_native_decoders_feed_back_only_ordinary_tokens(path):
    model = FixedLogitLM()
    tokenizer = SmallTokenizer()
    memory = torch.randn(1, 1, 4)
    if path == "greedy":
        _, ids = generate_from_compression(model, tokenizer, memory, max_new_tokens=2, return_generated_ids=True)
        assert ids.tolist() == [[2, 2]]
    else:
        from compression_horizon.analysis.perplexity import calculate_perplexity

        _, text = calculate_perplexity(
            model, tokenizer, memory, torch.empty(1, 0, 4), torch.empty(1, 0, dtype=torch.long), n=2, return_generated_text=True
        )
        assert text == "2 2"
    assert [x.shape[1] for x in model.contexts] == [1, 2]
    torch.testing.assert_close(model.contexts[1][:, 1], model.get_input_embeddings()(torch.tensor([2])))


def test_foreign_hf_decode_behavior_is_preserved():
    model = FixedLogitLM(native=False)
    _, ids = generate_from_compression(
        model, SmallTokenizer(), torch.zeros(1, 1, 4), max_new_tokens=1, return_generated_ids=True
    )
    assert ids.tolist() == [[33]]


def test_native_decoder_preserves_eos_and_stops():
    model = FixedLogitLM()
    model.config.pad_token_id = 29  # An EOS/pad overlap must not forbid EOS.
    original = model.forward

    def force_eos(*args, **kwargs):
        result = original(*args, **kwargs)
        result.logits[..., 29] = 400
        return result

    model.forward = force_eos
    tokenizer = SmallTokenizer()
    tokenizer.eos_token_id = 29
    tokenizer.eos_token = "<eos>"
    _, ids = generate_from_compression(model, tokenizer, torch.zeros(1, 1, 4), max_new_tokens=3, return_generated_ids=True)
    assert ids.tolist() == [[29]]
    assert len(model.contexts) == 1


def test_sps_greedy_matches_one_predict_with_ordinary_only_history():
    torch.manual_seed(94)
    model = SPSCausalLM(tiny_args(), sps=True).eval().requires_grad_(False)
    tokenizer = SmallTokenizer()
    memory = torch.randn(1, 1, 16)
    _, actual = generate_from_compression(model, tokenizer, memory, max_new_tokens=5, return_generated_ids=True)
    ordinary_context = memory.clone()
    expected = []
    for _ in range(5):
        logits = ordinary_history_next_logits(model, ordinary_context)
        token = logits[:, : len(tokenizer)].argmax(-1)
        expected.append(token.item())
        ordinary_context = torch.cat((ordinary_context, model.get_input_embeddings()(token)[:, None]), dim=1)
    assert actual.tolist() == [expected]
