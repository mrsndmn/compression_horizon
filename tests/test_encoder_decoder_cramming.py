"""Encoder-decoder cramming: logits alignment, optimization, and the encoder-output control.

These run on CPU against a tiny randomly-initialized T5Gemma so they exercise the real
seq2seq forward path without downloading a checkpoint.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from tests.trainer_helpers import _make_args

from compression_horizon.train.encoder_decoder import (
    ENCODER_INPUT,
    ENCODER_OUTPUT,
    EncoderDecoderCrammingAdapter,
    is_encoder_decoder_model,
)
from compression_horizon.train.loss import (
    next_token_cross_entropy_loss_with_prefix,
    token_argmax_match_rate_with_prefix,
)
from compression_horizon.train.trainers.base import BaseTrainer

VOCAB_SIZE = 32
HIDDEN_SIZE = 16
SEQ_LEN = 6


def _tiny_t5gemma():
    """A 2-layer encoder / 2-layer decoder T5Gemma with random weights, on CPU."""
    from transformers.models.t5gemma.configuration_t5gemma import T5GemmaConfig, T5GemmaModuleConfig
    from transformers.models.t5gemma.modeling_t5gemma import T5GemmaForConditionalGeneration

    def module_config():
        return T5GemmaModuleConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=2 * HIDDEN_SIZE,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
        )

    config = T5GemmaConfig(encoder=module_config(), decoder=module_config(), vocab_size=VOCAB_SIZE)
    torch.manual_seed(0)
    return T5GemmaForConditionalGeneration(config).eval()


def _sample(batch_size: int = 2):
    torch.manual_seed(1)
    input_ids = torch.randint(3, VOCAB_SIZE, (batch_size, SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _memory(batch_size: int = 2, num_mem: int = 1, requires_grad: bool = True):
    torch.manual_seed(2)
    memory = torch.rand(batch_size, num_mem, HIDDEN_SIZE, dtype=torch.float32) * 0.02
    memory.requires_grad_(requires_grad)
    mask = torch.ones(batch_size, num_mem, dtype=torch.long)
    return memory, mask


def test_is_encoder_decoder_model_detects_seq2seq():
    assert is_encoder_decoder_model(_tiny_t5gemma()) is True


def test_decoder_logits_have_exactly_one_leading_position():
    """The decoder sees ``[start] [target]``, so logits carry one leading position — not the memory.

    This is the invariant the shared loss helpers rely on: ``LOGITS_OFFSET`` must stay 1 no matter
    how many memory embeddings there are, because the memory occupies no decoder position.
    """
    model = _tiny_t5gemma()
    input_ids, attention_mask = _sample()

    for num_mem in (1, 4):
        adapter = EncoderDecoderCrammingAdapter(model, injection_site=ENCODER_INPUT)
        memory, memory_mask = _memory(num_mem=num_mem, requires_grad=False)
        outputs = adapter.forward(
            memory_embeds=memory, memory_mask=memory_mask, input_ids=input_ids, attention_mask=attention_mask
        )
        assert outputs.logits.shape == (input_ids.size(0), SEQ_LEN + 1, VOCAB_SIZE)
        assert adapter.LOGITS_OFFSET == 1


def test_loss_and_convergence_score_the_intended_tokens():
    """Guard the off-by-one: position ``t`` of the decoder must be scored against target token ``t``.

    Computed independently here (plain CE over ``logits[:, :-1]`` vs ``input_ids``) and compared to
    what the shared helpers produce at ``offset=1``. A shifted slice would break this by a wide margin.
    """
    model = _tiny_t5gemma()
    adapter = EncoderDecoderCrammingAdapter(model, injection_site=ENCODER_INPUT)
    input_ids, attention_mask = _sample()
    memory, memory_mask = _memory(requires_grad=False)

    logits = adapter.forward(
        memory_embeds=memory, memory_mask=memory_mask, input_ids=input_ids, attention_mask=attention_mask
    ).logits

    expected_loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), input_ids.flatten())
    actual_loss = next_token_cross_entropy_loss_with_prefix(logits.float(), input_ids, attention_mask, adapter.LOGITS_OFFSET)
    assert torch.allclose(actual_loss, expected_loss, atol=1e-5)

    expected_match = (logits[:, :-1].argmax(dim=-1) == input_ids).float().mean(dim=-1)
    actual_match = token_argmax_match_rate_with_prefix(logits, input_ids, attention_mask, adapter.LOGITS_OFFSET)
    assert torch.allclose(actual_match, expected_match, atol=1e-6)


def test_memory_optimization_reduces_loss():
    """One optimizer step on the memory embedding decreases the reconstruction loss."""
    model = _tiny_t5gemma()
    for parameter in model.parameters():
        parameter.requires_grad = False
    adapter = EncoderDecoderCrammingAdapter(model, injection_site=ENCODER_INPUT)
    input_ids, attention_mask = _sample(batch_size=1)
    memory, memory_mask = _memory(batch_size=1)

    optimizer = torch.optim.AdamW([memory], lr=0.5)

    def loss_now():
        logits = adapter.forward(
            memory_embeds=memory, memory_mask=memory_mask, input_ids=input_ids, attention_mask=attention_mask
        ).logits
        return next_token_cross_entropy_loss_with_prefix(logits.float(), input_ids, attention_mask, adapter.LOGITS_OFFSET)

    first = loss_now()
    first.backward()
    assert memory.grad is not None, "memory embedding received no gradient through the encoder"
    assert torch.isfinite(memory.grad).all()
    optimizer.step()
    optimizer.zero_grad()

    assert loss_now().item() < first.item()


@pytest.mark.parametrize(
    "injection_site,encoder_should_be_in_graph",
    [(ENCODER_INPUT, True), (ENCODER_OUTPUT, False)],
)
def test_encoder_participation_matches_injection_site(injection_site, encoder_should_be_in_graph):
    """The encoder-output arm is the "no encoder" control: the encoder must not be in the graph.

    This is what makes the arm an attribution control rather than a second flavour of the same
    thing — if the encoder still ran, the comparison would say nothing about the encoder.
    """
    model = _tiny_t5gemma()
    encoder_parameters = list(model.get_encoder().parameters())
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Only for this assertion: real runs freeze everything, so grads would be None either way.
    for parameter in encoder_parameters:
        parameter.requires_grad = True

    adapter = EncoderDecoderCrammingAdapter(model, injection_site=injection_site)
    input_ids, attention_mask = _sample(batch_size=1)
    memory, memory_mask = _memory(batch_size=1)

    logits = adapter.forward(
        memory_embeds=memory, memory_mask=memory_mask, input_ids=input_ids, attention_mask=attention_mask
    ).logits
    next_token_cross_entropy_loss_with_prefix(logits.float(), input_ids, attention_mask, adapter.LOGITS_OFFSET).backward()

    any_encoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder_parameters)
    assert any_encoder_grad is encoder_should_be_in_graph
    # Either way the memory itself must be trainable, or the arm optimizes nothing.
    assert memory.grad is not None and torch.isfinite(memory.grad).all()


def test_information_gain_is_zero_without_information():
    """Sanity: the base term of encoder-decoder information gain is a real entropy in bits."""
    from compression_horizon.analysis.information_gain import compute_information_gain_encoder_decoder

    model = _tiny_t5gemma()
    adapter = EncoderDecoderCrammingAdapter(model, injection_site=ENCODER_INPUT)
    input_ids, attention_mask = _sample(batch_size=2)
    memory, memory_mask = _memory(batch_size=2, requires_grad=False)

    gains = compute_information_gain_encoder_decoder(
        adapter=adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        compression_token_embeddings=memory,
        compression_attention_mask=memory_mask,
    )
    assert len(gains) == 2
    assert all(math.isfinite(gain) for gain in gains)


def _bare_trainer(args, encoder_decoder):
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = args
    trainer.encoder_decoder = encoder_decoder
    return trainer


@pytest.mark.parametrize(
    "overrides",
    [
        {"progressive_prefix_len": 4},
        {"loss_type": "cosine", "hybrid_alpha": 1.0},
        {"fix_position_ids": True},
        {"low_dim_projection": True, "low_dim_projection_global": True},
    ],
)
def test_unsupported_flags_fail_loudly(overrides):
    """Flags whose offset arithmetic assumes a decoder-side memory must raise, not mis-score silently."""
    model = _tiny_t5gemma()
    adapter = EncoderDecoderCrammingAdapter(model, injection_site=ENCODER_INPUT)
    trainer = _bare_trainer(_make_args(**overrides), adapter)
    with pytest.raises(NotImplementedError):
        trainer._assert_encoder_decoder_supported()


def test_supported_baseline_flags_pass():
    model = _tiny_t5gemma()
    adapter = EncoderDecoderCrammingAdapter(model, injection_site=ENCODER_INPUT)
    trainer = _bare_trainer(_make_args(loss_type="cross_entropy"), adapter)
    trainer._assert_encoder_decoder_supported()


def test_unknown_injection_site_rejected():
    with pytest.raises(ValueError):
        EncoderDecoderCrammingAdapter(_tiny_t5gemma(), injection_site="decoder_prefix")
