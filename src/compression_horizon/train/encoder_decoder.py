"""Encoder-decoder (seq2seq) support for cramming.

The decoder-only trainers cram by prepending the memory embedding to the sequence:
``[mem] [tokens]``, so the memory occupies decoder position 0 and every loss/convergence
helper slices logits at ``offset = num_compression_tokens + prefix_len``.

An encoder-decoder model has no such position. The memory lives on the *encoder* side and
the decoder reaches it only through cross-attention, so the memory occupies **zero** decoder
positions and that offset arithmetic does not apply. Feeding a seq2seq model through the
decoder-only path would silently score shifted tokens, which is why this adapter exists.

Two injection sites (see ``docs/adr/0005-encoder-effect-attributed-by-injection-site-control.md``):

* ``encoder_input`` — the memory embedding is the encoder's *input*; the decoder reaches it
  only after the full encoder stack. This is the arm the encoder hypothesis is about.
* ``encoder_output`` — the memory embedding *is* the encoder output the decoder's
  cross-attention reads; the encoder is never run. The control that removes the encoder.

To keep the loss helpers untouched, the decoder is teacher-forced on
``[start] [target tokens]``: logits at position ``t`` predict target token ``t``, so the
tensor has exactly the ``[one leading position] [sequence]`` shape the decoder-only layout
produces and ``LOGITS_OFFSET = 1`` reproduces the existing slicing. That 1 counts the
decoder start token, **not** the memory.
"""

from __future__ import annotations

import torch
from transformers.modeling_outputs import BaseModelOutput

ENCODER_INPUT = "encoder_input"
ENCODER_OUTPUT = "encoder_output"
INJECTION_SITES = (ENCODER_INPUT, ENCODER_OUTPUT)


def is_encoder_decoder_model(model) -> bool:
    """True when ``model`` is a seq2seq model whose decoder cross-attends to an encoder."""
    if model is None:
        return False
    return bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))


def _resolve_decoder_start_token_id(model) -> int:
    config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)
    candidates = (
        getattr(config, "decoder_start_token_id", None),
        getattr(generation_config, "decoder_start_token_id", None),
        getattr(getattr(config, "decoder", None), "bos_token_id", None),
        getattr(config, "bos_token_id", None),
        getattr(generation_config, "bos_token_id", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return int(candidate)
    raise ValueError(
        "Cannot determine the decoder start token id for this encoder-decoder model; " "set config.decoder_start_token_id."
    )


def _decoder_input_embeddings(model) -> torch.nn.Module:
    """The decoder's token-embedding module (may differ from ``model.get_input_embeddings()``)."""
    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        decoder = get_decoder()
        embeddings = decoder.get_input_embeddings() if hasattr(decoder, "get_input_embeddings") else None
        if embeddings is not None:
            return embeddings
    inner = getattr(model, "model", None)
    decoder = getattr(inner, "decoder", None)
    embed_tokens = getattr(decoder, "embed_tokens", None)
    if embed_tokens is not None:
        return embed_tokens
    return model.get_input_embeddings()


def _encoder_input_embeddings(model) -> torch.nn.Module:
    """The encoder's token-embedding module."""
    get_encoder = getattr(model, "get_encoder", None)
    if callable(get_encoder):
        encoder = get_encoder()
        embeddings = encoder.get_input_embeddings() if hasattr(encoder, "get_input_embeddings") else None
        if embeddings is not None:
            return embeddings
    inner = getattr(model, "model", None)
    encoder = getattr(inner, "encoder", None)
    embed_tokens = getattr(encoder, "embed_tokens", None)
    if embed_tokens is not None:
        return embed_tokens
    return model.get_input_embeddings()


class EncoderDecoderCrammingAdapter:
    """Runs seq2seq forwards with the memory embedding on the encoder side.

    ``LOGITS_OFFSET`` is 1 because the decoder is fed a leading start token, not because the
    memory takes a decoder slot — it takes none regardless of ``num_compression_tokens``.
    """

    LOGITS_OFFSET = 1

    def __init__(self, model, injection_site: str = ENCODER_INPUT) -> None:
        if injection_site not in INJECTION_SITES:
            raise ValueError(f"encoder_decoder_injection_site must be one of {INJECTION_SITES}, got {injection_site!r}!")
        if not is_encoder_decoder_model(model):
            raise ValueError("EncoderDecoderCrammingAdapter requires a model with config.is_encoder_decoder=True!")
        self.model = model
        self.injection_site = injection_site
        self.decoder_start_token_id = _resolve_decoder_start_token_id(model)
        self.decoder_embeddings = _decoder_input_embeddings(model)
        self.encoder_embeddings = _encoder_input_embeddings(model)

    # ------------------------------------------------------------------
    # Shapes and dtypes
    # ------------------------------------------------------------------

    @property
    def memory_hidden_size(self) -> int:
        """Hidden size of the space the memory embedding lives in (encoder input or output)."""
        config = self.model.config
        encoder_config = getattr(config, "encoder", None) or config
        return int(encoder_config.hidden_size)

    @property
    def model_dtype(self) -> torch.dtype:
        return self.decoder_embeddings.weight.dtype

    def embed_target_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Decoder-side token embeddings for the reconstruction target."""
        return self.decoder_embeddings(input_ids)

    # ------------------------------------------------------------------
    # Forwards
    # ------------------------------------------------------------------

    def build_decoder_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[start] [target]`` teacher-forcing inputs, so logits[t] predicts target[t]."""
        batch_size = input_ids.size(0)
        start = torch.full(
            (batch_size, 1),
            self.decoder_start_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        decoder_input_ids = torch.cat((start, input_ids), dim=1)
        start_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        decoder_attention_mask = torch.cat((start_mask, attention_mask), dim=1)
        return decoder_input_ids, decoder_attention_mask

    def forward(
        self,
        *,
        memory_embeds: torch.Tensor,  # [batch, compression, hidden]
        memory_mask: torch.Tensor,  # [batch, compression]
        input_ids: torch.Tensor,  # [batch, sequence] -- reconstruction target
        attention_mask: torch.Tensor,  # [batch, sequence]
    ):
        """Forward with the memory on the encoder side. Logits are ``[batch, 1 + sequence, vocab]``."""
        decoder_input_ids, decoder_attention_mask = self.build_decoder_inputs(input_ids, attention_mask)
        memory_embeds = memory_embeds.to(device=decoder_input_ids.device, dtype=self.model_dtype)
        common = {
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": False,
        }
        if self.injection_site == ENCODER_INPUT:
            return self.model(inputs_embeds=memory_embeds, attention_mask=memory_mask, **common)
        # encoder_output: hand the decoder's cross-attention the memory directly; the encoder
        # is not in the graph at all, which is exactly what makes this the "no encoder" control.
        return self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory_embeds),
            attention_mask=memory_mask,
            **common,
        )

    def forward_without_memory(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Base forward with no information on the encoder side (a single start token).

        The reference entropy for information gain: the same decoder, teacher-forced on the same
        target, conditioned on an encoder that was given nothing. Used identically by both
        injection sites so their information-gain numbers are mutually comparable.
        """
        decoder_input_ids, decoder_attention_mask = self.build_decoder_inputs(input_ids, attention_mask)
        batch_size = input_ids.size(0)
        encoder_input_ids = torch.full(
            (batch_size, 1),
            self.decoder_start_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        encoder_attention_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        return self.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
        )
