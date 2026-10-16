"""Cramming adapter for native Standard/SPS checkpoints from the SPS project.

Transformer/RoPE components are vendored unchanged except for local imports.
SDPA replaces Flex/Triton attention to support gradients to arbitrary input
embeddings and right padding. SPS retains interleaved input/predict slots,
shared RoPE positions, its temporary-key window, and odd-slot readout.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .sps_vendor.components import Block, CausalSelfAttention, RMSNorm, apply_rotary_emb, precompute_freqs_cis


def attention_visibility(attention_mask, *, sps=False, window_size=0, document_ids=None):
    """True means visible. SPS input keys persist; predict keys expire after w tokens."""
    valid = attention_mask.bool()
    if sps:
        valid = valid.repeat_interleave(2, dim=1)
        if document_ids is not None:
            document_ids = document_ids.repeat_interleave(2, dim=1)
    n = valid.shape[1]
    q = torch.arange(n, device=valid.device)[:, None]
    k = torch.arange(n, device=valid.device)[None, :]
    allowed = q >= k
    if sps:
        allowed = allowed & ((k % 2 == 0) | (q // 2 - k // 2 <= window_size))
    allowed = allowed[None, None] & valid[:, None, None, :]
    if document_ids is not None:
        allowed = allowed & (document_ids[:, None, :, None] == document_ids[:, None, None, :])
    return allowed


class SDPAAttention(CausalSelfAttention):
    def forward(self, x, freqs_cis, attn_block_mask=None, past_key_values=None):
        if past_key_values is not None:
            raise NotImplementedError("SPS cramming adapter uses full forwards, without KV caching")
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head)
        k = k.view(b, t, self.n_head, c // self.n_head)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v,
            attn_mask=attn_block_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y)), None


class SPSCausalLM(nn.Module):
    """HF-shaped output and embedding API; native state_dict names are preserved."""

    def __init__(self, model_args, *, sps=False):
        super().__init__()
        self.config = PretrainedConfig(**model_args, is_encoder_decoder=False, use_cache=False)
        self.config.sps = sps
        if sps and int(self.config.window_size) < 0:
            raise ValueError("SPS window_size must be nonnegative")
        cfg = self.config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.hidden_size),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                output_norm=RMSNorm(cfg),
            )
        )
        for block in self.transformer.h:
            block.attn = SDPAAttention(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(cfg.hidden_size // cfg.n_head, cfg.block_size + 1), persistent=False
        )

    def _apply(self, fn, recurse=True):
        # Module.to(dtype=...) otherwise destroys the imaginary part of RoPE.
        freqs = self._buffers.pop("freqs_cis")
        try:
            return super()._apply(fn, recurse=recurse)
        finally:
            self.register_buffer("freqs_cis", freqs.to(device=self.device), persistent=False)

    @property
    def device(self):
        return self.transformer.wte.weight.device

    @property
    def dtype(self):
        return self.transformer.wte.weight.dtype

    def get_input_embeddings(self):
        return self.transformer.wte

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        output_hidden_states=False,
        document_ids=None,
        use_cache=False,
        past_key_values=None,
    ):
        if use_cache or past_key_values is not None:
            raise NotImplementedError("KV caching is not supported by the SPS cramming adapter")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if attention_mask is None:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            if document_ids is None:
                eos = input_ids.eq(self.config.eos_token_id) & attention_mask.bool()
                document_ids = F.pad(eos[:, :-1].long().cumsum(1), (1, 0))
        b, t, _ = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones((b, t), device=inputs_embeds.device, dtype=torch.bool)
        if position_ids is None:
            position_ids = (attention_mask.long().cumsum(1) - 1).clamp_min(0)
        if position_ids.min() < 0:
            raise ValueError("Negative RoPE positions are unsupported")
        needed = int(position_ids.max()) + 1
        if needed > self.freqs_cis.shape[0]:
            self.freqs_cis = precompute_freqs_cis(self.config.hidden_size // self.config.n_head, needed).to(self.device)
        mask = attention_visibility(
            attention_mask, sps=self.config.sps, window_size=getattr(self.config, "window_size", 0), document_ids=document_ids
        )
        if self.config.sps:
            predicts = self.get_input_embeddings().weight[self.config.predict_token_id].view(1, 1, -1).expand(b, t, -1)
            inputs_embeds = torch.stack((inputs_embeds, predicts), dim=2).flatten(1, 2)
            position_ids = position_ids.repeat_interleave(2, dim=1)
        freqs = self.freqs_cis[position_ids]
        x = self.transformer.drop(inputs_embeds)
        states = []

        def readout(hidden):
            return hidden[:, 1::2] if self.config.sps else hidden

        for block in self.transformer.h:
            if output_hidden_states:
                states.append(readout(x))
            x, _ = block(x, freqs, attn_block_mask=mask)
        x = readout(self.transformer.output_norm(x))
        if output_hidden_states:
            states.append(x)
        return CausalLMOutputWithPast(logits=self.lm_head(x), hidden_states=tuple(states) if output_hidden_states else None)


def load_sps_checkpoint(path, *, torch_dtype=torch.float32):
    """Strict native load; select model family from checkpoint metadata, never filename."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False, mmap=True)
    target = checkpoint["config"]["model"]["_target_"]
    if target not in {"modeling.models.sps.SPSModel", "modeling.models.full_attention_model.Model"}:
        raise ValueError(f"Unsupported SPS model target: {target}")
    model = SPSCausalLM(checkpoint["model_args"], sps=target == "modeling.models.sps.SPSModel")
    state = {key.removeprefix("_orig_mod."): value for key, value in checkpoint["model"].items()}
    # assign preserves checkpoint tensors and avoids an extra copy of optimizer storage.
    model.load_state_dict(state, strict=True, assign=True)
    model.transformer.wte.weight = model.lm_head.weight
    if torch_dtype == "auto":
        torch_dtype = state["lm_head.weight"].dtype
    model.to(dtype=torch_dtype).eval().requires_grad_(False)
    return model


def load_sps_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Training used ordinary GPT-2 text encoding. Predict slots are inserted
    # by the model and must never change how literal text is tokenized.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer
