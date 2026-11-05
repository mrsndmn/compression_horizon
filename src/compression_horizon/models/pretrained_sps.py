"""Differentiable cramming interface to the audited pretrained AR-to-SPS models.

The reference implementation is vendored at sps_vendor/pretrained.py. Unlike
the native pilot, these models have a separate prediction parameter (no token
ID), and prediction positions are i+1 rather than i.
"""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .sps_vendor.pretrained import PretrainedSPS


class PretrainedSPSCausalLM(PretrainedSPS):
    def __init__(self, backbone, *, sps=True):
        super().__init__(backbone, sps=sps)
        self.config.sps = sps
        self.config.window_size = 0
        self.config.predict_token_id = None
        self.config.use_cache = False

    @property
    def device(self):
        return self.get_input_embeddings().weight.device

    @property
    def dtype(self):
        return self.get_input_embeddings().weight.dtype

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        attention_mask=None,
        predict_embeddings=None,
        position_ids=None,
        output_hidden_states=False,
        use_cache=False,
        past_key_values=None,
    ):
        if use_cache or past_key_values is not None:
            raise ValueError("Pretrained SPS cramming uses full prefixes without KV cache")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids and inputs_embeds")
        real = self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        b, length, hidden = real.shape
        if attention_mask is None:
            # BOS may equal PAD/EOS. Validity must never be inferred from IDs.
            attention_mask = torch.ones((b, length), device=real.device, dtype=torch.long)
        if attention_mask.shape != (b, length):
            raise ValueError("attention_mask must align with ordinary inputs")
        positions = (attention_mask.long().cumsum(-1) - 1).clamp_min(0) if position_ids is None else position_ids
        if positions.shape != (b, length):
            raise ValueError("position_ids must align with ordinary inputs")
        if not self.sps:
            if predict_embeddings is not None:
                raise ValueError("predict_embeddings requires SPS")
            return self.backbone(
                inputs_embeds=real,
                attention_mask=attention_mask,
                position_ids=positions,
                output_hidden_states=output_hidden_states,
                use_cache=False,
            )
        if predict_embeddings is None:
            pred = self.prediction.to(real.dtype).view(1, 1, hidden).expand(b, length, hidden)
        else:
            if predict_embeddings.shape != (b, 1, hidden):
                raise ValueError("predict_embeddings must be one shared vector per sample")
            pred = predict_embeddings.to(device=real.device, dtype=real.dtype).expand(b, length, hidden)
        states = torch.stack((real, pred), dim=2).flatten(1, 2)
        index = torch.arange(2 * length, device=real.device)
        q, k = index[:, None], index[None, :]
        visible = ((k % 2 == 0) & (k // 2 <= q // 2)) | (q == k)
        visible = visible[None, None] & attention_mask.bool().repeat_interleave(2, 1)[:, None, None, :]
        mask = torch.zeros(visible.shape, dtype=real.dtype, device=real.device)
        mask.masked_fill_(~visible, torch.finfo(real.dtype).min)
        positions = torch.stack((positions, positions + 1), dim=-1).flatten(1, 2)
        out = self.backbone.model(
            inputs_embeds=states,
            attention_mask=mask,
            position_ids=positions,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )
        return CausalLMOutputWithPast(
            logits=self.backbone.lm_head(out.last_hidden_state[:, 1::2]),
            hidden_states=tuple(x[:, 1::2] for x in out.hidden_states) if output_hidden_states else None,
        )


def load_pretrained_sps(source, revision, *, checkpoint=None, source_path=None, torch_dtype=torch.bfloat16):
    """Use pinned source architecture and strictly restore the whole adapted state."""
    location = str(source_path or source)
    config = AutoConfig.from_pretrained(location, revision=revision)
    config._attn_implementation = "sdpa"
    if checkpoint is None:
        backbone = AutoModelForCausalLM.from_pretrained(
            location, revision=revision, dtype=torch_dtype, attn_implementation="sdpa"
        )
        model = PretrainedSPSCausalLM(backbone, sps=False)
    else:
        saved = torch.load(Path(checkpoint), map_location="cpu", mmap=True, weights_only=False)
        if saved["source"] != source or saved["revision"] != revision:
            raise ValueError("Pretrained SPS checkpoint source/revision mismatch")
        backbone = AutoModelForCausalLM.from_config(config, dtype=torch_dtype)
        model = PretrainedSPSCausalLM(backbone, sps=True)
        model.load_state_dict(saved["model"], strict=True, assign=True)
    return model.to(dtype=torch_dtype).eval().requires_grad_(False)


def load_pretrained_tokenizer(source_path, revision):
    tokenizer = AutoTokenizer.from_pretrained(str(source_path), revision=revision)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def existing_bos(model, tokenizer):
    token_id = model.config.bos_token_id
    if not isinstance(token_id, int) or not 0 <= token_id < len(tokenizer):
        raise ValueError("A verified existing source-config BOS is required")
    if tokenizer.bos_token_id is not None and tokenizer.bos_token_id != token_id:
        raise ValueError("Model/tokenizer BOS mismatch")
    return token_id
