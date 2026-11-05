"""Native Llama/Qwen3 backbones with isolated zero-window SPS prediction slots."""

from types import SimpleNamespace

import torch
from torch import nn


class PretrainedSPS(nn.Module):
    def __init__(self, backbone, *, sps=True):
        super().__init__()
        if backbone.config.model_type not in {"llama", "qwen3"}:
            raise ValueError("Only audited Llama and Qwen3 architectures are supported")
        if backbone.config.model_type == "qwen3" and any(kind != "full_attention" for kind in backbone.config.layer_types):
            raise ValueError("Only full-attention Qwen3 is supported")
        self.backbone = backbone
        self.config = backbone.config
        self.sps = sps
        # A deterministic vocabulary mean avoids changing the tokenizer/head.
        self.prediction = nn.Parameter(backbone.get_input_embeddings().weight.detach().float().mean(0))

    def set_stage(self, embedding_only):
        self.backbone.requires_grad_(not embedding_only)
        self.prediction.requires_grad_(self.sps)

    def forward(self, input_ids, **kwargs):
        if kwargs.get("past_key_values") is not None:
            raise ValueError("Pilot uses full-prefix scoring; cached SPS is unsupported")
        if not self.sps:
            return self.backbone(input_ids=input_ids, use_cache=False)
        b, length = input_ids.shape
        real = self.backbone.get_input_embeddings()(input_ids)
        pred = self.prediction.to(real.dtype).view(1, 1, -1).expand(b, length, -1)
        states = torch.stack((real, pred), dim=2).flatten(1, 2)
        indices = torch.arange(2 * length, device=real.device)
        q, k = indices[:, None], indices[None, :]
        # Both real and predict queries read causal real keys. Only predict
        # queries additionally read themselves; no other prediction key is visible.
        visible = ((k % 2 == 0) & (k // 2 <= q // 2)) | (q == k)
        mask = torch.zeros((2 * length, 2 * length), dtype=real.dtype, device=real.device)
        mask.masked_fill_(~visible, torch.finfo(real.dtype).min)
        # Real positions stay native; prediction for token i+1 occupies i+1.
        positions = (indices // 2 + indices % 2)[None]
        hidden = self.backbone.model(
            inputs_embeds=states, attention_mask=mask[None, None], position_ids=positions, use_cache=False
        ).last_hidden_state
        return SimpleNamespace(logits=self.backbone.lm_head(hidden[:, 1::2]))
