from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaPreTrainedModel,
)


@dataclass
class CausalLMOutputWithPastAndCompression(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    compression_embeds: Optional[torch.FloatTensor] = None
    compression_embeds_all: Optional[torch.FloatTensor] = None


class LlamaForCausalLMCompressionHead(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        hidden = config.hidden_size
        self.compression_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        # transformers>=5.8 `LlamaPreTrainedModel._init_weights` calls
        # `init.normal_(module.weight.float(), ...)`, which fills a fp32 *copy* and leaves
        # the real bf16/fp16 weight uninitialized (random memory → zeros or NaN). Base-model
        # layers escape this because their weights get overwritten by the checkpoint, but our
        # compression_head is missing from the checkpoint and triggers a second `_init_weights`
        # pass after weight loading. Override here with an in-place init on the actual tensor.
        if isinstance(module, nn.Linear):
            std = getattr(self.config, "initializer_range", 0.02) or 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            super()._init_weights(module)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _select_compression_embeds(
        self,
        *,
        compression_embeds_all: torch.Tensor,
        prefix_lengths: torch.LongTensor,
    ) -> torch.Tensor:
        # compression_embeds_all: [B, T, H]
        # prefix_lengths: [B] (number of tokens compressed in the original sequence)
        # prefix_lengths must be >= 1 (enforced by caller)
        bsz, seq_len, hidden = compression_embeds_all.shape
        device = compression_embeds_all.device
        prefix_lengths = prefix_lengths.to(device=device).view(-1).clamp_min(1)

        selected = torch.empty((bsz, hidden), device=device, dtype=compression_embeds_all.dtype)
        for b in range(bsz):
            p = int(prefix_lengths[b].item())
            # p is guaranteed to be >= 1, so we can safely use p - 1 as index
            idx = min(p - 1, seq_len - 1)
            selected[b] = compression_embeds_all[b, idx]
        return selected.unsqueeze(1)  # [B, 1, H]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_embeddings_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        prefix_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPastAndCompression]:
        # Note: `special_embeddings_mask` is a dataset crutch in this repo; unused here but kept for compatibility.
        _ = special_embeddings_mask

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]  # [B, T, H]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        compression_embeds_all = None
        compression_embeds = None
        if prefix_lengths is not None:
            # Compute only the selected compression embedding to reduce memory:
            # pick hidden_state at idx = clamp(prefix_lengths - 1) and run compression_head on [B, H].
            bsz, seq_len, _hidden = hidden_states.shape
            device = hidden_states.device
            idx = prefix_lengths.to(device=device).view(-1).to(torch.long).clamp_min(1) - 1  # [B]
            idx = idx.clamp_max(seq_len - 1)
            selected_hidden = hidden_states[torch.arange(bsz, device=device), idx]  # [B, H]
            compression_embeds = self.compression_head(selected_hidden).unsqueeze(1)  # [B, 1, H]

        if not return_dict:
            output = (logits,) + outputs[1:]
            extra = (compression_embeds, compression_embeds_all)
            return (loss,) + output + extra if loss is not None else output + extra

        return CausalLMOutputWithPastAndCompression(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            compression_embeds=compression_embeds,
            compression_embeds_all=compression_embeds_all,
        )
