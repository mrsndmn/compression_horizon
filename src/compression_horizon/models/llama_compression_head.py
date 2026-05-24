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


class _QFormerBlock(nn.Module):
    """One pre-norm block: query self-attention, query→prefix cross-attention, FFN."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.sa_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ca_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, queries: torch.Tensor, prefix: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # queries: [B, Q, H]; prefix: [B, T, H]; key_padding_mask: [B, T] (True = ignore).
        q = self.sa_norm(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        q = self.ca_norm(queries)
        queries = queries + self.cross_attn(q, prefix, prefix, key_padding_mask=key_padding_mask, need_weights=False)[0]
        q = self.ffn_norm(queries)
        queries = queries + self.ffn(q)
        return queries


class CompressionQFormer(nn.Module):
    """Learnable queries that cross-attend to the prefix hidden states, producing [B, num_queries, H]."""

    def __init__(self, hidden_size: int, num_queries: int, num_layers: int, num_heads: int):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, hidden_size) * 0.02)
        self.layers = nn.ModuleList([_QFormerBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, T, H]; key_padding_mask: [B, T] bool, True marks positions to ignore.
        bsz = hidden_states.shape[0]
        queries = self.query.to(dtype=hidden_states.dtype).unsqueeze(0).expand(bsz, -1, -1)
        for layer in self.layers:
            queries = layer(queries, hidden_states, key_padding_mask)
        return self.norm(queries)  # [B, Q, H]


class LlamaForCausalLMCompressionHead(LlamaPreTrainedModel, GenerationMixin):
    # transformers 5.x expects a dict mapping each tied weight to its source (lm_head ↔ input embeddings).
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        hidden = config.hidden_size
        # Head architecture is persisted on the config so save_pretrained/from_pretrained round-trips.
        # Older checkpoints lack these attrs and default to the original MLP head (backward compatible).
        self.compression_head_kind = getattr(config, "compression_head_kind", "mlp")
        self.compression_head_num_queries = int(getattr(config, "compression_head_num_queries", 1))
        if self.compression_head_kind == "qformer":
            self.compression_head = CompressionQFormer(
                hidden_size=hidden,
                num_queries=self.compression_head_num_queries,
                num_layers=int(getattr(config, "compression_head_num_layers", 2)),
                num_heads=int(getattr(config, "compression_head_num_heads", 8)),
            )
        elif self.compression_head_kind == "mlp":
            self.compression_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
        else:
            raise ValueError(f"Unsupported compression_head_kind={self.compression_head_kind!r} (expected 'mlp' or 'qformer').")

        self.post_init()

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
            bsz, seq_len, _hidden = hidden_states.shape
            device = hidden_states.device
            prefix_len = prefix_lengths.to(device=device).view(-1).to(torch.long).clamp_min(1).clamp_max(seq_len)
            if self.compression_head_kind == "qformer":
                # Queries cross-attend to the prefix hidden states [0, prefix_len); mask out the rest.
                positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
                key_padding_mask = positions >= prefix_len.unsqueeze(1)  # [B, T] True = ignore
                compression_embeds = self.compression_head(hidden_states, key_padding_mask)  # [B, Q, H]
            else:
                # MLP head: run on the last prefix hidden state at idx = prefix_len - 1.
                idx = (prefix_len - 1).clamp_max(seq_len - 1)  # [B]
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
