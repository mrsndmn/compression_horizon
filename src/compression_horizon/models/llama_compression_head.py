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


class QFormerLayer(nn.Module):
    """One cross-attention block: N queries attend to L encoder hidden states, then FFN.

    Uses a manual scaled-dot-product attention path (separate Q/K/V/out linears) instead of
    nn.MultiheadAttention, which is more bf16-stable and gives us explicit control over the
    additive mask. nn.MultiheadAttention's internal flash-attn backend can yield NaNs at
    init with bf16 + key_padding_mask under some setups.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm_q1 = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm_q2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q: [B, N, H], kv: [B, L, H], key_padding_mask: [B, L] True at MASKED positions.
        bsz, n_q, hidden = q.shape
        _, n_kv, _ = kv.shape
        q_n = self.norm_q1(q)
        kv_n = self.norm_kv(kv)
        # Q, K, V projections -> [B, H, head_dim] per head -> [B, num_heads, N, head_dim].
        qh = self.q_proj(q_n).reshape(bsz, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        kh = self.k_proj(kv_n).reshape(bsz, n_kv, self.num_heads, self.head_dim).transpose(1, 2)
        vh = self.v_proj(kv_n).reshape(bsz, n_kv, self.num_heads, self.head_dim).transpose(1, 2)
        # Additive attention mask: 0 where allowed, -inf (in qh dtype) where masked.
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] -> [B, 1, 1, L] for broadcasting against [B, H, N, L].
            mask = key_padding_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(1)
            attn_mask = torch.zeros_like(mask, dtype=qh.dtype).masked_fill(mask, float("-inf"))
        attn = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask, dropout_p=0.0)
        # [B, num_heads, N, head_dim] -> [B, N, H].
        attn = attn.transpose(1, 2).reshape(bsz, n_q, hidden)
        attn = self.attn_dropout(self.out_proj(attn))
        q = q + attn
        q_n = self.norm_q2(q)
        q = q + self.ffn(q_n)
        return q


class QFormerCompressionHead(nn.Module):
    """Q-Former-style compression head: N learnable queries cross-attend to the prefix
    hidden states and emit N compression tokens of shape [B, N, H].

    With `query_proj_factor > 1`, the learnable query parameter is kept in a higher
    dimension `factor * H` and projected down to `H` via a linear layer before being
    fed into the cross-attention layers. Same final dimensionality, more degrees of
    freedom for the optimizer — analogous to the wide-init / low-dim-projection trick
    used in progressive cramming.
    """

    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 4,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
        query_proj_factor: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = int(num_queries)
        self.query_proj_factor = max(1, int(query_proj_factor))
        self.query_dim = self.hidden_size * self.query_proj_factor
        self.queries = nn.Parameter(torch.zeros(self.num_queries, self.query_dim))
        if self.query_proj_factor > 1:
            # No bias: keeps initialisation symmetric around zero, the same property
            # we get from the bias-free identity-like baseline at factor=1.
            self.query_proj = nn.Linear(self.query_dim, self.hidden_size, bias=False)
        else:
            self.query_proj = None
        self.layers = nn.ModuleList(
            [QFormerLayer(hidden_size, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        if self.query_proj is not None:
            # 1/sqrt(factor) keeps the post-projection magnitude on the same scale as
            # the baseline factor=1 queries (std≈0.02), so optimisation dynamics start
            # comparable across factors.
            std = 0.02 / (self.query_proj_factor**0.5)
            nn.init.normal_(self.query_proj.weight, mean=0.0, std=std)

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hidden_states: [B, L, H]; key_padding_mask: [B, L] (True = mask).
        bsz = hidden_states.size(0)
        q = self.queries.to(dtype=hidden_states.dtype)  # [N, query_dim]
        if self.query_proj is not None:
            q = self.query_proj(q)  # [N, H]
        q = q.unsqueeze(0).expand(bsz, -1, -1)  # [B, N, H]
        for layer in self.layers:
            q = layer(q, hidden_states, key_padding_mask)
        return self.final_norm(q)


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
        # `config.compression_head_kind` / `config.compression_head_num_queries` round-trip
        # through save_pretrained / from_pretrained so the right head is rebuilt at load time.
        kind = getattr(config, "compression_head_kind", "mlp")
        n_queries = int(getattr(config, "compression_head_num_queries", 1))
        if kind == "qformer":
            num_heads = int(getattr(config, "compression_head_num_heads", 8))
            num_layers = int(getattr(config, "compression_head_num_layers", 1))
            query_proj_factor = int(getattr(config, "compression_head_query_proj_factor", 1) or 1)
            self.compression_head = QFormerCompressionHead(
                hidden_size=hidden,
                num_queries=n_queries,
                num_heads=num_heads,
                num_layers=num_layers,
                query_proj_factor=query_proj_factor,
            )
        else:
            # Legacy MLP head: LayerNorm on the output keeps the compression-token embedding
            # in a bounded range matching the scale the base model's attention layers expect.
            self.compression_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
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
        elif isinstance(module, nn.LayerNorm):
            # Same bf16 phantom-init issue: re-init in-place.
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, QFormerCompressionHead):
            # The `queries` parameter is a bare nn.Parameter, not part of a Linear/LayerNorm,
            # so it would otherwise stay as uninit bf16 memory after from_pretrained.
            nn.init.normal_(module.queries, mean=0.0, std=0.02)
            if module.query_proj is not None:
                std = 0.02 / (module.query_proj_factor**0.5)
                nn.init.normal_(module.query_proj.weight, mean=0.0, std=std)
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

    def swap_to_qformer_head(self, num_queries: int = 4, num_heads: int = 8, num_layers: int = 1) -> None:
        """Replace the default MLP compression_head with a Q-Former head producing
        `num_queries` compression tokens per sample. Call AFTER from_pretrained so the
        legacy MLP weights from the checkpoint (if any) are discarded and the new module
        is freshly initialised in the model's dtype/device.
        """
        ref_param = next(self.compression_head.parameters(), None)
        dtype = ref_param.dtype if ref_param is not None else torch.float32
        device = ref_param.device if ref_param is not None else torch.device("cpu")
        new_head = QFormerCompressionHead(
            hidden_size=self.config.hidden_size,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
        ).to(dtype=dtype, device=device)
        self.compression_head = new_head

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
            if isinstance(self.compression_head, QFormerCompressionHead):
                # Q-Former path: queries cross-attend to all prefix positions; we mask
                # everything past prefix_lengths so the queries only see valid tokens.
                pl = prefix_lengths.to(device=device).view(-1).to(torch.long).clamp_min(1)
                pos = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]
                valid = pos < pl.unsqueeze(1)  # [B, L]
                key_padding_mask = ~valid  # True = mask
                compression_embeds = self.compression_head(hidden_states, key_padding_mask=key_padding_mask)
                # Shape: [B, num_queries, H]
            else:
                # Legacy MLP path: gather hidden at idx = prefix_length-1, run pointwise MLP.
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
