"""Encoder-decoder helpers shared by the downstream compress-evaluate scripts.

The HellaSwag and ARC evaluators are structurally identical: cram each benchmark context into a
memory embedding, then score the candidate continuations conditioned on it. Both need the same two
encoder-decoder pieces, so they live here rather than being copied into each script.

The single idea behind both: on a seq2seq model the memory sits on the encoder side and occupies
**no** decoder position, so every offset that counts memory positions in the decoder-only code
becomes ``EncoderDecoderCrammingAdapter.LOGITS_OFFSET`` (1, the decoder start token).
"""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW
from transformers import get_scheduler

from compression_horizon.analysis.perplexity import estimate_token_perplexity, estimate_token_perplexity_full_labels
from compression_horizon.train.encoder_decoder import EncoderDecoderCrammingAdapter
from compression_horizon.train.loss import (
    compute_hybrid_cross_entropy_and_alignment_loss,
    token_argmax_match_rate_with_prefix,
)
from compression_horizon.utils.launch import freeze_model_parameters


def compress_prefixes_batch_encoder_decoder(
    *,
    adapter: EncoderDecoderCrammingAdapter,
    tokenizer,
    texts: list[str],
    num_compression_tokens: int,
    max_steps: int,
    learning_rate: float,
    device: torch.device,
    add_special_tokens: bool = True,
) -> list[dict[str, torch.Tensor | float]]:
    """Encoder-decoder counterpart of the evaluators' ``compress_prefixes_batch``.

    The memory goes on the encoder side and the decoder is teacher-forced on ``[start] [prefix]``.
    Padding needs no special handling: the tokenizer already padded ``input_ids`` and the decoder
    attention mask masks it out, so there is no per-sample embedding concatenation to redo.
    """
    batch_size = len(texts)
    if batch_size == 0:
        return []

    model = adapter.model.to(device)
    model.eval()
    freeze_model_parameters(model)

    encoded = tokenizer(texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)  # [batch, sequence]
    attention_mask = encoded["attention_mask"].to(device)  # [batch, sequence]

    compression_token_embeddings = torch.nn.Parameter(
        torch.randn(
            [batch_size, num_compression_tokens, adapter.memory_hidden_size],
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )  # [batch, compression, hidden]
    memory_mask = torch.ones((batch_size, num_compression_tokens), dtype=attention_mask.dtype, device=device)

    optimizer = AdamW([compression_token_embeddings], lr=learning_rate)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=max_steps)

    offset = adapter.LOGITS_OFFSET
    for _ in range(max_steps):
        optimizer.zero_grad()
        outputs = adapter.forward(
            memory_embeds=compression_token_embeddings,
            memory_mask=memory_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        total_loss = 0.0
        for i in range(batch_size):
            sequence_len = int(attention_mask[i].sum().item())
            sample_loss, _ = compute_hybrid_cross_entropy_and_alignment_loss(
                logits=outputs.logits[i : i + 1, : offset + sequence_len],
                input_ids=input_ids[i : i + 1, :sequence_len],
                attention_mask=attention_mask[i : i + 1, :sequence_len],
                num_compression_tokens=offset,
                num_alignment_layers=0,
                inverted_alignment=False,
                loss_type="cross_entropy",
                hybrid_alpha=None,
            )
            total_loss = total_loss + sample_loss
        (total_loss / batch_size).backward()
        optimizer.step()
        lr_scheduler.step()

    with torch.no_grad():
        outputs = adapter.forward(
            memory_embeds=compression_token_embeddings,
            memory_mask=memory_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        convergences = token_argmax_match_rate_with_prefix(outputs.logits, input_ids, attention_mask, offset).tolist()
    print("Batch convergences:", convergences)
    return [
        {
            "compression_embedding": compression_token_embeddings[i].detach(),
            "convergence": convergences[i],
        }
        for i in range(batch_size)
    ]


@torch.no_grad()
def compute_ppl_compression_encoder_decoder(
    *,
    adapter: EncoderDecoderCrammingAdapter,
    compression_token_embeddings: torch.Tensor,  # [compression, hidden]
    input_ids: torch.Tensor,  # [batch, sequence]
    attention_mask: torch.Tensor,  # [batch, sequence]
    context_len: int,
) -> tuple[list[float], list[float], list[float]]:
    """Score candidate continuations with the memory conditioning the decoder via cross-attention."""
    num_texts = input_ids.size(0)
    num_compression_tokens = compression_token_embeddings.shape[0]
    memory = compression_token_embeddings.unsqueeze(0).expand(num_texts, -1, -1)
    memory_mask = torch.ones((num_texts, num_compression_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
    outputs = adapter.forward(
        memory_embeds=memory,
        memory_mask=memory_mask,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return extract_compression_ppls(
        logits=outputs.logits,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_texts=num_texts,
        context_len=context_len,
        logits_offset=adapter.LOGITS_OFFSET,
    )


def extract_compression_ppls(
    *,
    logits: torch.Tensor,  # [batch, logits_offset + sequence, vocabulary]
    input_ids: torch.Tensor,  # [batch, sequence]
    attention_mask: torch.Tensor,  # [batch, sequence]
    num_texts: int,
    context_len: int,
    logits_offset: int,
) -> tuple[list[float], list[float], list[float]]:
    """Per-sample (full, edge, endings) perplexities from memory-conditioned logits.

    ``logits_offset`` is how many leading positions precede the scored sequence: the number of
    memory tokens for a decoder-only model, 1 (the decoder start token) for an encoder-decoder.
    The two layouts are otherwise identical, which is why both architectures share this extraction.
    """
    ppls: list[float] = []
    edge_ppls: list[float] = []
    endings_ppls: list[float] = []
    for i in range(num_texts):
        sequence_len = int(attention_mask[i].sum().item())
        # Perplexity over the whole sequence.
        sample_logits = logits[i : i + 1, logits_offset : logits_offset + sequence_len]  # [1, sequence, vocabulary]
        sample_input_ids = input_ids[i : i + 1, :sequence_len]  # [1, sequence]
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids)
        if math.isnan(ppl):
            ppl = float("inf")
        ppls.append(ppl)
        # Perplexity including the leading position (the memory / start-token edge).
        sample_logits = logits[i : i + 1, logits_offset - 1 : logits_offset + sequence_len]  # [1, sequence + 1, vocabulary]
        sample_input_ids = input_ids[i : i + 1, :sequence_len]  # [1, sequence]
        edge_ppl = estimate_token_perplexity_full_labels(sample_logits, sample_input_ids)
        if math.isnan(edge_ppl):
            edge_ppl = float("inf")
        edge_ppls.append(edge_ppl)
        # Perplexity over the endings only.
        sample_logits = logits[
            i : i + 1, logits_offset + max(0, context_len - 1) : logits_offset + sequence_len, :
        ]  # [1, ending, vocabulary]
        sample_input_ids = input_ids[i : i + 1, max(0, context_len - 1) : sequence_len]  # [1, ending]
        ending_ppl = estimate_token_perplexity(sample_logits, sample_input_ids)
        if math.isnan(ending_ppl):
            ending_ppl = float("inf")
        endings_ppls.append(ending_ppl)
    return ppls, edge_ppls, endings_ppls
