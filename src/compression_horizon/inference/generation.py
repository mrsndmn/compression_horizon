from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


@torch.no_grad()
def generate_from_compression(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
    compressed_embeddings: torch.Tensor,  # [1, mem, hidden]
    max_new_tokens: int,
    num_return_sequences: int = 1,
) -> list[str]:
    """Generates a sequence using only compressed embeddings."""
    # Cast to the same device
    device = compressed_embeddings.device
    if model.device != device:
        model = model.to(device)
    model.eval()

    # Add pad_token to a tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    eos_token_id = tokenizer.eos_token_id

    # Prepare batch of prefixes
    if num_return_sequences > 1:
        compressed_embeddings = compressed_embeddings.expand(num_return_sequences, -1, -1)  # [batch, mem, hidden]
    batch_size, num_compression_tokens, hidden_size = compressed_embeddings.shape

    # Container for generated token ids
    generated_token_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    # Model's input embeddings layer
    input_embeddings = model.get_input_embeddings()

    for _ in range(max_new_tokens):
        # Embeddings
        if generated_token_ids.size(1) == 0:
            generated_embeddings = torch.empty(batch_size, 0, hidden_size, device=device)  # [batch, 0, hidden]
        else:
            generated_embeddings = input_embeddings(generated_token_ids)  # [batch, sequence, hidden]
        united_token_embeddings = torch.cat(
            [compressed_embeddings, generated_embeddings], dim=1
        )  # [batch, mem + sequence, hidden]

        # Attention mask
        compression_attention_mask = torch.ones(
            (batch_size, num_compression_tokens), dtype=torch.long, device=device
        )  # [batch, mem]
        attention_mask = torch.ones(
            (batch_size, generated_embeddings.size(1)), dtype=torch.long, device=device
        )  # [batch, sequence]
        united_attention_mask = torch.cat((compression_attention_mask, attention_mask), dim=1)  # [batch, mem + sequence]

        outputs = model(inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask)
        logits = outputs.logits[:, -1, :]  # [batch, vocabulary]
        next_token_ids = torch.argmax(logits, dim=-1)  # [batch]

        # If a sequence already reached EOS token leave EOS to the end
        if eos_token_id is not None:
            if generated_token_ids.size(1) > 0:
                reached_eos = generated_token_ids[:, -1].eq(eos_token_id)
                next_token_ids = torch.where(reached_eos, torch.full_like(next_token_ids, eos_token_id), next_token_ids)

        generated_token_ids = torch.cat([generated_token_ids, next_token_ids.unsqueeze(-1)], dim=-1)

        # Stop early if all sequences just produced eos and had eos previously
        if eos_token_id is not None and torch.all(next_token_ids.eq(eos_token_id)):
            break

    texts = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
    return texts
