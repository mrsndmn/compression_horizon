from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast


def mask_native_generation_logits(model, tokenizer, logits: torch.Tensor) -> torch.Tensor:
    """Exclude native checkpoint control/padded IDs from generated text only.

    Native Standard/SPS adapters expose config.sps (False/True respectively).
    Their padded model vocabulary exceeds the ordinary GPT-2 tokenizer vocabulary.
    Teacher-forced logits and losses retain the original model distribution.
    """
    config = getattr(model, "config", None)
    if not hasattr(config, "sps"):
        return logits
    filtered = logits.clone()
    filtered[..., len(tokenizer) :] = -torch.inf
    for field in ("predict_token_id", "pad_token_id"):
        token_id = getattr(config, field, None)
        if token_id is not None and token_id != tokenizer.eos_token_id and 0 <= token_id < logits.shape[-1]:
            filtered[..., token_id] = -torch.inf
    return filtered


@torch.no_grad()
def generate_from_compression(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
    compression_token_embeddings: torch.Tensor,  # [1, compression, hidden]
    max_new_tokens: int,
    num_return_sequences: int = 1,
    add_noise=False,
    random_position_ids: bool = False,
    return_generated_ids: bool = False,
    predict_embeddings: torch.Tensor | None = None,
    use_cache: bool = False,
) -> list[str] | tuple[list[str], torch.Tensor]:
    """Generates a sequence starting from compressed embeddings."""
    # Cast to the same device
    device = compression_token_embeddings.device
    if model.device != device:
        model = model.to(device)
    model.eval()

    # Add pad_token to a tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    eos_token_id = tokenizer.eos_token_id

    # Prepare batch of prefixes
    if num_return_sequences > 1:
        compression_token_embeddings = compression_token_embeddings.expand(
            num_return_sequences, -1, -1
        )  # [batch, sequence, hidden]

    if add_noise:
        noise = torch.randn_like(compression_token_embeddings) * 0.01
        compression_token_embeddings += noise

    if predict_embeddings is not None and num_return_sequences > 1:
        predict_embeddings = predict_embeddings.expand(num_return_sequences, -1, -1)
    forward_kwargs = {} if predict_embeddings is None else {"predict_embeddings": predict_embeddings}
    batch_size, num_compression_tokens, hidden_size = compression_token_embeddings.shape

    # Container for generated token ids
    generated_token_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)  # [batch, 0]
    # Model's input embeddings layer
    input_embeddings = model.get_input_embeddings()
    torch_dtype = input_embeddings.weight.dtype

    if use_cache and random_position_ids:
        raise ValueError("Cached generation requires stable position IDs")
    past = None
    for step in range(max_new_tokens):
        if step == 0:
            inputs = compression_token_embeddings
        elif use_cache:
            inputs = input_embeddings(generated_token_ids[:, -1:])
        else:
            inputs = torch.cat(
                (compression_token_embeddings, input_embeddings(generated_token_ids)),
                dim=1,
            )
        attention_mask = torch.ones((batch_size, num_compression_tokens + step), dtype=torch.long, device=device)
        if random_position_ids:
            forward_kwargs["position_ids"] = torch.randperm(inputs.size(1), device=device).expand(batch_size, -1)
        if use_cache:
            forward_kwargs.update(use_cache=True, past_key_values=past)
        outputs = model(
            inputs_embeds=inputs.to(torch_dtype),
            attention_mask=attention_mask,
            **forward_kwargs,
        )
        if use_cache:
            past = outputs.past_key_values
            if past is None:
                raise ValueError("Decoder returned no requested KV cache")
        logits = outputs.logits[:, -1, :]  # [batch, vocabulary]
        logits = mask_native_generation_logits(model, tokenizer, logits)
        next_token_ids = torch.argmax(logits, dim=-1)  # [batch]

        # If a sequence already reached EOS token leave EOS to the end
        if eos_token_id is not None:
            if generated_token_ids.size(1) > 0:
                reached_eos = generated_token_ids[:, -1].eq(eos_token_id)
                next_token_ids = torch.where(
                    reached_eos,
                    torch.full_like(next_token_ids, eos_token_id),
                    next_token_ids,
                )

        generated_token_ids = torch.cat((generated_token_ids, next_token_ids.unsqueeze(-1)), dim=-1)  # [batch, sequence]

        # Stop early if all sequences just produced eos and had eos previously
        if eos_token_id is not None and torch.all(next_token_ids.eq(eos_token_id)):
            break

    texts = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
    return texts, generated_token_ids if return_generated_ids else texts


@torch.no_grad()
def calculate_logits(
    model: PreTrainedModel,
    compressed_embeddings: torch.Tensor,  # [1, mem, hidden]
    sequence_embeddings: torch.Tensor,  # [1, sequence, hidden]
    attention_mask: torch.Tensor,  # [1, sequence]
) -> torch.Tensor:
    """Calculate logits for a sequence."""
    # Cast to the same device
    device = compressed_embeddings.device
    if model.device != device:
        model = model.to(device)
    model.eval()

    united_embeddings = torch.cat(
        (compressed_embeddings, sequence_embeddings),
        dim=1,
    )  # [1, mem + sequence, hidden]
    united_attention_mask = torch.cat(
        (
            torch.ones(
                compressed_embeddings.size(0),
                compressed_embeddings.size(1),
                dtype=torch.long,
                device=device,
            ),
            attention_mask,
        ),
        dim=1,
    )  # [1, mem + sequence]
    outputs = model(
        inputs_embeds=united_embeddings,
        attention_mask=united_attention_mask,
    )
    logits = outputs.logits  # [batch, mem + sequence, vocabulary]
    return logits


@torch.no_grad()
def calculate_outputs(
    model: PreTrainedModel,
    compressed_embeddings: torch.Tensor,
    sequence_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> CausalLMOutputWithPast:
    """Calculate outputs for a sequence."""
    # Cast to the same device
    device = compressed_embeddings.device
    if model.device != device:
        model = model.to(device)

    # Required to enable output_attentions
    model.set_attn_implementation("eager")
    model.eval()

    united_embeddings = torch.cat(
        (compressed_embeddings, sequence_embeddings),
        dim=1,
    )
    united_attention_mask = torch.cat(
        (
            torch.ones(
                compressed_embeddings.size(0),
                compressed_embeddings.size(1),
                dtype=torch.long,
                device=device,
            ),
            attention_mask,
        ),
        dim=1,
    )
    outputs = model(
        inputs_embeds=united_embeddings,
        attention_mask=united_attention_mask,
        output_attentions=True,
        output_hidden_states=True,
    )
    return outputs
