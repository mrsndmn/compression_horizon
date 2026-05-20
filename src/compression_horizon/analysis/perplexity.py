from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


def calculate_distances(compression_embeddings: torch.Tensor, sequence_embeddings: torch.Tensor) -> tuple[float, float, float]:
    # Cosine
    cosine = F.cosine_similarity(compression_embeddings, sequence_embeddings, dim=-1)
    cosine = (1.0 - cosine).mean().item()
    # l2
    l2 = torch.sqrt(torch.sum((sequence_embeddings - compression_embeddings) ** 2, dim=-1)).mean().item()
    # l1
    l1 = torch.sum(torch.abs(sequence_embeddings - compression_embeddings), dim=-1).mean().item()
    return cosine, l2, l1


def estimate_token_perplexity(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Compute perplexity from logits, labels, and attention mask."""
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    nll = -log_probs.gather(dim=-1, index=labels[:, 1:].unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        nll = nll[mask[:, 1:].bool()]
    if nll.numel() == 0:
        return float("nan")
    ppl = torch.exp(nll.mean()).item()
    return float(ppl)


def estimate_token_perplexity_full_labels(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """Compute perplexity from logits, labels, and attention mask."""
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        nll = nll[mask.bool()]
    if nll.numel() == 0:
        return float("nan")
    ppl = torch.exp(nll.mean()).item()
    return float(ppl)


@torch.no_grad()
def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
    compressed_embeddings: torch.Tensor,  # [1, mem, hidden]
    sequence_embeddings: torch.Tensor,  # [1, sequence, hidden]
    attention_mask: torch.Tensor,  # [1, sequence]
    *,
    n: int = 128,
    return_generated_text: str,
) -> tuple[int | float | bool, str] | int | float | bool:
    """Entropy measures the level of uncertainty in the model's output.

    Lower entropy means the model is more certain about its predictions and therefore, the perplexity is lower.
    Perplexity indicates the level of confidence the model has in its prediction—lower perplexity suggests higher
    confidence and better performance in predicting the next word,
    while higher perplexity signals more uncertainty and less reliability.
    """
    # This function measures how confident the model is in its own greedy outputs, not perplexity, and the resulting number is mathematically meaningless for evaluation.
    # Cast to the same device
    device = compressed_embeddings.device
    if model.device != device:
        model = model.to(device)
    model.eval()

    # Add pad_token to a tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    eos_token_id = tokenizer.eos_token_id

    _, num_compression_tokens, _ = compressed_embeddings.shape

    # Container for generated token logits
    generated_token_logits = []
    # Model's input embeddings layer
    input_embeddings = model.get_input_embeddings()
    torch_dtype = input_embeddings.weight.dtype

    for _ in range(n):
        # Embeddings
        united_token_embeddings = torch.cat((compressed_embeddings, sequence_embeddings), dim=1)  # [1, mem + sequence, hidden]
        united_token_embeddings = united_token_embeddings.to(torch_dtype)

        # Attention mask
        compression_attention_mask = torch.ones((1, num_compression_tokens), dtype=torch.long, device=device)  # [1, mem]
        united_attention_mask = torch.cat((compression_attention_mask, attention_mask), dim=1)  # [1, mem + sequence]

        # Model result
        outputs = model(inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask)
        logits = outputs.logits[:, -1, :]  # [1, vocabulary]
        next_token_id = torch.argmax(logits, dim=-1)  # [1]

        # Stop if a sequence has reached EOS token
        generated_token_logits.append(logits.cpu())
        if eos_token_id is not None:
            if next_token_id.item() == eos_token_id:
                break

        # Increment sequence embeddings and attention mask
        next_token_embedding = input_embeddings(next_token_id).unsqueeze(dim=1)  # [1, 1, hidden]
        sequence_embeddings = torch.cat((sequence_embeddings, next_token_embedding), dim=1)  # [1, sequence + 1, hidden]
        attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)), dim=1
        )  # [1, sequence + 1]

    generated_token_logits = torch.cat(generated_token_logits, dim=0)  # [n, vocabulary]
    generated_token_log_probs = F.log_softmax(generated_token_logits, dim=1)  # [n, vocabulary]
    cross_entropy = (
        -1
        * torch.gather(
            generated_token_log_probs,
            1,
            generated_token_log_probs.argmax(dim=1).view(-1, 1),
        ).mean()
    )
    perplexity = torch.exp(cross_entropy).item()

    if return_generated_text:
        generated_text = tokenizer.decode(generated_token_log_probs.argmax(dim=1), skip_special_tokens=True)
        return perplexity, generated_text
    return perplexity


@torch.no_grad()
def calculate_perplexity_logits(
    model: PreTrainedModel,
    compressed_embeddings: torch.Tensor,  # [1, mem, hidden]
    input_ids: torch.Tensor,  # [1, sequence]
    sequence_embeddings: torch.Tensor,  # [1, sequence, hidden]
    attention_mask: torch.Tensor,  # [1, sequence]
) -> float:
    """Entropy measures the level of uncertainty in the model's output.

    Lower entropy means the model is more certain about its predictions and therefore, the perplexity is lower.
    Perplexity indicates the level of confidence the model has in its prediction—lower perplexity suggests higher
    confidence and better performance in predicting the next word,
    while higher perplexity signals more uncertainty and less reliability.
    """
    # Cast to the same device
    device = compressed_embeddings.device
    if model.device != device:
        model = model.to(device)
    model.eval()

    _, num_compression_tokens, _ = compressed_embeddings.shape

    torch_dtype = model.get_input_embeddings().weight.dtype

    # Embeddings
    united_token_embeddings = torch.cat((compressed_embeddings, sequence_embeddings), dim=1)  # [1, mem + sequence, hidden]
    united_token_embeddings = united_token_embeddings.to(torch_dtype)

    # Attention mask
    compression_attention_mask = torch.ones((1, num_compression_tokens), dtype=torch.long, device=device)  # [1, mem]
    united_attention_mask = torch.cat((compression_attention_mask, attention_mask), dim=1)  # [1, mem + sequence]

    # Model result
    outputs = model(inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask)
    logits = outputs.logits[:, num_compression_tokens - 1 : -1, :].squeeze(dim=0)  # [sequence, vocabulary]

    # Perplexity by logits
    log_probs = F.log_softmax(logits, dim=1)  # [sequence, vocabulary]
    cross_entropy = (
        -1
        * torch.gather(
            log_probs,
            1,
            input_ids.view(-1, 1),  # [sequence, 1]
        ).mean()
    )
    perplexity = torch.exp(cross_entropy).item()
    return perplexity
