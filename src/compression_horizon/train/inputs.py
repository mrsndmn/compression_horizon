import torch


def build_compression_attention_mask(
    batch_size: int,
    num_compression_tokens: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compression tokens attention mask."""
    compression_attention_mask = torch.ones(
        (batch_size, num_compression_tokens),
        dtype=dtype,
        device=device,
    )
    return compression_attention_mask


def build_united_input(
    compression_token_embeddings: torch.Tensor,  # [batch, compression, hidden]
    compression_attention_mask: torch.Tensor,  # [batch, compression]
    token_embeddings: torch.Tensor,  # [batch, sequence, hidden]
    attention_mask: torch.Tensor,  # [batch, sequence]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather up compression and sequence token embeddings and attention masks."""
    compression_token_embeddings = compression_token_embeddings.to(token_embeddings.device).to(token_embeddings.dtype)
    united_token_embeddings = torch.cat(
        (compression_token_embeddings, token_embeddings),
        dim=1,
    )
    united_attention_mask = torch.cat(
        (compression_attention_mask, attention_mask),
        dim=1,
    )
    return united_token_embeddings, united_attention_mask
