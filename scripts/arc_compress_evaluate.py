"""Evaluate compression tokens on ARC benchmark.

This script:
1. Downloads ARC dataset (ARC-Easy or ARC-Challenge)
2. Compresses the context (question) for each item
3. Evaluates by computing PPL for all choices and choosing the one with lowest PPL
4. Dumps evaluation results
"""

import argparse
import json
import math
import os
from typing import Optional

import torch
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from compression_horizon.train.loss import compute_hybrid_cross_entropy_and_alignment_loss
from compression_horizon.utils.launch import freeze_model_parameters, get_device, resolve_torch_dtype, set_launch_seed


def count_text_tokens(tokenizer: AutoTokenizer, text: str, add_special_tokens: bool = True) -> int:
    """Count tokens in text using the provided tokenizer."""
    encoded = tokenizer(
        text,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=add_special_tokens,
    )
    return len(encoded["input_ids"])


def count_text_characters(text: str) -> int:
    """Count characters in text."""
    return len(text)


def estimate_token_perplexity(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute perplexity from logits, labels, and attention mask."""
    # logits: [B, T, V], labels: [B, T], mask: [B, T]
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = labels[:, 1:]
    m = mask[:, 1:].bool()
    nll = -log_probs.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
    nll = nll[m]
    if nll.numel() == 0:
        return float("nan")
    ppl = torch.exp(nll.mean()).item()
    return float(ppl)


def compress_prefixes_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    num_compression_tokens: int = 1,
    max_steps: int = 1000,
    learning_rate: float = 1e-2,
    loss_type: str = "cross_entropy",
    hybrid_alpha: Optional[float] = None,
    num_alignment_layers: int = 0,
    inverted_alignment: bool = False,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> list[torch.Tensor]:
    """Compress multiple text prefixes into compression tokens (batched).

    Args:
        model: The language model (frozen)
        tokenizer: Tokenizer
        texts: List of text prefixes to compress
        num_compression_tokens: Number of compression tokens
        max_steps: Maximum optimization steps
        learning_rate: Learning rate for optimization
        device: Device to use

    Returns:
        List of compression token embeddings, each [num_compression_tokens, hidden_size]
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()
    freeze_model_parameters(model)

    batch_size = len(texts)
    if batch_size == 0:
        return []

    loss_type = (loss_type or "cross_entropy").lower()
    use_alignment = hybrid_alpha is not None and loss_type != "cross_entropy"

    # Tokenize all texts with padding
    encoded = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)  # [batch_size, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # [batch_size, seq_len]

    hidden_size = model.config.hidden_size

    # Get token embeddings to determine dtype
    with torch.no_grad():
        token_embeddings = model.model.embed_tokens(input_ids)  # [batch_size, seq_len, hidden]

    # Get dtype from model embeddings
    embedding_dtype = token_embeddings.dtype
    compression_dtype = embedding_dtype

    # Initialize compression tokens for all samples
    compression_tokens = torch.nn.Parameter(
        torch.randn([batch_size, num_compression_tokens, hidden_size], dtype=compression_dtype, device=device) * 0.02
    )

    # Optimizer
    optimizer = AdamW([compression_tokens], lr=learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps,
    )

    model.train()

    # Get sequence lengths for all samples (vectorized)
    # seq_lens = attention_mask.sum(dim=1)  # [batch_size]
    # max_seq_len = seq_lens.max().item()

    # Prepare labels (mask padding tokens)
    labels = input_ids.clone()  # [batch_size, max_seq_len]
    labels[attention_mask == 0] = -100

    target_outputs = None
    if use_alignment:
        with torch.no_grad():
            target_outputs = model(
                inputs_embeds=token_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

    # Training loop
    for step in tqdm(range(max_steps)):
        optimizer.zero_grad()

        # Vectorized concatenation: compression tokens + token embeddings
        # compression_tokens: [batch_size, num_compression_tokens, hidden_size]
        # token_embeddings: [batch_size, max_seq_len, hidden_size]
        batch_embeddings = torch.cat(
            [compression_tokens, token_embeddings], dim=1
        )  # [batch_size, num_compression_tokens + max_seq_len, hidden_size]

        # Create attention mask for compression tokens (all ones) and concatenate
        compression_attention = torch.ones(
            (batch_size, num_compression_tokens), dtype=attention_mask.dtype, device=device
        )  # [batch_size, num_compression_tokens]
        batch_attention = torch.cat(
            [compression_attention, attention_mask], dim=1
        )  # [batch_size, num_compression_tokens + max_seq_len]

        # Forward pass with compression tokens
        compression_outputs = model(
            inputs_embeds=batch_embeddings,
            attention_mask=batch_attention,
            output_hidden_states=use_alignment,
        )

        # Compute loss for entire batch (vectorized)
        if use_alignment:
            assert target_outputs is not None
            # Slice compression hidden states to match target sequence lengths
            # For each sample, we need to take num_compression_tokens + seq_len from compression outputs
            # and seq_len from target outputs
            # Since we can't easily slice per sample in a vectorized way for hidden states,
            # we'll use the full sequences and let the loss function handle masking via attention_mask
            loss, _ = compute_hybrid_cross_entropy_and_alignment_loss(
                logits=compression_outputs.logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_prefix_tokens=num_compression_tokens,
                target_hidden_states=target_outputs.hidden_states,
                compression_hidden_states=compression_outputs.hidden_states,
                num_alignment_layers=num_alignment_layers,
                inverted_alignment=inverted_alignment,
                loss_type=loss_type,
                hybrid_alpha=hybrid_alpha,
            )
        else:
            loss, _ = compute_hybrid_cross_entropy_and_alignment_loss(
                logits=compression_outputs.logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_prefix_tokens=num_compression_tokens,
                num_alignment_layers=num_alignment_layers,
                inverted_alignment=inverted_alignment,
                loss_type=loss_type,
                hybrid_alpha=hybrid_alpha,
            )
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    model.eval()
    # Return compression tokens (remove batch dimension for each)
    return [compression_tokens[i].detach() for i in range(batch_size)]


def compress_prefix(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    num_compression_tokens: int = 1,
    max_steps: int = 1000,
    learning_rate: float = 1e-2,
    loss_type: str = "cross_entropy",
    hybrid_alpha: Optional[float] = None,
    num_alignment_layers: int = 0,
    inverted_alignment: bool = False,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> torch.Tensor:
    """Compress a text prefix into compression tokens.

    Args:
        model: The language model (frozen)
        tokenizer: Tokenizer
        text: Text prefix to compress
        num_compression_tokens: Number of compression tokens
        max_steps: Maximum optimization steps
        learning_rate: Learning rate for optimization
        device: Device to use

    Returns:
        Compression token embeddings [num_compression_tokens, hidden_size]
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()
    freeze_model_parameters(model)

    # Tokenize the text
    encoded = tokenizer(text, truncation=True, padding=False, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)  # [1, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # [1, seq_len]

    batch_size = input_ids.shape[0]
    hidden_size = model.config.hidden_size

    # Get token embeddings to determine dtype
    with torch.no_grad():
        token_embeddings = model.model.embed_tokens(input_ids)  # [1, seq_len, hidden]

    loss_type = (loss_type or "cross_entropy").lower()
    use_alignment = hybrid_alpha is not None and loss_type != "cross_entropy"

    # Get dtype from model embeddings (use bfloat16 as default for computation)
    embedding_dtype = token_embeddings.dtype
    # Use the same dtype as model embeddings for compression tokens
    compression_dtype = embedding_dtype

    # Initialize compression tokens with the correct dtype
    compression_tokens = torch.nn.Parameter(
        torch.randn([batch_size, num_compression_tokens, hidden_size], dtype=compression_dtype, device=device) * 0.02
    )

    # Optimizer
    optimizer = AdamW([compression_tokens], lr=learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps,
    )

    model.train()

    # Training loop
    for step in range(max_steps):
        optimizer.zero_grad()

        # Concatenate compression tokens with input embeddings
        # Both should already have the same dtype (compression_dtype == embedding_dtype)
        united_embeddings = torch.cat([compression_tokens, token_embeddings], dim=1)  # [1, mem+seq, hidden]

        # Create attention mask
        compression_attention = torch.ones((batch_size, num_compression_tokens), dtype=attention_mask.dtype, device=device)
        united_attention_mask = torch.cat([compression_attention, attention_mask], dim=1)  # [1, mem+seq]

        # Forward pass with compression tokens
        compression_outputs = model(
            inputs_embeds=united_embeddings,
            attention_mask=united_attention_mask,
            output_hidden_states=use_alignment,
        )

        if use_alignment:
            with torch.no_grad():
                target_outputs = model(
                    inputs_embeds=token_embeddings,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            loss, _ = compute_hybrid_cross_entropy_and_alignment_loss(
                logits=compression_outputs.logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_prefix_tokens=num_compression_tokens,
                target_hidden_states=target_outputs.hidden_states,
                compression_hidden_states=compression_outputs.hidden_states,
                num_alignment_layers=num_alignment_layers,
                inverted_alignment=inverted_alignment,
                loss_type=loss_type,
                hybrid_alpha=hybrid_alpha,
            )
        else:
            loss, _ = compute_hybrid_cross_entropy_and_alignment_loss(
                logits=compression_outputs.logits,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_prefix_tokens=num_compression_tokens,
                num_alignment_layers=num_alignment_layers,
                inverted_alignment=inverted_alignment,
                loss_type=loss_type,
                hybrid_alpha=hybrid_alpha,
            )

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    model.eval()
    # Return compression tokens (remove batch dimension)
    return compression_tokens.detach().squeeze(0)  # [num_compression_tokens, hidden_size]


@torch.no_grad()
def compute_ppl_baseline_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    contexts: list[str],
    endings: list[str],
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> list[float]:
    """Compute perplexity of context + ending pairs without compression tokens (batched).

    Args:
        model: The language model
        tokenizer: Tokenizer
        contexts: List of context texts
        endings: List of ending texts (same length as contexts)
        device: Device to use

    Returns:
        List of perplexity scores
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    if len(contexts) == 0:
        return []

    # Combine contexts and endings
    full_texts = [ctx + end for ctx, end in zip(contexts, endings)]

    # Tokenize with padding
    encoded = tokenizer(full_texts, truncation=True, padding=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)  # [batch_size, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # [batch_size, seq_len]

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Compute PPL for each sample
    ppls = []
    for i in range(len(contexts)):
        seq_len = attention_mask[i].sum().item()
        sample_logits = outputs.logits[i : i + 1, :seq_len]
        sample_input_ids = input_ids[i : i + 1, :seq_len]
        sample_attention = attention_mask[i : i + 1, :seq_len]
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids, sample_attention)
        if math.isnan(ppl):
            ppl = float("inf")
        ppls.append(ppl)

    return ppls


@torch.no_grad()
def compute_ppl_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: str,
    ending: str,
    device: Optional[torch.device] = None,
) -> float:
    """Compute perplexity of context + ending without compression tokens (baseline).

    Args:
        model: The language model
        tokenizer: Tokenizer
        context: Context text
        ending: Ending text
        device: Device to use

    Returns:
        Perplexity score
    """
    results = compute_ppl_baseline_batch(model, tokenizer, [context], [ending], device)
    return results[0] if results else float("inf")


@torch.no_grad()
def compute_ppl_with_compression_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    compression_tokens_list: list[torch.Tensor],
    contexts: list[str],
    endings: list[str],
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> list[float]:
    """Compute perplexity of context + ending pairs using compression tokens (batched).

    Args:
        model: The language model
        tokenizer: Tokenizer
        compression_tokens_list: List of compression token embeddings, each [num_compression_tokens, hidden_size]
        contexts: List of context texts
        endings: List of ending texts (same length as contexts and compression_tokens_list)
        device: Device to use

    Returns:
        List of perplexity scores
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    if len(contexts) == 0:
        return []

    # Combine contexts and endings
    full_texts = [ctx + end for ctx, end in zip(contexts, endings)]

    # Tokenize with padding
    encoded = tokenizer(full_texts, truncation=True, padding=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)  # [batch_size, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # [batch_size, seq_len]

    # Get token embeddings for all texts
    token_embeddings = model.model.embed_tokens(input_ids)  # [batch_size, seq_len, hidden]

    # Prepare batched inputs with compression tokens
    batch_embeddings_list = []
    batch_attention_list = []
    num_compression_tokens = compression_tokens_list[0].shape[0]

    for i in range(len(contexts)):
        seq_len = attention_mask[i].sum().item()
        sample_token_embeds = token_embeddings[i : i + 1, :seq_len]  # [1, seq_len, hidden]
        sample_attention = attention_mask[i : i + 1, :seq_len]  # [1, seq_len]
        sample_compression = (
            compression_tokens_list[i].unsqueeze(0).to(token_embeddings.dtype)
        )  # [1, num_compression_tokens, hidden]

        # Concatenate
        united_emb = torch.cat([sample_compression, sample_token_embeds], dim=1)  # [1, mem+seq, hidden]
        comp_attn = torch.ones((1, num_compression_tokens), dtype=sample_attention.dtype, device=device)
        united_attn = torch.cat([comp_attn, sample_attention], dim=1)  # [1, mem+seq]

        batch_embeddings_list.append(united_emb)
        batch_attention_list.append(united_attn)

    # Pad to same length for batching
    max_len = max(emb.shape[1] for emb in batch_embeddings_list)
    batch_embeddings = []
    batch_attention = []

    for i in range(len(contexts)):
        emb = batch_embeddings_list[i]  # [1, L, hidden]
        attn = batch_attention_list[i]  # [1, L]

        current_len = emb.shape[1]
        if current_len < max_len:
            pad_len = max_len - current_len
            hidden_size = emb.shape[2]
            emb = torch.cat([emb, torch.zeros(1, pad_len, hidden_size, dtype=emb.dtype, device=device)], dim=1)
            attn = torch.cat([attn, torch.zeros(1, pad_len, dtype=attn.dtype, device=device)], dim=1)

        batch_embeddings.append(emb)
        batch_attention.append(attn)

    batch_embeddings = torch.cat(batch_embeddings, dim=0)  # [batch_size, max_len, hidden]
    batch_attention = torch.cat(batch_attention, dim=0)  # [batch_size, max_len]

    # Forward pass
    outputs = model(inputs_embeds=batch_embeddings, attention_mask=batch_attention)

    # Compute PPL for each sample
    ppls = []
    for i in range(len(contexts)):
        seq_len = attention_mask[i].sum().item()
        sample_logits = outputs.logits[i : i + 1, num_compression_tokens : num_compression_tokens + seq_len]
        sample_input_ids = input_ids[i : i + 1, :seq_len]
        sample_attention = attention_mask[i : i + 1, :seq_len]
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids, sample_attention)
        if math.isnan(ppl):
            ppl = float("inf")
        ppls.append(ppl)

    return ppls


@torch.no_grad()
def compute_ppl_with_compression(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    compression_tokens: torch.Tensor,
    context: str,
    ending: str,
    device: Optional[torch.device] = None,
) -> float:
    """Compute perplexity of context + ending using compression tokens.

    Args:
        model: The language model
        tokenizer: Tokenizer
        compression_tokens: Compression token embeddings [num_compression_tokens, hidden_size]
        context: Context text
        ending: Ending text
        device: Device to use

    Returns:
        Perplexity score
    """
    results = compute_ppl_with_compression_batch(model, tokenizer, [compression_tokens], [context], [ending], device)
    return results[0] if results else float("inf")


def parse_choices(choices_data):
    """Parse choices JSON and extract text array.

    Args:
        choices_data: Can be a string (JSON) or dict

    Returns:
        List of choice texts
    """
    if isinstance(choices_data, str):
        choices_dict = json.loads(choices_data)
    else:
        choices_dict = choices_data

    if isinstance(choices_dict, dict) and "text" in choices_dict:
        return choices_dict["text"]
    elif isinstance(choices_dict, list):
        return choices_dict
    else:
        raise ValueError(f"Unexpected choices format: {choices_data}")


def map_answer_key(answer_key: str) -> int:
    """Map answerKey (A/B/C/D) to index (0/1/2/3).

    Args:
        answer_key: Answer key string (A, B, C, or D)

    Returns:
        Integer index (0, 1, 2, or 3)
    """
    if len(answer_key) != 1:
        raise ValueError(f"Invalid answerKey format: {answer_key}")
    return ord(answer_key.upper()) - ord("A")


def main():
    parser = argparse.ArgumentParser(description="Evaluate compression tokens on ARC benchmark")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="Model checkpoint to use",
    )
    parser.add_argument(
        "--arc_split",
        type=str,
        default="ARC-Easy",
        choices=["ARC-Easy", "ARC-Challenge"],
        help="ARC dataset split to use",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=100,
        help="Limit number of samples to evaluate",
    )
    parser.add_argument(
        "--num_compression_tokens",
        type=int,
        default=1,
        help="Number of compression tokens",
    )
    parser.add_argument(
        "--max_optimization_steps",
        type=int,
        default=1000,
        help="Maximum optimization steps for compression",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate for compression optimization",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/arc_evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["auto", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
        help="Torch dtype for model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for compression and evaluation",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "l2", "l1", "cosine"],
        help="Loss type for optimization. Use cross_entropy for CE-only; set hybrid_alpha to add activation alignment.",
    )
    parser.add_argument(
        "--hybrid_alpha",
        type=float,
        default=None,
        help="If set and loss_type != cross_entropy, adds hybrid_alpha * alignment_loss to CE loss.",
    )
    parser.add_argument(
        "--num_alignment_layers",
        type=int,
        default=0,
        help="Number of layers to align (0 = all layers). Used only when hybrid_alpha is set and loss_type != cross_entropy.",
    )
    parser.add_argument(
        "--inverted_alignment",
        action="store_true",
        help="If set, aligns the last num_alignment_layers instead of the first.",
    )
    parser.add_argument(
        "--no_bos_token",
        action="store_true",
        default=False,
        help="Disable BOS token insertion during tokenization.",
    )

    args = parser.parse_args()

    # Set random seed
    set_launch_seed(args.random_seed)

    # Resolve dtype
    torch_dtype = resolve_torch_dtype(args.dtype)
    device = get_device()

    # Load model and tokenizer
    print(f"Loading model from {args.model_checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_bos_supported = hasattr(tokenizer, "add_bos_token")
    if args.no_bos_token and add_bos_supported:
        tokenizer.add_bos_token = False
    add_special_tokens = not (args.no_bos_token and not add_bos_supported)

    # Load ARC dataset
    print(f"Loading ARC dataset ({args.arc_split})...")
    dataset = load_dataset("allenai/ai2_arc", name=args.arc_split, split="validation")
    if args.limit_samples:
        dataset = dataset.select(range(args.limit_samples))
    print(f"Evaluating on {len(dataset)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluation results
    results = []
    correct_predictions_compressed = 0
    correct_predictions_baseline = 0
    total_predictions = 0
    total_tokens = 0
    total_characters = 0
    correct_tokens_baseline = 0
    correct_tokens_compressed = 0
    correct_characters_baseline = 0
    correct_characters_compressed = 0

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_items = [dataset[i] for i in range(start_idx, end_idx)]
        actual_batch_size = len(batch_items)

        # Extract and process batch data
        batch_contexts = []
        batch_endings_list = []
        batch_labels = []

        for item in batch_items:
            # Use question as context
            question = item["question"]
            batch_contexts.append(question)

            # Parse choices and extract text array
            choices_text = parse_choices(item["choices"])
            batch_endings_list.append(choices_text)

            # Map answerKey to index
            answer_key = item["answerKey"]
            label = map_answer_key(answer_key)
            batch_labels.append(label)

        # Compute baseline PPL for all choices (batched)
        batch_baseline_ppls = []
        for sample_idx in range(actual_batch_size):
            sample_ppls = []
            contexts_for_batch = [batch_contexts[sample_idx]] * len(batch_endings_list[sample_idx])
            try:
                ppls = compute_ppl_baseline_batch(
                    model=model,
                    tokenizer=tokenizer,
                    contexts=contexts_for_batch,
                    endings=batch_endings_list[sample_idx],
                    device=device,
                    add_special_tokens=add_special_tokens,
                )
                sample_ppls = ppls
            except Exception as e:
                print(f"Error computing baseline PPL for sample {start_idx + sample_idx}: {e}")
                sample_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
            batch_baseline_ppls.append(sample_ppls)

        # Compress contexts (batched)
        batch_compression_tokens = None
        try:
            batch_compression_tokens = compress_prefixes_batch(
                model=model,
                tokenizer=tokenizer,
                texts=batch_contexts,
                num_compression_tokens=args.num_compression_tokens,
                max_steps=args.max_optimization_steps,
                learning_rate=args.learning_rate,
                loss_type=args.loss_type,
                hybrid_alpha=args.hybrid_alpha,
                num_alignment_layers=args.num_alignment_layers,
                inverted_alignment=args.inverted_alignment,
                device=device,
                add_special_tokens=add_special_tokens,
            )
        except Exception as e:
            print(f"Error compressing batch {batch_idx}: {e}")
            batch_compression_tokens = [None] * actual_batch_size

        # Compute compressed PPL for all choices (batched)
        batch_compressed_ppls = []
        for sample_idx in range(actual_batch_size):
            if batch_compression_tokens[sample_idx] is None:
                batch_compressed_ppls.append([float("inf")] * len(batch_endings_list[sample_idx]))
                continue

            sample_ppls = []
            try:
                # Prepare contexts and endings for this sample
                contexts_for_batch = [batch_contexts[sample_idx]] * len(batch_endings_list[sample_idx])
                compression_tokens_for_batch = [batch_compression_tokens[sample_idx]] * len(batch_endings_list[sample_idx])

                ppls = compute_ppl_with_compression_batch(
                    model=model,
                    tokenizer=tokenizer,
                    compression_tokens_list=compression_tokens_for_batch,
                    contexts=contexts_for_batch,
                    endings=batch_endings_list[sample_idx],
                    device=device,
                    add_special_tokens=add_special_tokens,
                )
                sample_ppls = ppls
            except Exception as e:
                print(f"Error computing compressed PPL for sample {start_idx + sample_idx}: {e}")
                sample_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
            batch_compressed_ppls.append(sample_ppls)

        # Process results for this batch
        for sample_idx in range(actual_batch_size):
            idx = start_idx + sample_idx
            context = batch_contexts[sample_idx]
            endings = batch_endings_list[sample_idx]
            label = batch_labels[sample_idx]

            baseline_ppls = batch_baseline_ppls[sample_idx]
            baseline_predicted_label = int(torch.tensor(baseline_ppls).argmin().item())
            baseline_is_correct = baseline_predicted_label == label
            if baseline_is_correct:
                correct_predictions_baseline += 1

            compressed_ppls = batch_compressed_ppls[sample_idx]
            compressed_predicted_label = None
            compressed_is_correct = None
            if compressed_ppls and len(compressed_ppls) > 0:
                compressed_predicted_label = int(torch.tensor(compressed_ppls).argmin().item())
                compressed_is_correct = compressed_predicted_label == label
                if compressed_is_correct:
                    correct_predictions_compressed += 1

            total_predictions += 1
            token_count = None
            char_count = None
            if 0 <= label < len(endings):
                correct_ending = endings[label]
                full_text = context + correct_ending
                token_count = count_text_tokens(tokenizer, full_text, add_special_tokens=add_special_tokens)
                char_count = count_text_characters(full_text)
                total_tokens += token_count
                total_characters += char_count
                if baseline_is_correct:
                    correct_tokens_baseline += token_count
                    correct_characters_baseline += char_count
                if compressed_is_correct:
                    correct_tokens_compressed += token_count
                    correct_characters_compressed += char_count

            # Store result
            result = {
                "sample_id": idx,
                "context": context,
                "endings": endings,
                "label": label,
                "lengths": {
                    "tokens": token_count,
                    "characters": char_count,
                },
                "baseline": {
                    "predicted_label": baseline_predicted_label,
                    "is_correct": baseline_is_correct,
                    "ppls": baseline_ppls,
                },
                "compressed": {
                    "predicted_label": compressed_predicted_label,
                    "is_correct": compressed_is_correct,
                    "ppls": compressed_ppls,
                },
            }
            results.append(result)

        # Print progress
        if (batch_idx + 1) % max(1, num_batches // 10) == 0 or (batch_idx + 1) == num_batches:
            baseline_accuracy = correct_predictions_baseline / total_predictions if total_predictions > 0 else 0.0
            compressed_accuracy = correct_predictions_compressed / total_predictions if total_predictions > 0 else 0.0
            print(
                f"Progress: {total_predictions}/{len(dataset)}, Baseline Accuracy: {baseline_accuracy:.4f}, "
                f"Compressed Accuracy: {compressed_accuracy:.4f}"
            )

    # Compute final accuracies
    baseline_accuracy = correct_predictions_baseline / total_predictions if total_predictions > 0 else 0.0
    compressed_accuracy = correct_predictions_compressed / total_predictions if total_predictions > 0 else 0.0
    baseline_token_accuracy = correct_tokens_baseline / total_tokens if total_tokens > 0 else 0.0
    compressed_token_accuracy = correct_tokens_compressed / total_tokens if total_tokens > 0 else 0.0
    baseline_char_accuracy = correct_characters_baseline / total_characters if total_characters > 0 else 0.0
    compressed_char_accuracy = correct_characters_compressed / total_characters if total_characters > 0 else 0.0

    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "arc_split": args.arc_split,
                "baseline": {
                    "accuracy": baseline_accuracy,
                    "token_normalized_accuracy": baseline_token_accuracy,
                    "char_normalized_accuracy": baseline_char_accuracy,
                    "correct_predictions": correct_predictions_baseline,
                    "total_predictions": total_predictions,
                    "total_tokens": total_tokens,
                    "total_characters": total_characters,
                },
                "compressed": {
                    "accuracy": compressed_accuracy,
                    "token_normalized_accuracy": compressed_token_accuracy,
                    "char_normalized_accuracy": compressed_char_accuracy,
                    "correct_predictions": correct_predictions_compressed,
                    "total_predictions": total_predictions,
                    "total_tokens": total_tokens,
                    "total_characters": total_characters,
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"ARC Split: {args.arc_split}")
    print(f"Total samples: {total_predictions}")
    print("\nBaseline (without compression):")
    print(f"  Correct predictions: {correct_predictions_baseline}")
    print(f"  Accuracy: {baseline_accuracy:.4f}")
    print(f"  Token-normalized Accuracy: {baseline_token_accuracy:.4f}")
    print(f"  Character-normalized Accuracy: {baseline_char_accuracy:.4f}")
    print("\nCompressed (with compression tokens):")
    print(f"  Correct predictions: {correct_predictions_compressed}")
    print(f"  Accuracy: {compressed_accuracy:.4f}")
    print(f"  Token-normalized Accuracy: {compressed_token_accuracy:.4f}")
    print(f"  Character-normalized Accuracy: {compressed_char_accuracy:.4f}")
    print(f"\nDifference: {compressed_accuracy - baseline_accuracy:+.4f}")
    print(f"Results saved to: {results_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
