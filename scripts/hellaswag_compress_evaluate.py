"""Evaluate compression on HellaSwag benchmark.

This script:
1. Downloads HellaSwag dataset
2. Compresses the context (prefix) for each item
3. Evaluates by computing PPL for all endings and choosing the one with lowest PPL
4. Dumps evaluation results
"""

import argparse
import inspect
import json
import math
import os
from typing import Optional

import torch
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from compression_horizon.analysis.attention_intervention import (
    build_intervention_result,
    build_intervention_summary,
    evaluate_sample_interventions,
    get_decoder_layers,
    print_intervention_summary,
)
from compression_horizon.analysis.perplexity import estimate_token_perplexity, estimate_token_perplexity_full_labels
from compression_horizon.train.loss import compute_hybrid_cross_entropy_and_alignment_loss
from compression_horizon.utils.launch import freeze_model_parameters, get_device, resolve_torch_dtype, set_launch_seed
from compression_horizon.utils.tokens import count_text_characters, count_text_tokens


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
) -> list[dict[str, torch.Tensor | float]]:
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
    # max_length=training_args.max_sequence_length
    encoded = tokenizer(texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)  # [batch, sequence]
    attention_mask = encoded["attention_mask"].to(device)  # [batch, sequence]

    # Get token embeddings to determine dtype and hidden_size
    with torch.no_grad():
        token_embeddings = model.get_input_embeddings()(input_ids)  # [batch, sequence, hidden]
    hidden_size = token_embeddings.shape[-1]

    # Some models (Gemma3) require token_type_ids during training
    _needs_token_type_ids = "token_type_ids" in inspect.signature(model.forward).parameters

    # Initialize compression tokens for all samples
    compression_token_embeddings = torch.nn.Parameter(
        torch.randn([batch_size, num_compression_tokens, hidden_size], dtype=token_embeddings.dtype, device=device) * 0.02
    )  # [batch, compression, hidden]

    # Optimizer
    optimizer = AdamW([compression_token_embeddings], lr=learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps,
    )

    model.train()
    # Training loop
    for _ in range(max_steps):
        optimizer.zero_grad()

        # Concatenate compression tokens with input embeddings
        # Expand compression tokens to match sequence lengths
        united_token_embeddings_list = []
        united_attention_mask_list = []
        labels_list = []
        for i in range(batch_size):
            sequence_len = int(attention_mask[i].sum().item())
            sample_token_embeddings = token_embeddings[i : i + 1, :sequence_len]  # [1, sequence, hidden]
            sample_attention_mask = attention_mask[i : i + 1, :sequence_len]  # [1, sequence]
            sample_compression_token_embeddings = compression_token_embeddings[i : i + 1]  # [1, compression, hidden]
            # Concatenate
            united_token_embeddings = torch.cat(
                [sample_compression_token_embeddings, sample_token_embeddings], dim=1
            )  # [1, compression + sequence, hidden]
            united_attention_mask = torch.cat(
                [
                    torch.ones((1, num_compression_tokens), dtype=sample_attention_mask.dtype, device=device),
                    sample_attention_mask,
                ],
                dim=1,
            )  # [1, compression + sequence]
            united_token_embeddings_list.append(united_token_embeddings)
            united_attention_mask_list.append(united_attention_mask)
            # Labels for this sample
            sample_input_ids = input_ids[i : i + 1, :sequence_len].clone()  # [1, sequence]
            labels_list.append(sample_input_ids)

        # Pad to maximum length and gather batches
        max_len = max(item.shape[1] for item in united_token_embeddings_list)
        max_seq_len = max(item.shape[1] for item in labels_list)  # max sequence length (without compression)
        batch_embeddings = []
        batch_attention = []
        batch_labels = []
        for i in range(batch_size):
            united_token_embeddings = united_token_embeddings_list[i]  # [1, compression + sequence, hidden]
            united_attention_mask = united_attention_mask_list[i]  # [1, compression + sequence]
            labels = labels_list[i]  # [1, sequence]
            current_len = united_token_embeddings.shape[1]
            current_seq_len = labels.shape[1]
            if current_len < max_len:
                pad_len = max_len - current_len
                united_token_embeddings = torch.cat(
                    [
                        united_token_embeddings,
                        torch.zeros(1, pad_len, hidden_size, dtype=united_token_embeddings.dtype, device=device),
                    ],
                    dim=1,
                )  # [1, max_len, hidden]
                united_attention_mask = torch.cat(
                    [
                        united_attention_mask,
                        torch.zeros(1, pad_len, dtype=united_attention_mask.dtype, device=device),
                    ],
                    dim=1,
                )  # [1, max_len]
            # Pad labels to max sequence length
            if current_seq_len < max_seq_len:
                labels_pad_len = max_seq_len - current_seq_len
                labels = torch.cat(
                    [
                        labels,
                        torch.zeros(1, labels_pad_len, dtype=labels.dtype, device=device),
                    ],
                    dim=1,
                )  # [1, max_seq_len]
            batch_embeddings.append(united_token_embeddings)
            batch_attention.append(united_attention_mask)
            batch_labels.append(labels)
        batch_embeddings = torch.cat(batch_embeddings, dim=0)  # [batch_size, max_len, hidden]
        batch_attention = torch.cat(batch_attention, dim=0)  # [batch_size, max_len]
        batch_labels = torch.cat(batch_labels, dim=0)  # [batch_size, max_seq_len]

        # Build optional kwargs for models that need token_type_ids (e.g. Gemma3)
        target_fwd_kwargs = {}
        compression_fwd_kwargs = {}
        if _needs_token_type_ids:
            target_fwd_kwargs["token_type_ids"] = torch.zeros(attention_mask.shape, dtype=torch.long, device=device)
            compression_fwd_kwargs["token_type_ids"] = torch.zeros(batch_attention.shape, dtype=torch.long, device=device)

        # Forward pass without compression tokens and gradient capturing
        target_outputs = None
        if use_alignment:
            with torch.no_grad():
                target_outputs = model(
                    inputs_embeds=token_embeddings,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    **target_fwd_kwargs,
                )
        # Forward pass with compression tokens and gradient capturing
        compression_outputs = model(
            inputs_embeds=batch_embeddings,
            attention_mask=batch_attention,
            output_hidden_states=use_alignment,
            **compression_fwd_kwargs,
        )

        # Compute loss per sample
        total_loss = 0.0
        for i in range(batch_size):
            sequence_len = int(attention_mask[i].sum().item())
            sample_logits = compression_outputs.logits[
                i : i + 1, : num_compression_tokens + sequence_len
            ]  # [1, compression + sequence, vocabulary]
            sample_input_ids = input_ids[i : i + 1, :sequence_len]  # [1, sequence]
            sample_attention_mask = attention_mask[i : i + 1, :sequence_len]  # [1, sequence]
            if use_alignment:
                assert target_outputs is not None
                sample_compression_hidden_states = tuple(
                    hs_layer[i : i + 1, : num_compression_tokens + sequence_len]
                    for hs_layer in compression_outputs.hidden_states
                )
                sample_target_hidden_states = tuple(
                    hs_layer[i : i + 1, :sequence_len] for hs_layer in target_outputs.hidden_states
                )
                sample_loss, _ = compute_hybrid_cross_entropy_and_alignment_loss(
                    logits=sample_logits,
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                    num_compression_tokens=num_compression_tokens,
                    target_hidden_states=sample_target_hidden_states,
                    compression_hidden_states=sample_compression_hidden_states,
                    num_alignment_layers=num_alignment_layers,
                    inverted_alignment=inverted_alignment,
                    loss_type=loss_type,
                    hybrid_alpha=hybrid_alpha,
                )
            else:
                sample_loss, _ = compute_hybrid_cross_entropy_and_alignment_loss(
                    logits=sample_logits,
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                    num_compression_tokens=num_compression_tokens,
                    num_alignment_layers=num_alignment_layers,
                    inverted_alignment=inverted_alignment,
                    loss_type=loss_type,
                    hybrid_alpha=hybrid_alpha,
                )
            total_loss = total_loss + sample_loss
        loss = total_loss / batch_size
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    model.eval()
    # Return compression tokens (remove batch dimension for each)
    convergences = calculate_convergence(model, batch_embeddings, batch_attention, batch_labels, num_compression_tokens)
    print("Batch convergences:", convergences)
    return [
        {
            "compression_embedding": compression_token_embeddings[i].detach(),  # [compression, hidden]
            "convergence": convergences[i],
        }
        for i in range(batch_size)
    ]


@torch.no_grad()
def calculate_convergence(
    model: AutoModelForCausalLM,
    batch_embeddings: torch.Tensor,
    batch_attention: torch.Tensor,
    batch_labels: torch.Tensor,
    num_compression_tokens: int,
) -> list[float]:
    """Calculate token-level accuracy between predicted and target tokens.

    Args:
        batch_embeddings: [batch_size, max_len, hidden] - input embeddings with compression tokens
        batch_attention: [batch_size, max_len] - attention mask
        batch_labels: [batch_size, max_seq_len] - target token IDs (without compression positions)
        num_compression_tokens: number of compression tokens prepended

    Returns:
        List of convergence scores (0.0 to 1.0) for each sample
    """
    outputs = model(inputs_embeds=batch_embeddings, attention_mask=batch_attention)
    batch_size = batch_embeddings.shape[0]
    convergences = []
    for i in range(batch_size):
        # total_len includes compression tokens but excludes padding
        total_len = int(batch_attention[i].sum().item())
        # original sequence length (without compression tokens)
        orig_seq_len = total_len - num_compression_tokens
        # For next-token prediction:
        # - logits[num_compression_tokens - 1] predicts token at position num_compression_tokens (first original token)
        # - logits[total_len - 2] predicts token at position total_len - 1 (last original token)
        # So we need logits from index (num_compression_tokens - 1) to (total_len - 2) inclusive
        sample_logits = outputs.logits[i, num_compression_tokens - 1 : total_len - 1]  # [orig_seq_len, vocabulary]
        sample_predicted_tokens = sample_logits.argmax(dim=-1)  # [orig_seq_len]
        sample_labels = batch_labels[i, :orig_seq_len]  # [orig_seq_len] - exclude padding

        matches = sample_predicted_tokens == sample_labels
        convergence = matches.sum().item() / matches.numel()
        convergences.append(convergence)
    return convergences


@torch.no_grad()
def compute_ppl_baseline_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: str,
    endings: list[str],
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> tuple[list[float], list[float]]:
    """Compute perplexity of context + ending pairs without compression tokens (batched).

    Args:
        model: The language model
        tokenizer: Tokenizer
        context: Context text
        endings: List of ending texts
        device: Device to use
        add_special_tokens: Whether to add special tokens during tokenization

    Returns:
        Tuple of (full_ppls, ending_only_ppls):
            - full_ppls: Perplexity scores for full context + ending
            - ending_only_ppls: Perplexity scores for ending part only
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    if not context:
        return [], []

    # Combine context and endings
    full_texts = [f"{context} {ending}" for ending in endings]

    # Tokenize with padding
    encoded = tokenizer(
        full_texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens
    )
    input_ids = encoded["input_ids"].to(device)  # [batch, sequence]
    attention_mask = encoded["attention_mask"].to(device)  # [batch, sequence]

    input_context_ids = tokenizer(f"{context} ", add_special_tokens=add_special_tokens)["input_ids"]
    context_len = len(input_context_ids)

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Compute PPL for each sample
    ppls = []
    endings_ppls = []
    for i in range(len(full_texts)):
        # Perplexity on all sequence
        sequence_len = int(attention_mask[i].sum().item())
        sample_logits = outputs.logits[i : i + 1, :sequence_len]  # [1, sequence, vocabulary]
        sample_input_ids = input_ids[i : i + 1, :sequence_len]  # [1, sequence]
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids)
        if math.isnan(ppl):
            ppl = float("inf")
        ppls.append(ppl)
        # Perplexity only on ending
        sample_ending_logits = sample_logits[:, context_len - 1 :, :]  # [1, ending, vocabulary]
        sample_ending_input_ids = sample_input_ids[:, context_len - 1 :]  # [1, ending]
        ending_ppl = estimate_token_perplexity(sample_ending_logits, sample_ending_input_ids)
        if math.isnan(ending_ppl):
            ending_ppl = float("inf")
        endings_ppls.append(ending_ppl)
    return ppls, endings_ppls


@torch.no_grad()
def compute_ppl_compression_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    compression_token_embeddings: torch.Tensor,
    context: str,
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

    # Combine context and endings (context can be empty for compression-only mode)
    full_texts = [f"{context}{ending}" for ending in endings]

    # Tokenize with padding
    encoded = tokenizer(
        full_texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens
    )
    input_ids = encoded["input_ids"].to(device)  # [batch, sequence]
    attention_mask = encoded["attention_mask"].to(device)  # [batch, sequence]

    # Context length (0 if context is empty)
    if context:
        input_context_ids = tokenizer(context, add_special_tokens=add_special_tokens)["input_ids"]
        context_len = len(input_context_ids)
    else:
        context_len = 0

    # Get token embeddings for all texts
    token_embeddings = model.get_input_embeddings()(input_ids)  # [batch_size, sequence, hidden]

    # Prepare batched inputs with compression tokens
    united_token_embeddings_list = []
    united_attention_mask_list = []
    num_compression_tokens = compression_token_embeddings.shape[0]
    for i in range(len(full_texts)):
        sequence_len = int(attention_mask[i].sum().item())
        sample_token_embeddings = token_embeddings[i : i + 1, :sequence_len]  # [1, sequence, hidden]
        sample_attention_mask = attention_mask[i : i + 1, :sequence_len]  # [1, sequence]
        sample_compression_token_embeddings = compression_token_embeddings.unsqueeze(0).to(
            token_embeddings.dtype
        )  # [1, compression, hidden]
        # Concatenate
        united_token_embeddings = torch.cat(
            [sample_compression_token_embeddings, sample_token_embeddings], dim=1
        )  # [1, compression + sequence, hidden]
        united_attention_mask = torch.cat(
            [torch.ones((1, num_compression_tokens), dtype=sample_attention_mask.dtype, device=device), sample_attention_mask],
            dim=1,
        )  # [1, compression + sequence]
        united_token_embeddings_list.append(united_token_embeddings)
        united_attention_mask_list.append(united_attention_mask)

    # Pad to maximum length and gather batches
    max_len = max(item.shape[1] for item in united_token_embeddings_list)
    batch_embeddings = []
    batch_attention = []
    for i in range(len(full_texts)):
        united_token_embeddings = united_token_embeddings_list[i]  # [1, compression + sequence, hidden]
        united_attention_mask = united_attention_mask_list[i]  # [1, compression + sequence]
        current_len = united_token_embeddings.shape[1]
        if current_len < max_len:
            pad_len = max_len - current_len
            hidden_size = united_token_embeddings.shape[2]
            united_token_embeddings = torch.cat(
                [
                    united_token_embeddings,
                    torch.zeros(1, pad_len, hidden_size, dtype=united_token_embeddings.dtype, device=device),
                ],
                dim=1,
            )  # [1, max_len, hidden]
            united_attention_mask = torch.cat(
                [
                    united_attention_mask,
                    torch.zeros(1, pad_len, dtype=united_attention_mask.dtype, device=device),
                ],
                dim=1,
            )  # [1, max_len]
        batch_embeddings.append(united_token_embeddings)
        batch_attention.append(united_attention_mask)
    batch_embeddings = torch.cat(batch_embeddings, dim=0)  # [batch_size, max_len, hidden]
    batch_attention = torch.cat(batch_attention, dim=0)  # [batch_size, max_len]

    # Forward pass
    outputs = model(inputs_embeds=batch_embeddings, attention_mask=batch_attention)

    # Compute PPL for each sample
    ppls = []
    edge_ppls = []
    endings_ppls = []
    for i in range(len(full_texts)):
        # Perplexity on all sequence
        sequence_len = int(attention_mask[i].sum().item())
        sample_logits = outputs.logits[
            i : i + 1, num_compression_tokens : num_compression_tokens + sequence_len
        ]  # [1, sequence, vocabulary]
        sample_input_ids = input_ids[i : i + 1, :sequence_len]  # [1, sequence]
        ppl = estimate_token_perplexity(sample_logits, sample_input_ids)
        if math.isnan(ppl):
            ppl = float("inf")
        ppls.append(ppl)
        # Perplexity with compression tokens edge
        sample_logits = outputs.logits[
            i : i + 1, num_compression_tokens - 1 : num_compression_tokens + sequence_len
        ]  # [1, sequence + 1, vocabulary]
        sample_input_ids = input_ids[i : i + 1, :sequence_len]  # [1, sequence]
        edge_ppl = estimate_token_perplexity_full_labels(sample_logits, sample_input_ids)
        if math.isnan(edge_ppl):
            edge_ppl = float("inf")
        edge_ppls.append(edge_ppl)
        # Perplexity only on endings
        sample_logits = outputs.logits[
            i : i + 1, num_compression_tokens + max(0, context_len - 1) : num_compression_tokens + sequence_len, :
        ]  # [1, ending, vocabulary]
        sample_input_ids = input_ids[i : i + 1, max(0, context_len - 1) : sequence_len]  # [1, ending]
        ending_ppl = estimate_token_perplexity(sample_logits, sample_input_ids)
        if math.isnan(ending_ppl):
            ending_ppl = float("inf")
        endings_ppls.append(ending_ppl)
    return ppls, edge_ppls, endings_ppls


def main():
    parser = argparse.ArgumentParser(description="Evaluate compression tokens on HellaSwag benchmark")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="Model checkpoint to use",
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
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for compression and evaluation",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["auto", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
        help="Torch dtype for model",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "l2", "l1", "cosine"],
        help="Loss type for optimization. Use cross_entropy for CE-only; set hybrid_alpha to add activation alignment.",
    )
    parser.add_argument(
        "--num_alignment_layers",
        type=int,
        default=0,
        help="Number of layers to align (0 = all layers). Used only when hybrid_alpha is set and loss_type != cross_entropy.",
    )
    parser.add_argument(
        "--hybrid_alpha",
        type=float,
        default=None,
        help="If set and loss_type != cross_entropy, adds hybrid_alpha * alignment_loss to CE loss.",
    )
    parser.add_argument(
        "--inverted_alignment",
        action="store_true",
        default=False,
        help="If set, aligns the last num_alignment_layers instead of the first.",
    )
    parser.add_argument(
        "--no-bos-token",
        "--no_bos_token",
        dest="bos_token",
        action="store_false",
        default=True,
        help="Disable BOS token insertion during tokenization.",
    )
    parser.add_argument(
        "--only_full_convergence",
        action="store_true",
        default=False,
        help="Only count examples with perfect convergence (accuracy=1.0) in compressed metrics.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/hellaswag_evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-intervention",
        "--no_intervention",
        dest="intervention",
        action="store_false",
        default=True,
        help="Disable attention knockout intervention mode (enabled by default).",
    )
    parser.add_argument(
        "--skip_per_layer",
        action="store_true",
        default=False,
        help="Skip per-layer knockout sweep (only used with --intervention).",
    )
    parser.add_argument(
        "--skip_cumulative",
        action="store_true",
        default=False,
        help="Skip cumulative knockout sweep (only used with --intervention).",
    )
    parser.add_argument(
        "--skip_reverse_cumulative",
        action="store_true",
        default=False,
        help="Skip reverse cumulative knockout sweep (only used with --intervention).",
    )
    args = parser.parse_args()

    # Set random seed
    set_launch_seed(args.random_seed)
    # Resolve dtype
    torch_dtype = resolve_torch_dtype(args.dtype)
    device = get_device()
    # Load model
    print(f"Loading model from {args.model_checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, dtype=torch_dtype)
    print("Loaded model dtype:", next(model.parameters()).dtype)
    # Get number of layers for intervention mode
    num_model_layers = None
    if args.intervention:
        num_model_layers = len(get_decoder_layers(model))
        print(f"Intervention mode enabled. Model has {num_model_layers} layers.")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Handle BOS token settings
    add_bos_supported = hasattr(tokenizer, "add_bos_token")
    if not args.bos_token:
        if add_bos_supported:
            tokenizer.add_bos_token = False
            add_special_tokens = True
        else:
            # Fallback: disable special tokens entirely if tokenizer doesn't support add_bos_token
            add_special_tokens = False
    else:
        add_special_tokens = True

    # Load HellaSwag dataset
    print("Loading HellaSwag dataset...")
    # ind - dataset ID
    # activity_label - specifies the subject areas for sentence completion evaluation
    # ctx - full context
    # endings - a list of 4 endings
    # label - correct label 0, 1, 2 or 3
    # split_type - indomain if the activity label is seen during training, else zeroshot
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    if args.limit_samples:
        dataset = dataset.select(range(args.limit_samples))
    print(f"Evaluating HellaSwag benchmark on {len(dataset)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluation results
    results = []
    # Baseline counters (always count all samples)
    total_predictions_baseline = 0
    correct_predictions_baseline = 0
    total_tokens_baseline = 0
    correct_tokens_baseline = 0
    total_characters_baseline = 0
    correct_characters_baseline = 0

    correct_predictions_baseline_endings = 0
    correct_tokens_baseline_endings = 0

    # Compressed counters (may exclude non-converged samples when only_full_convergence=True)
    # All compression variants share the same total_predictions and total_tokens
    # since they use the same should_count_compressed condition
    total_predictions_compression = 0
    total_tokens_compression = 0
    total_characters_compression = 0

    correct_predictions_compression = 0
    correct_tokens_compression = 0
    correct_characters_compression = 0

    correct_predictions_compression_edge = 0
    correct_tokens_compression_edge = 0

    correct_predictions_compression_endings = 0
    correct_tokens_compression_endings = 0

    correct_predictions_compression_only = 0
    correct_tokens_compression_only = 0

    correct_predictions_compression_only_edge = 0
    correct_tokens_compression_only_edge = 0

    correct_predictions_compression_only_endings = 0
    correct_tokens_compression_only_endings = 0

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_items = [dataset[i] for i in range(start_idx, end_idx)]
        current_batch_size = len(batch_items)

        # Extract batch data
        batch_contexts = [item["ctx"] for item in batch_items]
        batch_endings_list = [item["endings"] for item in batch_items]
        batch_labels = [int(item["label"]) for item in batch_items]

        # Compute baseline PPL for all endings (batched)
        batch_baseline_ppls = []
        batch_baseline_endings_ppls = []
        for sample_idx in range(current_batch_size):
            try:
                sample_ppls, sample_endings_ppls = compute_ppl_baseline_batch(
                    model=model,
                    tokenizer=tokenizer,
                    context=batch_contexts[sample_idx],
                    endings=batch_endings_list[sample_idx],
                    device=device,
                    add_special_tokens=add_special_tokens,
                )
            except Exception as e:
                print(f"Error computing baseline PPL for sample {start_idx + sample_idx}: {e}")
                sample_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
                sample_endings_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
            batch_baseline_ppls.append(sample_ppls)
            batch_baseline_endings_ppls.append(sample_endings_ppls)

        # Compress contexts (batched)
        batch_compression_results = None
        try:
            # List[dict] with keys: compression_embedding, convergence
            batch_compression_results = compress_prefixes_batch(
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
            batch_compression_results = [{"compression_embedding": None, "convergence": 0.0}] * current_batch_size

        # Compute compression PPL for all endings (batched)
        batch_compression_ppls = []
        batch_compression_edge_ppls = []
        batch_compression_endings_ppls = []
        batch_compression_only_ppls = []
        batch_compression_only_edge_ppls = []
        batch_compression_only_endings_ppls = []
        for sample_idx in range(current_batch_size):
            compression_embedding = batch_compression_results[sample_idx]["compression_embedding"]
            convergence = batch_compression_results[sample_idx]["convergence"]

            # If compression failed
            if compression_embedding is None:
                batch_compression_ppls.append(
                    {"ppls": [float("inf")] * len(batch_endings_list[sample_idx]), "convergence": convergence}
                )
                batch_compression_edge_ppls.append(
                    {"ppls": [float("inf")] * len(batch_endings_list[sample_idx]), "convergence": convergence}
                )
                batch_compression_endings_ppls.append(
                    {"ppls": [float("inf")] * len(batch_endings_list[sample_idx]), "convergence": convergence}
                )
                batch_compression_only_ppls.append(
                    {"ppls": [float("inf")] * len(batch_endings_list[sample_idx]), "convergence": convergence}
                )
                batch_compression_only_edge_ppls.append(
                    {"ppls": [float("inf")] * len(batch_endings_list[sample_idx]), "convergence": convergence}
                )
                batch_compression_only_endings_ppls.append(
                    {"ppls": [float("inf")] * len(batch_endings_list[sample_idx]), "convergence": convergence}
                )
                continue

            # Compute PPL with compression tokens
            try:
                sample_ppls, sample_edge_ppls, sample_endings_ppls = compute_ppl_compression_batch(
                    model=model,
                    tokenizer=tokenizer,
                    compression_token_embeddings=compression_embedding,
                    context=batch_contexts[sample_idx] + " ",
                    endings=batch_endings_list[sample_idx],
                    device=device,
                    add_special_tokens=add_special_tokens,
                )
                sample_only_ppls, sample_only_edge_ppls, sample_only_endings_ppls = compute_ppl_compression_batch(
                    model=model,
                    tokenizer=tokenizer,
                    compression_token_embeddings=compression_embedding,
                    context="",
                    endings=batch_endings_list[sample_idx],
                    device=device,
                    add_special_tokens=add_special_tokens,
                )
            except Exception as e:
                print(f"Error computing compression PPL for sample {start_idx + sample_idx}: {e}")
                sample_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
                sample_edge_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
                sample_endings_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
                sample_only_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
                sample_only_edge_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
                sample_only_endings_ppls = [float("inf")] * len(batch_endings_list[sample_idx])
            batch_compression_ppls.append({"ppls": sample_ppls, "convergence": convergence})
            batch_compression_edge_ppls.append({"ppls": sample_edge_ppls, "convergence": convergence})
            batch_compression_endings_ppls.append({"ppls": sample_endings_ppls, "convergence": convergence})
            batch_compression_only_ppls.append({"ppls": sample_only_ppls, "convergence": convergence})
            batch_compression_only_edge_ppls.append({"ppls": sample_only_edge_ppls, "convergence": convergence})
            batch_compression_only_endings_ppls.append({"ppls": sample_only_endings_ppls, "convergence": convergence})

        # Compute knockout PPLs and attention mass for intervention mode
        batch_intervention_data = []
        if args.intervention:
            for sample_idx in range(current_batch_size):
                compression_embedding = batch_compression_results[sample_idx]["compression_embedding"]
                if compression_embedding is None:
                    batch_intervention_data.append(None)
                    continue
                batch_intervention_data.append(
                    evaluate_sample_interventions(
                        model=model,
                        tokenizer=tokenizer,
                        compression_embedding=compression_embedding,
                        context=batch_contexts[sample_idx],
                        endings=batch_endings_list[sample_idx],
                        num_compression_tokens=args.num_compression_tokens,
                        num_model_layers=num_model_layers,
                        device=device,
                        add_special_tokens=add_special_tokens,
                        skip_per_layer=args.skip_per_layer,
                        skip_cumulative=args.skip_cumulative,
                        skip_reverse_cumulative=args.skip_reverse_cumulative,
                    )
                )

        # Process results for this batch
        for sample_idx in range(current_batch_size):
            idx = start_idx + sample_idx
            context = batch_contexts[sample_idx]
            endings = batch_endings_list[sample_idx]
            label = batch_labels[sample_idx]

            # Baseline evaluation
            baseline_ppls = batch_baseline_ppls[sample_idx]
            baseline_predicted_label = int(torch.tensor(baseline_ppls).argmin().item())
            baseline_is_correct = baseline_predicted_label == label

            # Baseline endings evaluation
            baseline_endings_ppls = batch_baseline_endings_ppls[sample_idx]
            baseline_endings_predicted_label = int(torch.tensor(baseline_endings_ppls).argmin().item())
            baseline_endings_is_correct = baseline_endings_predicted_label == label

            # Compressed evaluation
            convergence = batch_compression_ppls[sample_idx]["convergence"]
            is_fully_converged = convergence >= 1.0

            # Compression
            compression_ppls = batch_compression_ppls[sample_idx]["ppls"]
            compression_predicted_label = int(torch.tensor(compression_ppls).argmin().item())
            compression_is_correct = compression_predicted_label == label

            # Compression with edge
            compression_edge_ppls = batch_compression_edge_ppls[sample_idx]["ppls"]
            compression_edge_predicted_label = int(torch.tensor(compression_edge_ppls).argmin().item())
            compression_edge_is_correct = compression_edge_predicted_label == label

            # Compression endings
            compression_endings_ppls = batch_compression_endings_ppls[sample_idx]["ppls"]
            compression_endings_predicted_label = int(torch.tensor(compression_endings_ppls).argmin().item())
            compression_endings_is_correct = compression_endings_predicted_label == label

            # Compression only
            compression_only_ppls = batch_compression_only_ppls[sample_idx]["ppls"]
            compression_only_predicted_label = int(torch.tensor(compression_only_ppls).argmin().item())
            compression_only_is_correct = compression_only_predicted_label == label

            # Compression only with edge
            compression_only_edge_ppls = batch_compression_only_edge_ppls[sample_idx]["ppls"]
            compression_only_edge_predicted_label = int(torch.tensor(compression_only_edge_ppls).argmin().item())
            compression_only_edge_is_correct = compression_only_edge_predicted_label == label

            # Compression only endings
            compression_only_endings_ppls = batch_compression_only_endings_ppls[sample_idx]["ppls"]
            compression_only_endings_predicted_label = int(torch.tensor(compression_only_endings_ppls).argmin().item())
            compression_only_endings_is_correct = compression_only_endings_predicted_label == label

            # Update baseline counters (always count all samples)
            total_predictions_baseline += 1
            correct_predictions_baseline += int(baseline_is_correct)
            correct_predictions_baseline_endings += int(baseline_endings_is_correct)

            # Determine if this sample should be counted in compressed metrics
            should_count_compressed = not args.only_full_convergence or is_fully_converged
            # Update compressed counters (respects only_full_convergence flag)
            if should_count_compressed:
                total_predictions_compression += 1
                correct_predictions_compression += int(compression_is_correct)
                correct_predictions_compression_edge += int(compression_edge_is_correct)
                correct_predictions_compression_endings += int(compression_endings_is_correct)
                correct_predictions_compression_only += int(compression_only_is_correct)
                correct_predictions_compression_only_edge += int(compression_only_edge_is_correct)
                correct_predictions_compression_only_endings += int(compression_only_endings_is_correct)

            # Compute token/char counts for correct ending
            token_count = None
            char_count = None
            if 0 <= label < len(endings):
                correct_ending = endings[label]
                full_text = f"{context} {correct_ending}"
                token_count = count_text_tokens(tokenizer, full_text, add_special_tokens=add_special_tokens)
                char_count = count_text_characters(full_text)

                # Baseline token/char counts
                total_tokens_baseline += token_count
                total_characters_baseline += char_count
                if baseline_is_correct:
                    correct_tokens_baseline += token_count
                    correct_characters_baseline += char_count
                if baseline_endings_is_correct:
                    correct_tokens_baseline_endings += token_count

                # Compressed token/char counts (respects only_full_convergence flag)
                if should_count_compressed:
                    total_tokens_compression += token_count
                    total_characters_compression += char_count
                    if compression_is_correct:
                        correct_tokens_compression += token_count
                        correct_characters_compression += char_count
                    if compression_edge_is_correct:
                        correct_tokens_compression_edge += token_count
                    if compression_endings_is_correct:
                        correct_tokens_compression_endings += token_count
                    if compression_only_is_correct:
                        correct_tokens_compression_only += token_count
                    if compression_only_edge_is_correct:
                        correct_tokens_compression_only_edge += token_count
                    if compression_only_endings_is_correct:
                        correct_tokens_compression_only_endings += token_count

            # Store result (always save, regardless of convergence)
            result = {
                "sample_id": idx,
                "context": context,
                "endings": endings,
                "label": label,
                "convergence": convergence,
                "is_fully_converged": is_fully_converged,
                "counted_in_metrics": should_count_compressed,
                "lengths": {
                    "tokens": token_count,
                    "characters": char_count,
                },
                "baseline": {
                    "predicted_label": baseline_predicted_label,
                    "is_correct": baseline_is_correct,
                    "ppls": baseline_ppls,
                },
                "baseline_endings": {
                    "predicted_label": baseline_endings_predicted_label,
                    "is_correct": baseline_endings_is_correct,
                    "ppls": baseline_endings_ppls,
                },
                "compression": {
                    "predicted_label": compression_predicted_label,
                    "is_correct": compression_is_correct,
                    "ppls": compression_ppls,
                },
                "compression_edge": {
                    "predicted_label": compression_edge_predicted_label,
                    "is_correct": compression_edge_is_correct,
                    "ppls": compression_edge_ppls,
                },
                "compression_endings": {
                    "predicted_label": compression_endings_predicted_label,
                    "is_correct": compression_endings_is_correct,
                    "ppls": compression_endings_ppls,
                },
                "compression_only": {
                    "predicted_label": compression_only_predicted_label,
                    "is_correct": compression_only_is_correct,
                    "ppls": compression_only_ppls,
                },
                "compression_only_edge": {
                    "predicted_label": compression_only_edge_predicted_label,
                    "is_correct": compression_only_edge_is_correct,
                    "ppls": compression_only_edge_ppls,
                },
                "compression_only_endings": {
                    "predicted_label": compression_only_endings_predicted_label,
                    "is_correct": compression_only_endings_is_correct,
                    "ppls": compression_only_endings_ppls,
                },
            }

            # Add intervention results
            if args.intervention and batch_intervention_data:
                intervention_data = batch_intervention_data[sample_idx]
                if intervention_data is not None:
                    result.update(build_intervention_result(intervention_data, label, num_model_layers))

            results.append(result)

        # Print progress
        if (batch_idx + 1) % max(1, num_batches // 10) == 0 or (batch_idx + 1) == num_batches:
            baseline_accuracy = (
                correct_predictions_baseline / total_predictions_baseline if total_predictions_baseline > 0 else 0.0
            )
            compression_accuracy = (
                correct_predictions_compression / total_predictions_compression if total_predictions_compression > 0 else 0.0
            )
            print(
                f"Progress: {total_predictions_baseline}/{len(dataset)}, Baseline Accuracy: {baseline_accuracy:.4f}, "
                f"Compression Accuracy: {compression_accuracy:.4f} ({total_predictions_compression} samples)"
            )

    # Compute final accuracies for all 8 variants
    # Note: baseline variants use total_predictions_baseline, compression variants use total_predictions_compression

    # 1. Baseline
    baseline_accuracy = correct_predictions_baseline / total_predictions_baseline if total_predictions_baseline > 0 else 0.0
    baseline_token_accuracy = correct_tokens_baseline / total_tokens_baseline if total_tokens_baseline > 0 else 0.0
    baseline_char_accuracy = correct_characters_baseline / total_characters_baseline if total_characters_baseline > 0 else 0.0

    # 2. Baseline endings
    baseline_endings_accuracy = (
        correct_predictions_baseline_endings / total_predictions_baseline if total_predictions_baseline > 0 else 0.0
    )
    baseline_endings_token_accuracy = (
        correct_tokens_baseline_endings / total_tokens_baseline if total_tokens_baseline > 0 else 0.0
    )

    # 3. Compression
    compression_accuracy = (
        correct_predictions_compression / total_predictions_compression if total_predictions_compression > 0 else 0.0
    )
    compression_token_accuracy = correct_tokens_compression / total_tokens_compression if total_tokens_compression > 0 else 0.0
    compression_char_accuracy = (
        correct_characters_compression / total_characters_compression if total_characters_compression > 0 else 0.0
    )

    # 4. Compression edge
    compression_edge_accuracy = (
        correct_predictions_compression_edge / total_predictions_compression if total_predictions_compression > 0 else 0.0
    )
    compression_edge_token_accuracy = (
        correct_tokens_compression_edge / total_tokens_compression if total_tokens_compression > 0 else 0.0
    )

    # 5. Compression endings
    compression_endings_accuracy = (
        correct_predictions_compression_endings / total_predictions_compression if total_predictions_compression > 0 else 0.0
    )
    compression_endings_token_accuracy = (
        correct_tokens_compression_endings / total_tokens_compression if total_tokens_compression > 0 else 0.0
    )

    # 6. Compression only
    compression_only_accuracy = (
        correct_predictions_compression_only / total_predictions_compression if total_predictions_compression > 0 else 0.0
    )
    compression_only_token_accuracy = (
        correct_tokens_compression_only / total_tokens_compression if total_tokens_compression > 0 else 0.0
    )

    # 7. Compression only edge
    compression_only_edge_accuracy = (
        correct_predictions_compression_only_edge / total_predictions_compression if total_predictions_compression > 0 else 0.0
    )
    compression_only_edge_token_accuracy = (
        correct_tokens_compression_only_edge / total_tokens_compression if total_tokens_compression > 0 else 0.0
    )

    # 8. Compression only endings
    compression_only_endings_accuracy = (
        correct_predictions_compression_only_endings / total_predictions_compression
        if total_predictions_compression > 0
        else 0.0
    )
    compression_only_endings_token_accuracy = (
        correct_tokens_compression_only_endings / total_tokens_compression if total_tokens_compression > 0 else 0.0
    )

    # Build intervention summary
    intervention_summary = None
    if args.intervention:
        intervention_summary = build_intervention_summary(
            results,
            num_model_layers,
            skip_per_layer=args.skip_per_layer,
            skip_cumulative=args.skip_cumulative,
            skip_reverse_cumulative=args.skip_reverse_cumulative,
        )

    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    output_data = {
        "args": vars(args),
        "only_full_convergence": args.only_full_convergence,
        "total_samples": total_predictions_baseline,
        # 1. Baseline - full sequence PPL (context + ending)
        "baseline": {
            "accuracy": baseline_accuracy,
            "token_normalized_accuracy": baseline_token_accuracy,
            "char_normalized_accuracy": baseline_char_accuracy,
            "correct_predictions": correct_predictions_baseline,
            "total_predictions": total_predictions_baseline,
            "correct_tokens": correct_tokens_baseline,
            "total_tokens": total_tokens_baseline,
            "correct_characters": correct_characters_baseline,
            "total_characters": total_characters_baseline,
        },
        # 2. Baseline endings - PPL of endings only
        "baseline_endings": {
            "accuracy": baseline_endings_accuracy,
            "token_normalized_accuracy": baseline_endings_token_accuracy,
            "correct_predictions": correct_predictions_baseline_endings,
            "total_predictions": total_predictions_baseline,
            "correct_tokens": correct_tokens_baseline_endings,
            "total_tokens": total_tokens_baseline,
        },
        # 3. Compression - full sequence PPL with compression tokens prepended
        "compression": {
            "accuracy": compression_accuracy,
            "token_normalized_accuracy": compression_token_accuracy,
            "char_normalized_accuracy": compression_char_accuracy,
            "correct_predictions": correct_predictions_compression,
            "total_predictions": total_predictions_compression,
            "correct_tokens": correct_tokens_compression,
            "total_tokens": total_tokens_compression,
            "correct_characters": correct_characters_compression,
            "total_characters": total_characters_compression,
        },
        # 4. Compression edge - PPL of (context + ending) with compression, excluding compression tokens from PPL
        "compression_edge": {
            "accuracy": compression_edge_accuracy,
            "token_normalized_accuracy": compression_edge_token_accuracy,
            "correct_predictions": correct_predictions_compression_edge,
            "total_predictions": total_predictions_compression,
            "correct_tokens": correct_tokens_compression_edge,
            "total_tokens": total_tokens_compression,
        },
        # 5. Compression endings - PPL of endings only with compression tokens as context
        "compression_endings": {
            "accuracy": compression_endings_accuracy,
            "token_normalized_accuracy": compression_endings_token_accuracy,
            "correct_predictions": correct_predictions_compression_endings,
            "total_predictions": total_predictions_compression,
            "correct_tokens": correct_tokens_compression_endings,
            "total_tokens": total_tokens_compression,
        },
        # 6. Compression only - PPL with compression tokens instead of original context
        "compression_only": {
            "accuracy": compression_only_accuracy,
            "token_normalized_accuracy": compression_only_token_accuracy,
            "correct_predictions": correct_predictions_compression_only,
            "total_predictions": total_predictions_compression,
            "correct_tokens": correct_tokens_compression_only,
            "total_tokens": total_tokens_compression,
        },
        # 7. Compression only edge - PPL of endings only when replacing context with compression tokens
        "compression_only_edge": {
            "accuracy": compression_only_edge_accuracy,
            "token_normalized_accuracy": compression_only_edge_token_accuracy,
            "correct_predictions": correct_predictions_compression_only_edge,
            "total_predictions": total_predictions_compression,
            "correct_tokens": correct_tokens_compression_only_edge,
            "total_tokens": total_tokens_compression,
        },
        # 8. Compression only endings - PPL of endings with compression tokens (alternative method)
        "compression_only_endings": {
            "accuracy": compression_only_endings_accuracy,
            "token_normalized_accuracy": compression_only_endings_token_accuracy,
            "correct_predictions": correct_predictions_compression_only_endings,
            "total_predictions": total_predictions_compression,
            "correct_tokens": correct_tokens_compression_only_endings,
            "total_tokens": total_tokens_compression,
        },
        "results": results,
    }
    if intervention_summary is not None:
        output_data["intervention_summary"] = intervention_summary
        output_data["num_model_layers"] = num_model_layers
    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Total samples: {total_predictions_baseline}")
    print(f"Only full convergence: {args.only_full_convergence}")
    print(f"Samples with full convergence: {total_predictions_compression}/{total_predictions_baseline}")

    print("\n" + "-" * 70)
    print("BASELINE METHODS (no compression)")
    print("-" * 70)

    print("\n1. Baseline (full sequence PPL):")
    print(f"   Accuracy: {baseline_accuracy:.4f}")
    print(f"   Token-normalized: {baseline_token_accuracy:.4f}")
    print(f"   Char-normalized: {baseline_char_accuracy:.4f}")

    print("\n2. Baseline Endings (endings-only PPL):")
    print(f"   Accuracy: {baseline_endings_accuracy:.4f}")
    print(f"   Token-normalized: {baseline_endings_token_accuracy:.4f}")

    print("\n" + "-" * 70)
    print("COMPRESSION METHODS (compression tokens prepended to context)")
    print("-" * 70)

    print("\n3. Compression (full sequence PPL with compression):")
    print(f"   Accuracy: {compression_accuracy:.4f} (diff: {compression_accuracy - baseline_accuracy:+.4f})")
    print(f"   Token-normalized: {compression_token_accuracy:.4f}")
    print(f"   Char-normalized: {compression_char_accuracy:.4f}")

    print("\n4. Compression Edge (excluding compression tokens from PPL):")
    print(f"   Accuracy: {compression_edge_accuracy:.4f} (diff: {compression_edge_accuracy - baseline_accuracy:+.4f})")
    print(f"   Token-normalized: {compression_edge_token_accuracy:.4f}")

    print("\n5. Compression Endings (endings-only PPL with compression context):")
    print(
        f"   Accuracy: {compression_endings_accuracy:.4f} (diff: {compression_endings_accuracy - baseline_endings_accuracy:+.4f})"
    )
    print(f"   Token-normalized: {compression_endings_token_accuracy:.4f}")

    print("\n" + "-" * 70)
    print("COMPRESSION-ONLY METHODS (compression tokens replace context)")
    print("-" * 70)

    print("\n6. Compression Only (compression tokens + ending):")
    print(f"   Accuracy: {compression_only_accuracy:.4f} (diff: {compression_only_accuracy - baseline_accuracy:+.4f})")
    print(f"   Token-normalized: {compression_only_token_accuracy:.4f}")

    print("\n7. Compression Only Edge (endings-only PPL):")
    print(
        f"   Accuracy: {compression_only_edge_accuracy:.4f} (diff: {compression_only_edge_accuracy - baseline_accuracy:+.4f})"
    )
    print(f"   Token-normalized: {compression_only_edge_token_accuracy:.4f}")

    print("\n8. Compression Only Endings (alternative endings PPL):")
    print(
        f"   Accuracy: {compression_only_endings_accuracy:.4f} (diff: {compression_only_endings_accuracy - baseline_endings_accuracy:+.4f})"
    )
    print(f"   Token-normalized: {compression_only_endings_token_accuracy:.4f}")

    if args.intervention and intervention_summary:
        print_intervention_summary(intervention_summary, num_model_layers, baseline_accuracy)

    print(f"\nResults saved to: {results_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
