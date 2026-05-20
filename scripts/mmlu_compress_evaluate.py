"""Evaluate compression tokens on MMLU benchmark (generative).

This script:
1. Downloads MMLU dataset
2. Compresses the few-shot prefix (and optionally the full prompt) for each item
3. Generates the answer letter with compression tokens + question text in context
4. Compares generated answer to correct answer
"""

import argparse
import inspect
import json
import os
from typing import Optional

import torch
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from compression_horizon.train.loss import compute_hybrid_cross_entropy_and_alignment_loss
from compression_horizon.utils.launch import freeze_model_parameters, get_device, resolve_torch_dtype, set_launch_seed
from compression_horizon.utils.tokens import count_text_characters, count_text_tokens

ANSWER_LETTERS = ["A", "B", "C", "D"]


def format_few_shot_prefix(subject: str, few_shot_examples: list[dict]) -> str:
    """Format the few-shot prefix string (the part to compress).

    Returns a string like:
        The following are multiple choice questions about Abstract Algebra.

        Question: ...
        A. ... B. ... C. ... D. ...
        Answer: A

        Question: ...
        ...
    """
    subject_formatted = subject.replace("_", " ").title()
    lines = [f"The following are multiple choice questions about {subject_formatted}.\n"]
    for ex in few_shot_examples:
        lines.append(f"Question: {ex['question']}")
        for i, choice in enumerate(ex["choices"]):
            lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
        correct_letter = ANSWER_LETTERS[ex["answer"]]
        lines.append(f"Answer: {correct_letter}\n")
    return "\n".join(lines)


def format_question_suffix(question: str, choices: list[str]) -> str:
    """Format the question suffix (always kept as plain text).

    Returns a string like:
        Question: ...
        A. ... B. ... C. ... D. ...
        Answer:
    """
    lines = [f"Question: {question}"]
    for i, choice in enumerate(choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def format_full_prompt(subject: str, question: str, choices: list[str], few_shot_examples: list[dict]) -> str:
    """Format the full prompt (few-shot prefix + question suffix)."""
    prefix = format_few_shot_prefix(subject, few_shot_examples)
    suffix = format_question_suffix(question, choices)
    return prefix + suffix


def get_few_shot_examples(dev_dataset, subject: str, num_shots: int = 5) -> list[dict]:
    """Get few-shot examples for a given subject from the dev split."""
    subject_examples = [item for item in dev_dataset if item["subject"] == subject]
    examples = subject_examples[:num_shots]
    return [{"question": ex["question"], "choices": ex["choices"], "answer": ex["answer"]} for ex in examples]


def extract_answer_letter(text: str) -> Optional[str]:
    """Extract the answer letter (A/B/C/D) from generated text.

    Strips whitespace, looks for the first valid letter.
    """
    text = text.strip()
    for char in text:
        upper = char.upper()
        if upper in ANSWER_LETTERS:
            return upper
    return None


@torch.no_grad()
def generate_baseline_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 5,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> list[str]:
    """Generate answers from full text prompts (no compression).

    Returns list of generated text strings (only the new tokens).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    # Left-padding is required for correct batched generation with causal LMs
    tokenizer.padding_side = "left"

    encoded = tokenizer(prompts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Decode only the new tokens
    generated_texts = []
    for i in range(len(prompts)):
        prompt_len = input_ids[i].shape[0]
        new_tokens = output_ids[i, prompt_len:]
        generated_texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return generated_texts


@torch.no_grad()
def generate_with_compression_and_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    compression_embedding: torch.Tensor,
    text_suffix: str,
    max_new_tokens: int = 5,
    device: Optional[torch.device] = None,
    add_special_tokens: bool = True,
) -> str:
    """Generate answer with compression tokens prepended to text suffix.

    Layout: [compression_embedding] [text_suffix_tokens] → generate

    Args:
        compression_embedding: [num_compression_tokens, hidden_size]
        text_suffix: text to tokenize and append after compression tokens
        max_new_tokens: max tokens to generate

    Returns:
        Generated text string (only the new tokens).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    # Tokenize suffix
    encoded = tokenizer(text_suffix, truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    suffix_input_ids = encoded["input_ids"].to(device)  # [1, suffix_len]

    embed_fn = model.get_input_embeddings()
    torch_dtype = embed_fn.weight.dtype

    suffix_embeddings = embed_fn(suffix_input_ids)  # [1, suffix_len, hidden]
    comp_embeds = compression_embedding.unsqueeze(0).to(torch_dtype).to(device)  # [1, num_ct, hidden]
    # num_compression_tokens = comp_embeds.shape[1]
    # hidden_size = comp_embeds.shape[2]

    # Initial embeddings: compression + suffix
    current_embeddings = torch.cat([comp_embeds, suffix_embeddings], dim=1)  # [1, num_ct + suffix_len, hidden]
    current_attention = torch.ones((1, current_embeddings.shape[1]), dtype=torch.long, device=device)

    eos_token_id = tokenizer.eos_token_id
    generated_token_ids = []

    for _ in range(max_new_tokens):
        outputs = model(inputs_embeds=current_embeddings, attention_mask=current_attention)
        next_token_logits = outputs.logits[:, -1, :]  # [1, vocab]
        next_token_id = torch.argmax(next_token_logits, dim=-1)  # [1]

        generated_token_ids.append(next_token_id.item())

        # Stop if EOS
        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

        # Append new token embedding
        next_token_embedding = embed_fn(next_token_id.unsqueeze(0)).to(torch_dtype)  # [1, 1, hidden]
        current_embeddings = torch.cat([current_embeddings, next_token_embedding], dim=1)
        current_attention = torch.ones((1, current_embeddings.shape[1]), dtype=torch.long, device=device)

    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return generated_text


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
    init_embeddings: Optional[torch.Tensor] = None,
) -> list[dict[str, torch.Tensor | float]]:
    """Compress multiple text prefixes into compression tokens (batched).

    Args:
        model: The language model (frozen)
        tokenizer: Tokenizer
        texts: List of text prefixes to compress
        num_compression_tokens: Number of compression tokens
        max_steps: Maximum optimization steps
        learning_rate: Learning rate for optimization
        init_embeddings: Optional initial compression embeddings [num_compression_tokens, hidden_size]
            to use as warm start (cloned for each sample in the batch).

    Returns:
        List of dicts with keys: compression_embedding [num_compression_tokens, hidden_size], convergence float
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

    # Right-padding is required for compression training so that compression tokens
    # are prepended at correct positions before the actual content
    tokenizer.padding_side = "right"

    encoded = tokenizer(texts, padding="longest", truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        token_embeddings = model.get_input_embeddings()(input_ids)

    hidden_size = token_embeddings.shape[-1]

    _needs_token_type_ids = "token_type_ids" in inspect.signature(model.forward).parameters

    embedding_dtype = token_embeddings.dtype
    compression_dtype = embedding_dtype

    # Initialize compression tokens
    if init_embeddings is not None:
        # Warm start: clone the provided embeddings for each sample
        init_embeds = init_embeddings.detach().clone().to(compression_dtype).to(device)
        compression_token_embeddings = torch.nn.Parameter(init_embeds.unsqueeze(0).expand(batch_size, -1, -1).clone())
    else:
        compression_token_embeddings = torch.nn.Parameter(
            torch.randn([batch_size, num_compression_tokens, hidden_size], dtype=compression_dtype, device=device) * 0.02
        )

    optimizer = AdamW([compression_token_embeddings], lr=learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps,
    )

    model.train()

    for _ in range(max_steps):
        optimizer.zero_grad()

        united_token_embeddings_list = []
        united_attention_mask_list = []
        labels_list = []
        for i in range(batch_size):
            seq_len = int(attention_mask[i].sum().item())
            sample_token_embeddings = token_embeddings[i : i + 1, :seq_len]
            sample_attention_mask = attention_mask[i : i + 1, :seq_len]
            sample_compression_token_embeddings = compression_token_embeddings[i : i + 1]
            united_token_embeddings = torch.cat([sample_compression_token_embeddings, sample_token_embeddings], dim=1)
            united_attention_mask = torch.cat(
                [
                    torch.ones((1, num_compression_tokens), dtype=sample_attention_mask.dtype, device=device),
                    sample_attention_mask,
                ],
                dim=1,
            )
            united_token_embeddings_list.append(united_token_embeddings)
            united_attention_mask_list.append(united_attention_mask)
            sample_input_ids = input_ids[i : i + 1, :seq_len]
            sample_labels = sample_input_ids.clone()
            sample_labels[sample_attention_mask == 0] = -100
            labels_list.append(sample_labels)

        max_len = max(item.shape[1] for item in united_token_embeddings_list)
        batch_embeddings = []
        batch_attention = []
        batch_labels = []
        for i in range(batch_size):
            ute = united_token_embeddings_list[i]
            uam = united_attention_mask_list[i]
            labels = labels_list[i]
            current_len = ute.shape[1]
            if current_len < max_len:
                pad_len = max_len - current_len
                ute = torch.cat([ute, torch.zeros(1, pad_len, hidden_size, dtype=ute.dtype, device=device)], dim=1)
                uam = torch.cat([uam, torch.zeros(1, pad_len, dtype=uam.dtype, device=device)], dim=1)
                labels = torch.cat([labels, torch.full((1, pad_len), -100, dtype=labels.dtype, device=device)], dim=1)
            batch_embeddings.append(ute)
            batch_attention.append(uam)
            batch_labels.append(labels)
        batch_embeddings = torch.cat(batch_embeddings, dim=0)
        batch_attention = torch.cat(batch_attention, dim=0)
        batch_labels = torch.cat(batch_labels, dim=0)

        target_outputs = None
        target_fwd_kwargs = {}
        compression_fwd_kwargs = {}
        if _needs_token_type_ids:
            target_fwd_kwargs["token_type_ids"] = torch.zeros(attention_mask.shape, dtype=torch.long, device=device)
            compression_fwd_kwargs["token_type_ids"] = torch.zeros(batch_attention.shape, dtype=torch.long, device=device)

        if use_alignment:
            with torch.no_grad():
                target_outputs = model(
                    inputs_embeds=token_embeddings,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    **target_fwd_kwargs,
                )

        compression_outputs = model(
            inputs_embeds=batch_embeddings,
            attention_mask=batch_attention,
            output_hidden_states=use_alignment,
            **compression_fwd_kwargs,
        )

        total_loss = 0.0
        for i in range(batch_size):
            seq_len = int(attention_mask[i].sum().item())
            sample_logits = compression_outputs.logits[i : i + 1, : num_compression_tokens + seq_len]
            sample_input_ids = input_ids[i : i + 1, :seq_len]
            sample_attention_mask = attention_mask[i : i + 1, :seq_len]
            if use_alignment:
                assert target_outputs is not None
                sample_compression_hidden_states = tuple(
                    hs_layer[i : i + 1, : num_compression_tokens + seq_len] for hs_layer in compression_outputs.hidden_states
                )
                sample_target_hidden_states = tuple(hs_layer[i : i + 1, :seq_len] for hs_layer in target_outputs.hidden_states)
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
    convergences = calculate_convergence(model, batch_embeddings, batch_attention, batch_labels, num_compression_tokens)
    print("Batch convergences:", convergences)
    return [
        {
            "compression_embedding": compression_token_embeddings[i].detach(),
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
    """Calculate token-level accuracy between predicted and target tokens."""
    outputs = model(inputs_embeds=batch_embeddings, attention_mask=batch_attention)
    batch_size = batch_embeddings.shape[0]
    convergences = []
    for i in range(batch_size):
        seq_len = int(batch_attention[i].sum().item())
        orig_seq_len = seq_len - num_compression_tokens

        if orig_seq_len <= 0:
            convergences.append(0.0)
            continue

        sample_logits = outputs.logits[i, num_compression_tokens - 1 : seq_len - 1]
        sample_predicted_tokens = sample_logits.argmax(dim=-1)
        sample_labels = batch_labels[i, :orig_seq_len]

        convergence = (sample_predicted_tokens == sample_labels).float().mean().item()
        convergences.append(convergence)
    return convergences


def main():
    parser = argparse.ArgumentParser(description="Evaluate compression tokens on MMLU benchmark (generative)")
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
        help="Loss type for optimization.",
    )
    parser.add_argument(
        "--num_alignment_layers",
        type=int,
        default=0,
        help="Number of layers to align (0 = all layers).",
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
        "--no_bos_token",
        action="store_true",
        default=False,
        help="Disable BOS token insertion during tokenization.",
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
        default="artifacts/mmlu_evaluation",
        help="Output directory for results",
    )
    # MMLU-specific arguments
    parser.add_argument(
        "--subject",
        type=str,
        default="all",
        help="MMLU subject to evaluate, or 'all' for all subjects.",
    )
    parser.add_argument(
        "--num_few_shot",
        type=int,
        default=5,
        help="Number of few-shot examples from dev split.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=5,
        help="Maximum new tokens to generate for the answer.",
    )
    parser.add_argument(
        "--compression_mode",
        type=str,
        default="prefix_only",
        choices=["prefix_only", "full_prompt", "random"],
        help="Compression mode: 'prefix_only' compresses few-shot prefix only (shared per subject), "
        "'full_prompt' compresses entire prompt per sample (warm-started from prefix compression), "
        "'random' uses random normal embeddings instead of optimized compression tokens.",
    )
    args = parser.parse_args()

    set_launch_seed(args.random_seed)
    torch_dtype = resolve_torch_dtype(args.dtype)
    device = get_device()

    print(f"Loading model from {args.model_checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, torch_dtype=torch_dtype)
    print("Loaded model dtype:", next(model.parameters()).dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_bos_supported = hasattr(tokenizer, "add_bos_token")
    if args.no_bos_token and add_bos_supported:
        tokenizer.add_bos_token = False
    add_special_tokens = not (args.no_bos_token and not add_bos_supported)

    # Load MMLU dataset
    print("Loading MMLU dataset...")
    if args.subject == "all":
        dataset = load_dataset("cais/mmlu", "all", split="test")
        dev_dataset = load_dataset("cais/mmlu", "all", split="dev")
    else:
        dataset = load_dataset("cais/mmlu", args.subject, split="test")
        dev_dataset = load_dataset("cais/mmlu", args.subject, split="dev")

    if args.limit_samples:
        dataset = dataset.select(range(min(args.limit_samples, len(dataset))))
    print(f"Evaluating MMLU benchmark on {len(dataset)} samples")

    os.makedirs(args.output_dir, exist_ok=True)

    # Cache few-shot examples per subject
    few_shot_cache: dict[str, list[dict]] = {}
    # Cache compressed prefix embeddings per subject (for prefix_only and full_prompt warm start)
    prefix_compression_cache: dict[str, torch.Tensor] = {}

    # Evaluation counters
    results = []
    total_predictions_baseline = 0
    correct_predictions_baseline = 0
    valid_predictions_baseline = 0
    total_tokens_baseline = 0
    total_characters_baseline = 0
    correct_tokens_baseline = 0
    correct_characters_baseline = 0

    total_predictions_compressed = 0
    correct_predictions_compressed = 0
    valid_predictions_compressed = 0
    total_tokens_compressed = 0
    total_characters_compressed = 0
    correct_tokens_compressed = 0
    correct_characters_compressed = 0

    # Per-subject tracking
    per_subject_stats: dict[str, dict] = {}

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_items = [dataset[i] for i in range(start_idx, end_idx)]
        actual_batch_size = len(batch_items)

        # Build prompts and suffixes
        batch_subjects = [item["subject"] for item in batch_items]
        batch_questions = [item["question"] for item in batch_items]
        batch_choices = [item["choices"] for item in batch_items]
        batch_answers = [int(item["answer"]) for item in batch_items]

        # Ensure few-shot examples are cached
        for subj in set(batch_subjects):
            if subj not in few_shot_cache:
                few_shot_cache[subj] = get_few_shot_examples(dev_dataset, subj, args.num_few_shot)

        # Build full prompts for baseline
        batch_full_prompts = []
        batch_suffixes = []
        for i in range(actual_batch_size):
            subj = batch_subjects[i]
            few_shots = few_shot_cache[subj]
            full_prompt = format_full_prompt(subj, batch_questions[i], batch_choices[i], few_shots)
            suffix = format_question_suffix(batch_questions[i], batch_choices[i])
            batch_full_prompts.append(full_prompt)
            batch_suffixes.append(suffix)

        # --- Baseline generation ---
        try:
            baseline_generated = generate_baseline_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_full_prompts,
                max_new_tokens=args.max_new_tokens,
                device=device,
                add_special_tokens=add_special_tokens,
            )
        except Exception as e:
            print(f"Error in baseline generation for batch {batch_idx}: {e}")
            baseline_generated = [""] * actual_batch_size

        # --- Compression ---
        if args.compression_mode == "random":
            # Random mode: use random normal embeddings matching the scale of real embeddings
            embed_weight = model.get_input_embeddings().weight
            hidden_size = embed_weight.shape[1]
            embedding_dtype = embed_weight.dtype
            embed_std = embed_weight.detach().float().std().item()
            batch_compression_embeddings = [
                torch.randn(args.num_compression_tokens, hidden_size, dtype=embedding_dtype, device=device) * embed_std
                for _ in range(actual_batch_size)
            ]
            batch_convergences = [0.0] * actual_batch_size
        else:
            # Ensure prefix compression is cached for all subjects in this batch
            for subj in set(batch_subjects):
                if subj not in prefix_compression_cache:
                    few_shots = few_shot_cache[subj]
                    prefix_text = format_few_shot_prefix(subj, few_shots)
                    print(f"Compressing prefix for subject: {subj}")
                    try:
                        prefix_results = compress_prefixes_batch(
                            model=model,
                            tokenizer=tokenizer,
                            texts=[prefix_text],
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
                        prefix_compression_cache[subj] = prefix_results[0]["compression_embedding"]
                    except Exception as e:
                        print(f"Error compressing prefix for subject {subj}: {e}")
                        prefix_compression_cache[subj] = None

            # Determine compression embeddings per sample
            if args.compression_mode == "prefix_only":
                # Use cached prefix compression directly
                batch_compression_embeddings = []
                batch_convergences = []
                for i in range(actual_batch_size):
                    subj = batch_subjects[i]
                    emb = prefix_compression_cache.get(subj)
                    batch_compression_embeddings.append(emb)
                    batch_convergences.append(1.0 if emb is not None else 0.0)
            else:
                # full_prompt mode: compress full prompt with warm start from prefix
                batch_texts_to_compress = batch_full_prompts
                # Get warm-start embeddings per sample
                init_emb = None
                # Use the first subject's cached embedding as warm start
                # (in practice, batch may have mixed subjects; use per-sample init)
                # We compress the batch, using a single init for simplicity
                # For mixed subjects, we use the first available cached embedding
                first_subj = batch_subjects[0]
                init_emb = prefix_compression_cache.get(first_subj)

                try:
                    full_results = compress_prefixes_batch(
                        model=model,
                        tokenizer=tokenizer,
                        texts=batch_texts_to_compress,
                        num_compression_tokens=args.num_compression_tokens,
                        max_steps=args.max_optimization_steps,
                        learning_rate=args.learning_rate,
                        loss_type=args.loss_type,
                        hybrid_alpha=args.hybrid_alpha,
                        num_alignment_layers=args.num_alignment_layers,
                        inverted_alignment=args.inverted_alignment,
                        device=device,
                        add_special_tokens=add_special_tokens,
                        init_embeddings=init_emb,
                    )
                    batch_compression_embeddings = [r["compression_embedding"] for r in full_results]
                    batch_convergences = [r["convergence"] for r in full_results]
                except Exception as e:
                    print(f"Error compressing full prompts for batch {batch_idx}: {e}")
                    batch_compression_embeddings = [None] * actual_batch_size
                    batch_convergences = [0.0] * actual_batch_size

        # --- Compressed generation ---
        compressed_generated = []
        for i in range(actual_batch_size):
            emb = batch_compression_embeddings[i]
            if emb is None:
                compressed_generated.append("")
                continue
            try:
                gen_text = generate_with_compression_and_text(
                    model=model,
                    tokenizer=tokenizer,
                    compression_embedding=emb,
                    text_suffix=batch_full_prompts[i],
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    add_special_tokens=add_special_tokens,
                )
                compressed_generated.append(gen_text)
            except Exception as e:
                print(f"Error in compressed generation for sample {start_idx + i}: {e}")
                compressed_generated.append("")

        # --- Accumulate results ---
        for i in range(actual_batch_size):
            idx = start_idx + i
            subject = batch_subjects[i]
            question = batch_questions[i]
            choices = batch_choices[i]
            correct_idx = batch_answers[i]
            correct_letter = ANSWER_LETTERS[correct_idx]

            # Initialize per-subject stats
            if subject not in per_subject_stats:
                per_subject_stats[subject] = {
                    "baseline_correct": 0,
                    "compressed_correct": 0,
                    "baseline_valid": 0,
                    "compressed_valid": 0,
                    "total": 0,
                }

            # Baseline
            baseline_text = baseline_generated[i]
            baseline_letter = extract_answer_letter(baseline_text)
            baseline_is_valid = baseline_letter is not None
            baseline_is_correct = baseline_letter == correct_letter

            # Compressed
            compressed_text = compressed_generated[i]
            compressed_letter = extract_answer_letter(compressed_text)
            compressed_is_valid = compressed_letter is not None
            compressed_is_correct = compressed_letter == correct_letter
            convergence = batch_convergences[i]

            # Token/char counts
            full_prompt = batch_full_prompts[i]
            token_count = count_text_tokens(tokenizer, full_prompt, add_special_tokens=add_special_tokens)
            char_count = count_text_characters(full_prompt)

            # Update baseline counters
            total_predictions_baseline += 1
            if baseline_is_valid:
                valid_predictions_baseline += 1
            if baseline_is_correct:
                correct_predictions_baseline += 1
            total_tokens_baseline += token_count
            total_characters_baseline += char_count
            if baseline_is_correct:
                correct_tokens_baseline += token_count
                correct_characters_baseline += char_count

            # Update compressed counters
            total_predictions_compressed += 1
            if compressed_is_valid:
                valid_predictions_compressed += 1
            if compressed_is_correct:
                correct_predictions_compressed += 1
            total_tokens_compressed += token_count
            total_characters_compressed += char_count
            if compressed_is_correct:
                correct_tokens_compressed += token_count
                correct_characters_compressed += char_count

            # Per-subject
            per_subject_stats[subject]["total"] += 1
            if baseline_is_correct:
                per_subject_stats[subject]["baseline_correct"] += 1
            if compressed_is_correct:
                per_subject_stats[subject]["compressed_correct"] += 1
            if baseline_is_valid:
                per_subject_stats[subject]["baseline_valid"] += 1
            if compressed_is_valid:
                per_subject_stats[subject]["compressed_valid"] += 1

            result = {
                "sample_id": idx,
                "subject": subject,
                "question": question,
                "choices": choices,
                "correct_answer_index": correct_idx,
                "correct_answer_letter": correct_letter,
                "lengths": {
                    "tokens": token_count,
                    "characters": char_count,
                },
                "baseline": {
                    "generated_text": baseline_text,
                    "predicted_letter": baseline_letter,
                    "is_correct": baseline_is_correct,
                    "is_valid": baseline_is_valid,
                },
                "compressed": {
                    "generated_text": compressed_text,
                    "predicted_letter": compressed_letter,
                    "is_correct": compressed_is_correct,
                    "is_valid": compressed_is_valid,
                    "convergence": convergence,
                },
            }
            results.append(result)

        # Print progress
        if (batch_idx + 1) % max(1, num_batches // 10) == 0 or (batch_idx + 1) == num_batches:
            baseline_accuracy = (
                correct_predictions_baseline / total_predictions_baseline if total_predictions_baseline > 0 else 0.0
            )
            compressed_accuracy = (
                correct_predictions_compressed / total_predictions_compressed if total_predictions_compressed > 0 else 0.0
            )
            print(
                f"Progress: {total_predictions_baseline}/{len(dataset)}, "
                f"Baseline Accuracy: {baseline_accuracy:.4f} ({valid_predictions_baseline} valid), "
                f"Compressed Accuracy: {compressed_accuracy:.4f} ({valid_predictions_compressed} valid)"
            )

    # Compute final accuracies
    baseline_accuracy = correct_predictions_baseline / total_predictions_baseline if total_predictions_baseline > 0 else 0.0
    compressed_accuracy = (
        correct_predictions_compressed / total_predictions_compressed if total_predictions_compressed > 0 else 0.0
    )
    baseline_token_accuracy = correct_tokens_baseline / total_tokens_baseline if total_tokens_baseline > 0 else 0.0
    compressed_token_accuracy = correct_tokens_compressed / total_tokens_compressed if total_tokens_compressed > 0 else 0.0
    baseline_char_accuracy = correct_characters_baseline / total_characters_baseline if total_characters_baseline > 0 else 0.0
    compressed_char_accuracy = (
        correct_characters_compressed / total_characters_compressed if total_characters_compressed > 0 else 0.0
    )

    # Build per-subject summary
    per_subject_summary = {}
    for subj, stats in per_subject_stats.items():
        total = stats["total"]
        per_subject_summary[subj] = {
            "baseline_accuracy": stats["baseline_correct"] / total if total > 0 else 0.0,
            "compressed_accuracy": stats["compressed_correct"] / total if total > 0 else 0.0,
            "baseline_valid": stats["baseline_valid"],
            "compressed_valid": stats["compressed_valid"],
            "total": total,
        }

    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    output_data = {
        "args": vars(args),
        "baseline": {
            "accuracy": baseline_accuracy,
            "token_normalized_accuracy": baseline_token_accuracy,
            "char_normalized_accuracy": baseline_char_accuracy,
            "correct_predictions": correct_predictions_baseline,
            "valid_predictions": valid_predictions_baseline,
            "total_predictions": total_predictions_baseline,
            "total_tokens": total_tokens_baseline,
            "total_characters": total_characters_baseline,
        },
        "compressed": {
            "accuracy": compressed_accuracy,
            "token_normalized_accuracy": compressed_token_accuracy,
            "char_normalized_accuracy": compressed_char_accuracy,
            "correct_predictions": correct_predictions_compressed,
            "valid_predictions": valid_predictions_compressed,
            "total_predictions": total_predictions_compressed,
            "total_tokens": total_tokens_compressed,
            "total_characters": total_characters_compressed,
        },
        "per_subject": per_subject_summary,
        "results": results,
    }
    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Total samples: {total_predictions_baseline}")
    print(f"Compression mode: {args.compression_mode}")
    print(f"Subjects: {len(per_subject_stats)}")
    print("\nBaseline (without compression):")
    print(f"  Correct predictions: {correct_predictions_baseline}/{total_predictions_baseline}")
    print(f"  Valid predictions: {valid_predictions_baseline}/{total_predictions_baseline}")
    print(f"  Accuracy: {baseline_accuracy:.4f}")
    print(f"  Token-normalized Accuracy: {baseline_token_accuracy:.4f}")
    print(f"  Character-normalized Accuracy: {baseline_char_accuracy:.4f}")
    print("\nCompressed (with compression tokens):")
    print(f"  Correct predictions: {correct_predictions_compressed}/{total_predictions_compressed}")
    print(f"  Valid predictions: {valid_predictions_compressed}/{total_predictions_compressed}")
    print(f"  Accuracy: {compressed_accuracy:.4f}")
    print(f"  Token-normalized Accuracy: {compressed_token_accuracy:.4f}")
    print(f"  Character-normalized Accuracy: {compressed_char_accuracy:.4f}")
    print(f"\nDifference: {compressed_accuracy - baseline_accuracy:+.4f}")
    print(f"\nResults saved to: {results_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
