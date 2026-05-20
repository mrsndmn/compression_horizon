import hashlib
import json
import os
import shutil
import subprocess
import sys

import transformers
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from compression_horizon.train import (
    CompressionHeadTrainer,
    FullCrammingTrainer,
    LowDimTrainer,
    PrefixTuningTrainer,
    ProgressiveCrammingTrainer,
)
from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.utils.exceptions import NvidiaSMIError
from compression_horizon.utils.launch import resolve_torch_dtype, set_launch_seed


def load_or_create_tokenized_dataset(
    cache_dir: str,
    dataset_name: str,
    split: str,
    tokenizer: AutoTokenizer,
    max_sequence_length: int,
    model_checkpoint: str,
    no_bos_token: bool = False,
    limit_dataset_items: int | None = None,
    offset_dataset_items: int | None = None,
    cache_prefix: str = "dataset",
    num_proc: int = 4,
    fallback_length: int | None = None,
) -> Dataset:
    """
    Load a tokenized dataset from cache or create and cache it.

    Args:
        cache_dir: Directory for caching datasets
        dataset_name: Name of the dataset (e.g., "mrsndmn/pg19")
        split: Dataset split (e.g., "test")
        tokenizer: Tokenizer to use for tokenization
        max_sequence_length: Maximum sequence length for tokenization
        model_checkpoint: Model checkpoint name (for cache key)
        no_bos_token: Disable BOS token insertion during tokenization
        limit_dataset_items: Optional limit on number of items to select
        offset_dataset_items: Optional offset for dataset items selection (applied before limit)
        cache_prefix: Prefix for cache file name (default: "dataset")
        num_proc: Number of processes for dataset loading
        fallback_length: If provided and limit_dataset_items is None, use this length

    Returns:
        Tokenized Dataset
    """
    # Generate cache key based on dataset parameters
    cache_params = {
        "dataset": dataset_name,
        "split": split,
        "limit_dataset_items": limit_dataset_items,
        "offset_dataset_items": offset_dataset_items,
        "max_sequence_length": max_sequence_length,
        "model_checkpoint": model_checkpoint,
        "no_bos_token": no_bos_token,
    }
    cache_key_json = json.dumps(cache_params, sort_keys=True, ensure_ascii=False, default=str)
    cache_key_hash = hashlib.sha256(cache_key_json.encode("utf-8")).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"{cache_prefix}_{cache_key_hash}")

    # Try to load cached tokenized dataset
    if os.path.exists(cache_path):
        print(f"Loading tokenized dataset from cache: {cache_path}")
        ds = Dataset.load_from_disk(cache_path)
        return ds.with_format("torch")

    # Create dataset if not cached
    print("Tokenizing dataset (this may take a while)...")
    if dataset_name in ["mrsndmn/pg19-model-sampled-llama3.1-8B-prefix-64-max_len-2048"]:
        split = "train"

    kwargs = {
        "num_proc": num_proc,
        "split": split,
    }
    if dataset_name == "HuggingFaceFW/fineweb-edu":
        # split = "sample-10BT"
        del kwargs["num_proc"]
        del kwargs["split"]
        kwargs["data_files"] = [f"sample/10BT/{i:03}_00000.parquet" for i in range(14)]
        # kwargs['streaming'] = True
    elif dataset_name == "LarryLovestein/pg19_1k":
        kwargs["split"] = "train"
    elif dataset_name == "LarryLovestein/fanfics_1k":
        kwargs["split"] = "train"

    raw_dataset = load_dataset(dataset_name, **kwargs)

    if dataset_name == "HuggingFaceFW/fineweb-edu":
        raw_dataset = raw_dataset["train"]

    # Apply offset and limit
    if offset_dataset_items is not None:
        start_idx = offset_dataset_items
        if limit_dataset_items is not None:
            end_idx = start_idx + limit_dataset_items
            dataset = raw_dataset.select(range(start_idx, end_idx))
        elif fallback_length is not None:
            end_idx = start_idx + fallback_length
            dataset = raw_dataset.select(range(start_idx, end_idx))
        else:
            # If only offset is provided, select from offset to end
            dataset = raw_dataset.select(range(start_idx, len(raw_dataset)))
    elif limit_dataset_items is not None:
        dataset = raw_dataset.select(range(limit_dataset_items))
    elif fallback_length is not None:
        dataset = raw_dataset.select(range(fallback_length))
    else:
        dataset = raw_dataset

    add_bos_supported = hasattr(tokenizer, "add_bos_token")
    original_add_bos = getattr(tokenizer, "add_bos_token", None)
    if no_bos_token and add_bos_supported:
        tokenizer.add_bos_token = False

    def _tokenize(example):
        # Important: do NOT use return_tensors="pt" here.
        # HF Datasets stores tensors as nested lists like [1, T] which makes __getitem__ very slow.
        # We'll instead store plain lists and set .with_format("torch") on the resulting dataset.
        add_special_tokens = True
        if no_bos_token and not add_bos_supported:
            add_special_tokens = False
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length,
            add_special_tokens=add_special_tokens,
        )

    dataset = dataset.map(
        _tokenize,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    if no_bos_token and add_bos_supported:
        tokenizer.add_bos_token = original_add_bos

    # Save tokenized dataset to cache
    print(f"Saving tokenized dataset to cache: {cache_path}")
    dataset.save_to_disk(cache_path)

    return dataset.with_format("torch")


if __name__ == "__main__":
    # Check for nvidia-smi availability
    try:
        subprocess.check_output(["nvidia-smi"], shell=True)
    except subprocess.CalledProcessError:
        raise NvidiaSMIError("nvidia-smi is not available")

    # Parse command-line arguments and defaults
    hf_parser = transformers.HfArgumentParser(MyTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    # Determine output directory:
    # - If user provided --output_dir, respect it.
    # - Otherwise, construct: artifacts/{experiments|experiments_progressive|experiments_prefix_tuning}/
    #   ch_{essential_params}_{hash8}, where hash8 is derived from training args.
    if getattr(training_args, "train_compression_head", False):
        default_base = "artifacts/experiments_compression_head"
        default_base_in_progress = "artifacts/experiments_in_progress"
    elif training_args.progressive_train:
        default_base = "artifacts/experiments_progressive"
        default_base_in_progress = "artifacts/experiments_in_progress"
    elif getattr(training_args, "train_prefix_tuning", False):
        default_base = "artifacts/experiments_prefix_tuning"
        default_base_in_progress = "artifacts/experiments_in_progress"
    else:
        default_base = "artifacts/experiments"
        default_base_in_progress = "artifacts/experiments_in_progress"
    os.makedirs(default_base, exist_ok=True)
    os.makedirs(default_base_in_progress, exist_ok=True)
    # Build short, human-readable prefix
    loss_type = training_args.loss_type
    hybrid_alpha = training_args.hybrid_alpha

    if getattr(training_args, "train_compression_head", False):
        prefix = f"ch_head_seq_len_{training_args.max_sequence_length}"
    elif training_args.progressive_train:
        prefix = f"ch_{loss_type}_init_{training_args.embedding_init_method}_seq_len_{training_args.max_sequence_length}"
    elif getattr(training_args, "train_prefix_tuning", False):
        prefix = (
            f"ch_prefix_tuning_{loss_type}_hybrid_alpha_{hybrid_alpha}_init_{training_args.embedding_init_method}"
            f"_seq_len_{training_args.max_sequence_length}"
        )
    else:
        prefix = (
            f"ch_{loss_type}_hybrid_alpha_{hybrid_alpha}_init_{training_args.embedding_init_method}"
            f"_seq_len_{training_args.max_sequence_length}"
        )

    # Compute stable hash from training arguments (excluding volatile dirs)
    args_dict = training_args.to_dict()
    args_dict.pop("output_dir", None)
    args_dict.pop("logging_dir", None)
    args_json = json.dumps(args_dict, sort_keys=True, ensure_ascii=False, default=str)

    # Detect if user explicitly passed --output_dir on the command line
    argv = sys.argv[1:]
    user_provided_output_dir = False
    for i, token in enumerate(argv):
        if token == "--output_dir" and i + 1 < len(argv):
            user_provided_output_dir = True
            break
        if token.startswith("--output_dir="):
            user_provided_output_dir = True
            break

    if user_provided_output_dir and getattr(training_args, "output_dir", None):
        output_dir = training_args.output_dir
        # Detect if this is in the in_progress directory
        is_in_progress = "experiments_in_progress" in output_dir
        if is_in_progress:
            # Compute the final directory by replacing experiments_in_progress with the appropriate final base
            output_dir_final = output_dir.replace("experiments_in_progress", default_base.split("/")[-1])
        else:
            output_dir_final = None
    else:
        output_dir = os.path.join(default_base, f"{prefix}")
        output_dir_final = None

    print("output_dir", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir
    # Also persist raw CLI (excluding --output_dir) and its hash for auditability
    filtered_argv: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--output_dir":
            skip_next = True
            continue
        if token.startswith("--output_dir="):
            continue
        filtered_argv.append(token)
    cmdline_str = " ".join(filtered_argv).strip()
    cmd_hash8 = hashlib.sha1(cmdline_str.encode("utf-8")).hexdigest()[:8]
    with open(os.path.join(output_dir, "cmd.txt"), "w", encoding="utf-8") as f:
        f.write(cmdline_str + "\n")
    with open(os.path.join(output_dir, "cmd_hash.txt"), "w", encoding="utf-8") as f:
        f.write(cmd_hash8 + "\n")

    random_seed = getattr(training_args, "random_seed", 42)
    set_launch_seed(random_seed)
    print(f"Random seed set to: {random_seed}")

    torch_dtype = resolve_torch_dtype(getattr(training_args, "dtype", "float32"))
    print("torch_dtype", torch_dtype)
    if training_args.train_compression_head or "experiments_compression_head/ch_head_" in training_args.model_checkpoint:
        from compression_horizon.models.llama_compression_head import LlamaForCausalLMCompressionHead

        model = LlamaForCausalLMCompressionHead.from_pretrained(
            training_args.model_checkpoint, torch_dtype=torch_dtype, attn_implementation="flash_attention_2"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            training_args.model_checkpoint, torch_dtype=torch_dtype, attn_implementation="flash_attention_2"
        )
        for p in model.parameters():
            p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(training_args.model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Create cache directory for tokenized datasets
    cache_dir = "artifacts/cache/tokenized_datasets"
    os.makedirs(cache_dir, exist_ok=True)

    # Load or create training dataset
    train_dataset = load_or_create_tokenized_dataset(
        cache_dir=cache_dir,
        dataset_name=training_args.dataset_name,
        split="test",
        tokenizer=tokenizer,
        max_sequence_length=training_args.max_sequence_length,
        model_checkpoint=training_args.model_checkpoint,
        no_bos_token=training_args.no_bos_token,
        limit_dataset_items=getattr(training_args, "limit_dataset_items", None),
        offset_dataset_items=getattr(training_args, "offset_dataset_items", None),
        cache_prefix="dataset",
    )

    print("train_dataset", len(train_dataset))
    print("train_dataset", train_dataset)

    # Prepare evaluation dataset with twice the sequence length for noop_train
    eval_dataset = None
    if training_args.noop_train:
        eval_seq_length = training_args.max_sequence_length * 2
        print(f"Preparing evaluation dataset with sequence length {eval_seq_length} (2x training length)...")

        eval_dataset = load_or_create_tokenized_dataset(
            cache_dir=cache_dir,
            dataset_name=training_args.dataset_name,
            split="test",
            tokenizer=tokenizer,
            max_sequence_length=eval_seq_length,
            model_checkpoint=training_args.model_checkpoint,
            no_bos_token=training_args.no_bos_token,
            limit_dataset_items=getattr(training_args, "limit_dataset_items", None),
            offset_dataset_items=getattr(training_args, "offset_dataset_items", None),
            cache_prefix="eval_dataset",
            fallback_length=len(train_dataset),
        )

        print(f"eval_dataset length: {len(eval_dataset)}")
        print(f"eval_dataset sequence length: {eval_seq_length}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train: select trainer class from args and call train()
    transformers.logging.set_verbosity_info()
    if getattr(training_args, "train_compression_head", False):
        trainer_cls = CompressionHeadTrainer
    elif training_args.progressive_train:
        trainer_cls = ProgressiveCrammingTrainer
    elif training_args.low_dim_train:
        trainer_cls = LowDimTrainer
    elif getattr(training_args, "train_prefix_tuning", False):
        trainer_cls = PrefixTuningTrainer
    else:
        trainer_cls = FullCrammingTrainer

    trainer = trainer_cls(
        model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    training_artifacts = trainer.train()
    print(f"Saved compressed prefixes to: {training_artifacts}.")

    # Move from experiments_in_progress to final directory after successful training
    if output_dir_final:
        print(f"\n{'='*80}")
        print("Training completed successfully!")
        print("Moving results from in-progress to final location:")
        print(f"  From: {output_dir}")
        print(f"  To:   {output_dir_final}")
        print(f"{'='*80}\n")

        # Ensure parent directory of final location exists
        os.makedirs(os.path.dirname(output_dir_final), exist_ok=True)

        # If final directory already exists, remove it first
        if os.path.exists(output_dir_final):
            print(f"Removing existing final directory: {output_dir_final}")
            shutil.rmtree(output_dir_final)

        # Move the directory
        shutil.move(output_dir, output_dir_final)
        print(f"Successfully moved experiment results to: {output_dir_final}")
    else:
        print(f"Training completed. Results saved at: {output_dir}")
