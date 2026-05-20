"""Clean training entry-point for thesis-reproduction experiments."""

from __future__ import annotations

import os

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from compression_horizon.data import load_or_create_tokenized_dataset
from compression_horizon.train import (
    CompressionHeadTrainer,
    FullCrammingTrainer,
    LowDimTrainer,
    PrefixTuningTrainer,
    ProgressiveCrammingTrainer,
)
from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.utils.launch import freeze_model_parameters, resolve_torch_dtype, set_launch_seed


def _select_trainer_cls(args: MyTrainingArguments) -> type:
    """Pick the trainer class based on which mode flag is set in args."""
    if getattr(args, "train_compression_head", False):
        return CompressionHeadTrainer
    if args.progressive_train:
        return ProgressiveCrammingTrainer
    if args.low_dim_train:
        return LowDimTrainer
    if getattr(args, "train_prefix_tuning", False):
        return PrefixTuningTrainer
    return FullCrammingTrainer


def _resolve_attn_implementation() -> str:
    """Use flash_attention_2 if the flash_attn package is available, else fall back to sdpa."""
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        print("flash_attn not installed, falling back to attn_implementation='sdpa'")
        return "sdpa"


def main() -> None:
    """Train specified setup."""
    parser = transformers.HfArgumentParser(MyTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    if not args.output_dir:
        raise ValueError("--output_dir must be provided!")
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.logging_dir:
        args.logging_dir = args.output_dir

    set_launch_seed(args.random_seed)
    print(f"Random seed: {args.random_seed}")

    torch_dtype = resolve_torch_dtype(args.dtype)
    print(f"torch_dtype: {torch_dtype}")

    attn_implementation = _resolve_attn_implementation()
    print(f"attn_implementation: {attn_implementation}")

    if args.train_compression_head or "experiments_compression_head/ch_head_" in args.model_checkpoint:
        from compression_horizon.models.llama_compression_head import LlamaForCausalLMCompressionHead

        model = LlamaForCausalLMCompressionHead.from_pretrained(
            args.model_checkpoint, dtype=torch_dtype, attn_implementation=attn_implementation
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_checkpoint, dtype=torch_dtype, attn_implementation=attn_implementation
        )
        freeze_model_parameters(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    cache_dir = "artifacts/cache/tokenized_datasets"
    os.makedirs(cache_dir, exist_ok=True)
    train_dataset = load_or_create_tokenized_dataset(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        split="test",
        tokenizer=tokenizer,
        max_sequence_length=args.max_sequence_length,
        model_checkpoint=args.model_checkpoint,
        no_bos_token=args.no_bos_token,
        limit_dataset_items=args.limit_dataset_items,
        offset_dataset_items=args.offset_dataset_items,
    )
    print(f"train_dataset: {len(train_dataset)} samples")

    transformers.logging.set_verbosity_info()
    trainer_cls = _select_trainer_cls(args)
    print(f"Trainer: {trainer_cls.__name__}")

    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    save_path = trainer.train()
    print(f"Saved compressed prefixes to: {save_path}.")


if __name__ == "__main__":
    main()
