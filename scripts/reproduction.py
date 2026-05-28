import argparse
import os
import subprocess
import sys

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from compression_horizon.data.tokenization import load_or_create_tokenized_dataset
from compression_horizon.train import FullCrammingTrainer
from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.utils.exceptions import NvidiaSMIError
from compression_horizon.utils.launch import resolve_torch_dtype

SUPPORTED_SETUPS = ("common", "no_bos", "2leading")


def _split_setup_arg(argv: list[str]) -> tuple[str, list[str]]:
    """Pull --setup_name out of argv before handing the rest to HfArgumentParser."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--setup_name", required=True, choices=SUPPORTED_SETUPS)
    parsed, remaining = pre.parse_known_args(argv)
    return parsed.setup_name, remaining


def _validate_setup_flags(training_args: MyTrainingArguments, setup_name: str) -> None:
    if setup_name == "common":
        assert not training_args.no_bos_token, "common setup requires no_bos_token=False"
        assert training_args.leading_token_loss_count == 0, "common setup requires leading_token_loss_count=0"
    elif setup_name == "no_bos":
        assert training_args.no_bos_token, "no_bos setup requires --no_bos_token"
        assert training_args.leading_token_loss_count == 0, "no_bos setup requires leading_token_loss_count=0"
    elif setup_name == "2leading":
        assert not training_args.no_bos_token, "2leading setup requires no_bos_token=False"
        assert training_args.leading_token_loss_count == 2, "2leading setup requires --leading_token_loss_count 2"
        assert training_args.leading_token_loss_weight == 3.0, "2leading setup requires --leading_token_loss_weight 3.0"


if __name__ == "__main__":
    # Check for nvidia-smi availability
    try:
        subprocess.check_output(["nvidia-smi"], shell=True)
    except subprocess.CalledProcessError:
        raise NvidiaSMIError("nvidia-smi is not available")

    setup_name, remaining_argv = _split_setup_arg(sys.argv[1:])

    # Parse command-line arguments and defaults
    hf_parser = transformers.HfArgumentParser(MyTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses(args=remaining_argv)

    # Reference-paper hyperparameters (shared by all three setups)
    assert training_args.model_checkpoint in (
        "unsloth/Llama-3.2-1B",
        "unsloth/Llama-3.2-3B",
        "unsloth/Meta-Llama-3.1-8B",
    ), "model_checkpoint is not part of reproduction!"
    assert training_args.max_sequence_length in (512, 1024, 1568), "Wrong maximum sequence length!"
    assert training_args.number_of_mem_tokens == 1, "Only one [mem] token is supported!"
    assert training_args.embedding_init_method == "random", "Only random initialisation is supported!"
    assert training_args.max_optimization_steps_per_sample == 5000, "5000 steps in original paper!"
    assert training_args.hybrid_alpha is None, "Activation alignment is prohibited!"
    assert training_args.learning_rate == 0.01, "Learning rate equals to 0.01 in original paper!"
    assert training_args.adam_beta1 == 0.9, "AdamW beta1 equals to 0.9 in original paper!"
    assert training_args.adam_beta2 == 0.9, "AdamW beta2 equals 0.9 in original paper!"
    assert training_args.weight_decay == 0.01, "AdamW weight decay equals to 0.01 in original paper!"
    assert not training_args.progressive_train, "Progressive training is prohibited!"
    assert training_args.dtype == "bfloat16", "Supported dtype is bfloat16"
    assert training_args.dataset_name == "LarryLovestein/pg19_1k", "Table 18 uses LarryLovestein/pg19_1k (camera-ready dataset)"
    assert training_args.full_cramming_convergence_threshold == 0.99, (
        "Table 18 follows the reference paper's 0.99 teacher-forcing threshold; "
        "pass --full_cramming_convergence_threshold 0.99"
    )
    _validate_setup_flags(training_args, setup_name)

    # Determine output directory based on setup
    default_base = "artifacts/experiments"
    os.makedirs(default_base, exist_ok=True)
    model_short = training_args.model_checkpoint.split("/")[-1]
    prefix = f"{model_short}_{training_args.max_sequence_length}_{setup_name}"
    output_dir = os.path.join(default_base, prefix)
    os.makedirs(output_dir, exist_ok=True)
    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir
    print(f"[reproduction] setup={setup_name}  output_dir={output_dir}")

    # Initializing the model and its tokenizer
    torch_dtype = resolve_torch_dtype(training_args.dtype)
    print("bfloat16 supported:", torch.cuda.is_bf16_supported())
    model = AutoModelForCausalLM.from_pretrained(training_args.model_checkpoint, dtype=torch_dtype)
    print("Initialized dtype:", next(model.parameters()).dtype)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_checkpoint)
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.padding_side = "right"

    # Load samples to compress (cached tokenization, honors --no_bos_token).
    cache_dir = "artifacts/cache/tokenized_datasets"
    os.makedirs(cache_dir, exist_ok=True)
    train_dataset = load_or_create_tokenized_dataset(
        cache_dir=cache_dir,
        dataset_name=training_args.dataset_name,
        split="test",
        tokenizer=tokenizer,
        max_sequence_length=training_args.max_sequence_length,
        model_checkpoint=training_args.model_checkpoint,
        no_bos_token=training_args.no_bos_token,
        limit_dataset_items=training_args.limit_dataset_items,
        offset_dataset_items=training_args.offset_dataset_items,
        cache_prefix="dataset",
    )
    print("train_dataset", len(train_dataset))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train
    transformers.logging.set_verbosity_info()
    trainer = FullCrammingTrainer(
        model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    training_artifacts = trainer.train()
    print(f"Saved compressed prefixes to: {training_artifacts}.")
