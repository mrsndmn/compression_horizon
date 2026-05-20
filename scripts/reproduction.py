import os
import subprocess

import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from compression_horizon.train import FullCrammingTrainer
from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.utils.exceptions import NvidiaSMIError
from compression_horizon.utils.launch import resolve_torch_dtype

if __name__ == "__main__":
    # Check for nvidia-smi availability
    try:
        subprocess.check_output(["nvidia-smi"], shell=True)
    except subprocess.CalledProcessError:
        raise NvidiaSMIError("nvidia-smi is not available")

    # Parse command-line arguments and defaults
    hf_parser = transformers.HfArgumentParser(MyTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    # Assert parameters for reproduction
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

    # Determine output directory
    default_base = "artifacts/experiments"
    os.makedirs(default_base, exist_ok=True)
    prefix = f"{training_args.model_checkpoint.split('/')[-1]}_{training_args.max_sequence_length}_2leading"
    output_dir = os.path.join(default_base, prefix)
    os.makedirs(output_dir, exist_ok=True)
    # Attach to args so trainer can save artifacts there (respecting any user-provided output_dir)
    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir

    # Initializing the model and its tokenizer
    torch_dtype = resolve_torch_dtype(training_args.dtype)
    print("bfloat16 supported:", torch.cuda.is_bf16_supported())
    model = AutoModelForCausalLM.from_pretrained(training_args.model_checkpoint, dtype=torch_dtype)
    print("Initialized dtype:", next(model.parameters()).dtype)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_checkpoint)
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Load samples to compress
    raw_dataset = load_dataset("mrsndmn/pg19", split="test", num_proc=4)
    train_dataset = raw_dataset.select(range(training_args.limit_dataset_items))
    train_dataset = train_dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=training_args.max_sequence_length,
            return_tensors="pt",
        ),
        remove_columns=train_dataset.column_names,
    )
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
