import os
import subprocess
import uuid

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.train.trainer import MyTrainer
from compression_horizon.utils.exceptions import NvidiaSMIError

if __name__ == "__main__":
    # Check for nvidia-smi availability
    try:
        subprocess.check_output(["nvidia-smi"], shell=True)
    except subprocess.CalledProcessError:
        raise NvidiaSMIError("nvidia-smi is not available")

    # Parse command-line arguments and defaults
    hf_parser = transformers.HfArgumentParser(MyTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    # Make output/logging directory
    if training_args.hybrid_alpha is None:
        output_dir = (
            f"artifacts/experiments/common_loss/"
            f"{training_args.model_checkpoint}|{training_args.max_sequence_length}|{training_args.number_of_mem_tokens}|{uuid.uuid4()}"
        )
    else:
        output_dir = (
            f"artifacts/experiments/hybrid_loss/"
            f"{training_args.model_checkpoint}|{training_args.max_sequence_length}|{training_args.number_of_mem_tokens}|{training_args.learning_rate}|{training_args.loss_type}|{training_args.hybrid_alpha}|{training_args.num_alignment_layers}|{uuid.uuid4()}"
        )
    os.makedirs(output_dir, exist_ok=True)
    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir

    # Initializing the model and its tokenizer
    model = AutoModelForCausalLM.from_pretrained(training_args.model_checkpoint, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_checkpoint)
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Load a sample to compress
    raw_dataset = load_dataset("mrsndmn/pg19", split="test")
    train_dataset = raw_dataset.select(range(1))
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
    trainer = MyTrainer(
        model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    if training_args.progressive_train:
        training_artifacts = trainer.progressive_train()
    else:
        training_artifacts = trainer.train()
    print(f"Saved compressed prefixes to: {training_artifacts}")
