import torch
import os
import random
import string
import transformers
from datasets import load_dataset

from train.arguments import MyTrainingArguments
from train.trainer import MyTrainer

from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":

    import subprocess

    subprocess.check_output(["nvidia-smi"])

    hf_parser = transformers.HfArgumentParser(MyTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    # Build output directory: ch_{loss_type}_{6random_letters}
    def _rand_suffix(n=6):
        return "".join(random.choice(string.ascii_lowercase) for _ in range(n))

    if training_args.progressive_train:
        run_dir_name = f"artifacts/experiments_progressive/ch_{getattr(training_args, 'loss_type', 'l2')}_init_{training_args.embedding_init_method}_seq_len_{training_args.max_sequence_length}_{_rand_suffix(6)}"
    else:
        run_dir_name = f"artifacts/experiments/ch_{getattr(training_args, 'loss_type', 'l2')}_init_{training_args.embedding_init_method}_seq_len_{training_args.max_sequence_length}_{_rand_suffix(6)}"

    # Place at repo root with exact template
    output_dir = run_dir_name
    os.makedirs(output_dir, exist_ok=True)
    # Attach to args so trainer can save artifacts there
    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir

    model = AutoModelForCausalLM.from_pretrained(training_args.model_checkpoint, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_checkpoint)

    raw_dataset = load_dataset("mrsndmn/pg19", split="test", num_proc=4)
    train_dataset = raw_dataset.select(range(1))
    # eval_dataset = raw_dataset.select(range(10, 20))

    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = train_dataset.map(
        lambda x: tokenizer(
            x["text"], truncation=True, padding="max_length", max_length=training_args.max_sequence_length, return_tensors="pt"
        ),
        remove_columns=train_dataset.column_names,
    )

    print("train_dataset", len(train_dataset))
    print("train_dataset", train_dataset)
    # print("eval_dataset", len(eval_dataset))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
