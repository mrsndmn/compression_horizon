"""Smoke tests for LowDimTrainer."""

import os
import sys

import pytest
import torch
from tests.trainer_helpers import TinyDataset, _collate_batch, _make_args
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.train import LowDimTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_low_dim_trainer_smoke():
    """Instantiate LowDimTrainer, run train() with tiny data, check no crash."""
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(
        low_dim_train=True,
        low_dim_size=8,
        max_optimization_steps_per_sample=2,
        number_of_mem_tokens=1,
        logging_dir=None,
    )
    dataset = TinyDataset(num_samples=1, seq_len=4, vocab_size=16)

    trainer = LowDimTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=_collate_batch,
    )
    out = trainer.train()
    assert out is None or isinstance(out, str)
