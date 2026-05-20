"""Smoke tests for PrefixTuningTrainer."""

import os
import sys

import pytest
import torch
from tests.trainer_helpers import TinyDataset, _collate_batch, _make_args
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.train import PrefixTuningTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefix_tuning_trainer_smoke():
    """Instantiate PrefixTuningTrainer, run train() with tiny data, check no crash."""
    pytest.importorskip("peft")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(
        train_prefix_tuning=True,
        max_optimization_steps_per_sample=2,
        number_of_mem_tokens=1,
        logging_dir=None,
    )
    dataset = TinyDataset(num_samples=1, seq_len=4, vocab_size=16)

    trainer = PrefixTuningTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=_collate_batch,
    )
    out = trainer.train()
    # May return save path or None depending on config
    assert out is None or isinstance(out, str)
