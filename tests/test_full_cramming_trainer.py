"""Smoke tests for FullCrammingTrainer."""

import os
import sys

import pytest
import torch
from tests.trainer_helpers import TinyDataset, _collate_batch, _make_args
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.train import FullCrammingTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_cramming_trainer_smoke():
    """Instantiate FullCrammingTrainer, run train() with tiny data, check return / no crash."""
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(
        max_optimization_steps_per_sample=1,
        per_device_train_batch_size=1,
        lr_scheduler_type="constant",
        logging_dir=None,
    )
    dataset = TinyDataset(num_samples=2, seq_len=4, vocab_size=16)

    trainer = FullCrammingTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=_collate_batch,
    )
    out = trainer.train()
    assert out is None or isinstance(out, str)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_cramming_trainer_freezes_model_params():
    """FullCrammingTrainer.train() freezes base model parameters."""
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(
        max_optimization_steps_per_sample=1,
        per_device_train_batch_size=1,
        lr_scheduler_type="constant",
        logging_dir=None,
    )
    dataset = TinyDataset(num_samples=2, seq_len=4, vocab_size=16)

    trainer = FullCrammingTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=_collate_batch,
    )
    assert any(p.requires_grad for p in model.parameters())
    trainer.train()
    assert all(not p.requires_grad for p in model.parameters())
