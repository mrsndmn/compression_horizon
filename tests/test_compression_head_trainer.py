"""Smoke tests for CompressionHeadTrainer."""

import os
import sys
import tempfile

import pytest
import torch
from tests.trainer_helpers import TinyDataset, _collate_batch, _make_args
from transformers import AutoTokenizer

from compression_horizon.train import CompressionHeadTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _get_compression_head_model():
    """Return a compression head model if available, else None (SmolLM2 is not one)."""
    try:
        from compression_horizon.models.llama_compression_head import LlamaForCausalLMCompressionHead
    except Exception:
        return None
    # Use a small Llama checkpoint if available; otherwise skip
    small_checkpoints = ["HuggingFaceTB/SmolLM2-135M"]
    for ckpt in small_checkpoints:
        try:
            # LlamaForCausalLMCompressionHead only supports Llama configs
            model = LlamaForCausalLMCompressionHead.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
            return model
        except Exception:
            continue
    return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compression_head_trainer_smoke():
    """Instantiate CompressionHeadTrainer, run train() with tiny data, check no crash."""
    model = _get_compression_head_model()
    if model is None:
        pytest.skip(
            "CompressionHeadTrainer requires LlamaForCausalLMCompressionHead; " "SmolLM2 is not a compression head model."
        )
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(
        train_compression_head=True,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        compression_head_freeze_base_model=True,
        logging_dir=None,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        args.output_dir = tmpdir
        dataset = TinyDataset(num_samples=2, seq_len=4, vocab_size=16)

        trainer = CompressionHeadTrainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=dataset,
            eval_dataset=None,
            data_collator=_collate_batch,
        )
        out = trainer.train()
    assert out is not None
    assert isinstance(out, str)
