import os
import sys

import pytest
import torch
from tests.trainer_helpers import TinyDataset, _collate_batch, _make_args
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.train import BaseTrainer, FullCrammingTrainer

# Ensure we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compute_ce_loss():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(loss_type="cross_entropy", num_alignment_layers=0)
    trainer = BaseTrainer(model=model, processing_class=tokenizer, args=args)

    batch_size, L, H = (
        2,
        4,
        model.config.hidden_size,
    )  # base length 4, compression length 5
    num_comp = args.number_of_mem_tokens

    # Fake token ids and mask
    input_ids = torch.randint(0, 16, (batch_size, L), device="cuda")
    attention_mask = torch.ones(batch_size, L, dtype=torch.long, device="cuda")

    with torch.no_grad():
        model_token_embeddings = model.model.embed_tokens(input_ids)

    compression_tokens = torch.randn(batch_size, num_comp, H, device="cuda")
    model_tokens_with_comp = torch.cat([compression_tokens, model_token_embeddings], dim=1).to("cuda").to(torch.bfloat16)
    attention_mask_with_comp = torch.cat(
        [torch.ones(batch_size, num_comp, dtype=attention_mask.dtype, device="cuda"), attention_mask],
        dim=1,
    )

    loss, *_ = trainer.forward_and_compute_loss(
        model,
        input_ids.to("cuda"),
        model_token_embeddings.to("cuda"),
        attention_mask.to("cuda"),
        model_tokens_with_comp.to("cuda"),
        attention_mask_with_comp.to("cuda"),
        num_comp,
    )
    # Each selected layer contributes an MSE of 1.0 (constant delta of 1),
    # num_alignment_layers=0 => all 3 layers
    assert loss.item() < 50


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compute_l2_loss_num_alignment_layers():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(hybrid_alpha=1.0, loss_type="l2", num_alignment_layers=0)
    trainer = BaseTrainer(model=model, processing_class=tokenizer, args=args)

    batch_size, L, H = (
        2,
        4,
        model.config.hidden_size,
    )  # base length 4, compression length 5
    num_comp = args.number_of_mem_tokens

    # Fake token ids and mask
    input_ids = torch.randint(0, 16, (batch_size, L), device="cuda")
    attention_mask = torch.ones(batch_size, L, dtype=torch.long, device="cuda")

    with torch.no_grad():
        model_token_embeddings = model.model.embed_tokens(input_ids)

    compression_tokens = torch.randn(batch_size, num_comp, H, device="cuda")
    model_tokens_with_comp = torch.cat([compression_tokens, model_token_embeddings], dim=1).to("cuda").to(torch.bfloat16)
    attention_mask_with_comp = torch.cat(
        [torch.ones(batch_size, num_comp, dtype=attention_mask.dtype, device="cuda"), attention_mask],
        dim=1,
    )

    loss_all_layers, *_ = trainer.forward_and_compute_loss(
        model,
        input_ids.to("cuda"),
        model_token_embeddings.to("cuda"),
        attention_mask.to("cuda"),
        model_tokens_with_comp.to("cuda"),
        attention_mask_with_comp.to("cuda"),
        num_comp,
    )

    trainer.args.num_alignment_layers = 1
    loss_1_layer, *_ = trainer.forward_and_compute_loss(
        model,
        input_ids.to("cuda"),
        model_token_embeddings.to("cuda"),
        attention_mask,
        model_tokens_with_comp.to("cuda"),
        attention_mask_with_comp.to("cuda"),
        num_comp,
    )

    assert loss_all_layers.item() > loss_1_layer.item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_train_freezes_model_params(monkeypatch):
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(
        max_optimization_steps_per_sample=1,
        per_device_train_batch_size=1,
        lr_scheduler_type="constant",
    )

    dataset = TinyDataset(num_samples=2, seq_len=4, vocab_size=16)

    trainer = FullCrammingTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        data_collator=_collate_batch,
    )

    # Before training, parameters are trainable
    assert any(p.requires_grad for p in model.parameters())

    trainer.train()

    # After trainer freezes, all model params must be non-trainable
    assert all(not p.requires_grad for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compute_loss_convergence_metric_shape_and_range():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.bfloat16)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(loss_type="cross_entropy", num_alignment_layers=0)
    trainer = BaseTrainer(model=model, processing_class=tokenizer, args=args)

    batch_size, L, H = 3, 4, model.config.hidden_size
    num_comp = args.number_of_mem_tokens

    input_ids = torch.randint(0, 16, (batch_size, L), device="cuda")
    attention_mask = torch.ones(batch_size, L, dtype=torch.long, device="cuda")

    with torch.no_grad():
        model_token_embeddings = model.model.embed_tokens(input_ids)

    compression_tokens = torch.randn(batch_size, num_comp, H, device="cuda")
    model_tokens_with_comp = torch.cat([compression_tokens, model_token_embeddings], dim=1).to("cuda").to(torch.bfloat16)
    attention_mask_with_comp = torch.cat(
        [torch.ones(batch_size, num_comp, dtype=attention_mask.dtype, device="cuda"), attention_mask],
        dim=1,
    )

    loss, _, convergence, *_ = trainer.forward_and_compute_loss(
        model,
        input_ids.to("cuda"),
        model_token_embeddings.to("cuda"),
        attention_mask.to("cuda"),
        model_tokens_with_comp.to("cuda"),
        attention_mask_with_comp.to("cuda"),
        num_comp,
    )

    assert convergence.shape == (batch_size,)
    assert torch.all(convergence >= 0) and torch.all(convergence <= 1)
