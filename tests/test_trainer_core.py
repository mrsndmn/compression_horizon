import os
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

# Ensure we can import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from train.trainer import MyTrainer  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer


def _make_args(**overrides):
    # Minimal args object for MyTrainer
    defaults = dict(
        model_checkpoint="dummy",
        max_optimization_steps_per_sample=1,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=False,
        max_sequence_length=8,
        loss_type="cross_entropy",
        num_alignment_layers=0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        per_device_train_batch_size=1,
        weight_decay=0.0,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        warmup_steps=0,
        logging_dir=False,  # disable SummaryWriter
        number_of_eos_tokens=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_compute_ce_loss():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(loss_type="cross_entropy", num_alignment_layers=0)  # 0 = all layers

    trainer = MyTrainer(model=model, processing_class=tokenizer, args=args)

    batch_size, L, H = 2, 4, model.config.hidden_size  # base length 4, compression length 5
    num_comp = args.number_of_eos_tokens

    # Fake token ids and mask
    input_ids = torch.randint(0, 16, (batch_size, L))
    attention_mask = torch.ones(batch_size, L, dtype=torch.long)

    with torch.no_grad():
        model_token_embeddings = model.model.embed_tokens(input_ids)

    compression_tokens = torch.randn(batch_size, num_comp, H)
    model_tokens_with_comp = torch.cat([compression_tokens, model_token_embeddings], dim=1)
    attention_mask_with_comp = torch.cat([torch.ones(batch_size, num_comp, dtype=attention_mask.dtype), attention_mask], dim=1)

    loss, _ = trainer.compute_loss(
        model,
        input_ids,
        model_token_embeddings,
        attention_mask,
        model_tokens_with_comp,
        attention_mask_with_comp,
        num_comp,
    )
    # Each selected layer contributes an MSE of 1.0 (constant delta of 1),
    # num_alignment_layers=0 => all 3 layers
    assert loss.item() < 50


def test_compute_l2_loss_num_alignment_layers():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(loss_type="l2", num_alignment_layers=0)  # 0 = all layers

    trainer = MyTrainer(model=model, processing_class=tokenizer, args=args)

    batch_size, L, H = 2, 4, model.config.hidden_size  # base length 4, compression length 5
    num_comp = args.number_of_eos_tokens

    # Fake token ids and mask
    input_ids = torch.randint(0, 16, (batch_size, L))
    attention_mask = torch.ones(batch_size, L, dtype=torch.long)

    with torch.no_grad():
        model_token_embeddings = model.model.embed_tokens(input_ids)

    compression_tokens = torch.randn(batch_size, num_comp, H)
    model_tokens_with_comp = torch.cat([compression_tokens, model_token_embeddings], dim=1)
    attention_mask_with_comp = torch.cat([torch.ones(batch_size, num_comp, dtype=attention_mask.dtype), attention_mask], dim=1)

    loss_all_layers, _ = trainer.compute_loss(
        model,
        input_ids,
        model_token_embeddings,
        attention_mask,
        model_tokens_with_comp,
        attention_mask_with_comp,
        num_comp,
    )

    trainer.args.num_alignment_layers = 1
    loss_1_layer, _ = trainer.compute_loss(
        model,
        input_ids,
        model_token_embeddings,
        attention_mask,
        model_tokens_with_comp,
        attention_mask_with_comp,
        num_comp,
    )

    assert loss_all_layers.item() > loss_1_layer.item()


class TinyDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class MultiAccessBatch:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError as e:
            raise AttributeError(str(e))


def _collate_batch(samples):
    # Stack into shape [B, 1, L] to mirror tokenizer return_tensors behavior
    input_ids = torch.stack([s["input_ids"] for s in samples], dim=0).unsqueeze(1)
    attention_mask = torch.stack([s["attention_mask"] for s in samples], dim=0).unsqueeze(1)
    return MultiAccessBatch({"input_ids": input_ids, "attention_mask": attention_mask})


def test_train_freezes_model_params(monkeypatch):
    # Force CPU to avoid device mismatches in CI environments
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(
        max_optimization_steps_per_sample=1,
        per_device_train_batch_size=1,
        lr_scheduler_type="constant",
    )

    dataset = TinyDataset(num_samples=2, seq_len=4, vocab_size=16)

    trainer = MyTrainer(
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


def test_compute_loss_convergence_metric_shape_and_range():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    args = _make_args(loss_type="cross_entropy", num_alignment_layers=0)
    trainer = MyTrainer(model=model, processing_class=tokenizer, args=args)

    batch_size, L, H = 3, 4, model.config.hidden_size
    num_comp = args.number_of_eos_tokens

    input_ids = torch.randint(0, 16, (batch_size, L))
    attention_mask = torch.ones(batch_size, L, dtype=torch.long)

    with torch.no_grad():
        model_token_embeddings = model.model.embed_tokens(input_ids)

    compression_tokens = torch.randn(batch_size, num_comp, H)
    model_tokens_with_comp = torch.cat([compression_tokens, model_token_embeddings], dim=1)
    attention_mask_with_comp = torch.cat([torch.ones(batch_size, num_comp, dtype=attention_mask.dtype), attention_mask], dim=1)

    _, conv = trainer.compute_loss(
        model,
        input_ids,
        model_token_embeddings,
        attention_mask,
        model_tokens_with_comp,
        attention_mask_with_comp,
        num_comp,
    )

    assert conv.shape == (batch_size,)
    assert torch.all(conv >= 0) and torch.all(conv <= 1)
