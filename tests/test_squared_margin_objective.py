import pytest
import torch

from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.train.loss import next_token_squared_margin_loss_with_prefix
from compression_horizon.train.trainers.base import BaseTrainer


def test_squared_margin_offsets_mask_and_ignores_already_safe_tokens():
    logits = torch.zeros(1, 6, 4, requires_grad=True)
    with torch.no_grad():
        logits[0, 2] = torch.tensor([0.0, 1.0, -2.0, -3.0])
        logits[0, 3] = torch.tensor([-2.0, 3.0, 0.0, -3.0])
        logits[0, 4] = torch.tensor([9.0, 0.0, 0.0, 0.0])
    ids = torch.tensor([[0, 1, 2]])
    loss = next_token_squared_margin_loss_with_prefix(logits, ids, torch.tensor([[1, 1, 0]]), 2, prefix_len=1, margin=0.1)
    assert loss.item() == pytest.approx(1.1**2 / 2)
    loss.backward()
    assert logits.grad[0, 2, 0] < 0 and logits.grad[0, 2, 1] > 0
    assert logits.grad[0, :2].abs().sum() == 0
    assert logits.grad[0, 3:].abs().sum() == 0


@pytest.mark.parametrize("sps", [False, True])
def test_native_base_dispatch_matches_rescue_hinge_and_backpropagates(sps):
    from scripts.analyze_sps_margin_geometry import target_margins
    from tests.test_sps_causal_lm import tiny_args

    from compression_horizon.models.sps_causal_lm import SPSCausalLM

    model = SPSCausalLM(tiny_args(), sps=sps).eval().requires_grad_(False)
    ids = torch.tensor([[1, 2, 3]])
    memory = torch.randn(1, 1, 16, requires_grad=True)
    logits = model(inputs_embeds=torch.cat((memory, model.get_input_embeddings()(ids)), 1)).logits
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = MyTrainingArguments(loss_type="cross_entropy", token_objective="squared_margin", loss_margin=0.05)
    loss, alignment = trainer.compute_loss(logits, ids, torch.ones_like(ids), 1)
    expected = (0.05 - target_margins(logits[:, :-1], ids)).clamp_min(0).square().mean()
    torch.testing.assert_close(loss, expected)
    assert alignment is None
    loss.backward()
    assert memory.grad.isfinite().all() and memory.grad.norm() > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(token_objective="unknown"),
        dict(token_objective="squared_margin", loss_type="cross_entropy"),
        dict(token_objective="squared_margin", loss_type="l2", loss_margin=0.05),
        dict(token_objective="squared_margin", loss_type="cross_entropy", loss_margin=0.05, train_prefix_tuning=True),
    ],
)
def test_invalid_or_unsupported_combinations_fail_early(kwargs):
    with pytest.raises(ValueError, match="token_objective|squared_margin"):
        MyTrainingArguments(**kwargs)


def test_objective_launcher_factorial_keeps_same_lr_data_batch_and_budgets():
    from scripts.jobs.run_jobs_sps import TRAINING_ARGS, build_config, objective_experiment_configs

    assert len(objective_experiment_configs) == 6
    cells = {}
    for e in objective_experiment_configs:
        d = build_config(e, "test")["design"]["training_args"]
        for key, value in TRAINING_ARGS.items():
            if key != "learning_rate":
                assert d[key] == value
        assert d["learning_rate"] == 0.3 and d["per_device_train_batch_size"] == 1
        cells.setdefault(e["checkpoint"], set()).add((d["token_objective"], d["loss_margin"], d["convergence_margin"]))
    assert len(cells) == 2
    assert all(
        v == {("cross_entropy", 0.0, 0.05), ("squared_margin", 0.05, 0.0), ("squared_margin", 0.05, 0.05)}
        for v in cells.values()
    )


def test_default_ce_branch_remains_exact():
    from compression_horizon.train.loss import compute_hybrid_cross_entropy_and_alignment_loss

    args = MyTrainingArguments(loss_type="cross_entropy")
    logits = torch.randn(1, 4, 7)
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = args
    actual, _ = trainer.compute_loss(logits, ids, mask, 1)
    expected, _ = compute_hybrid_cross_entropy_and_alignment_loss(
        logits=logits,
        input_ids=ids,
        attention_mask=mask,
        num_compression_tokens=1,
        num_alignment_layers=args.num_alignment_layers,
        inverted_alignment=args.inverted_alignment,
        loss_type="cross_entropy",
        hybrid_alpha=None,
    )
    assert torch.equal(actual, expected)


def test_bf16_loss_is_fp32_and_all_padding_has_zero_gradient():
    logits = torch.randn(1, 4, 5, dtype=torch.bfloat16, requires_grad=True)
    loss = next_token_squared_margin_loss_with_prefix(
        logits, torch.tensor([[1, 2, 3]]), torch.zeros(1, 3, dtype=torch.long), 1, margin=0.05
    )
    assert loss.dtype == torch.float32 and loss.item() == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_encoder_decoder_guard_does_not_silently_ignore_objective():
    trainer = BaseTrainer.__new__(BaseTrainer)
    from types import SimpleNamespace

    trainer.args = SimpleNamespace(token_objective="squared_margin")
    with pytest.raises(NotImplementedError, match="token_objective"):
        trainer._assert_encoder_decoder_supported()
