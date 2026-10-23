"""Check an independently pinned LR trajectory while the update budget grows."""

import copy

import pytest
import torch
from transformers import HfArgumentParser

from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.train.optimization import build_optimizer_and_scheduler


def make_args(**kwargs):
    return MyTrainingArguments(output_dir="/tmp/lr-control", report_to="none", learning_rate=0.3, warmup_steps=100, **kwargs)


@pytest.mark.parametrize("processes", [1, 2])
def test_fixed_decay_matches_original_then_holds_floor(processes):
    original = make_args()
    # A misspelled or ignored CLI override must not silently stretch the curve.
    (controlled,) = HfArgumentParser(MyTrainingArguments).parse_args_into_dataclasses(
        [
            "--output_dir",
            "/tmp/lr-control",
            "--report_to",
            "none",
            "--learning_rate",
            "0.3",
            "--warmup_steps",
            "100",
            "--lr_scheduler_decay_steps",
            "10000",
        ]
    )
    parameter = torch.nn.Parameter(torch.zeros(1))
    old_optimizer, old = build_optimizer_and_scheduler(original, [parameter], 10000, processes)
    optimizer, scheduler = build_optimizer_and_scheduler(controlled, [parameter], 30000, processes)
    for step in range(30000 * processes + 1):
        actual = scheduler.get_last_lr()[0]
        if step <= 10000 * processes:
            assert actual == old.get_last_lr()[0]
            old_optimizer.step()
            old.step()
        else:
            assert actual == pytest.approx(0.001, abs=1e-12)
        if step == 6000 * processes:
            assert actual == pytest.approx(0.10611040388842295)
        optimizer.step()
        scheduler.step()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr_scheduler_decay_steps": 0},
        {"lr_scheduler_decay_steps": -1},
        {"lr_scheduler_decay_steps": 100},
        {"lr_scheduler_decay_steps": 10000, "lr_scheduler_type": "constant"},
    ],
)
def test_invalid_fixed_decay_fails_before_training(kwargs):
    with pytest.raises(ValueError, match="lr_scheduler_decay_steps"):
        make_args(**kwargs)


def test_fixed_decay_requires_a_scheduled_optimizer():
    args = make_args(lr_scheduler_decay_steps=10000)
    with pytest.raises(ValueError, match="num_training_steps"):
        build_optimizer_and_scheduler(args, [torch.nn.Parameter(torch.zeros(1))])


def test_fixed_decay_checkpoint_restores_steps_across_the_boundary():
    args = make_args(lr_scheduler_decay_steps=110)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer, scheduler = build_optimizer_and_scheduler(args, [parameter], 330)
    for _ in range(105):
        optimizer.step()
        scheduler.step()
    saved_optimizer, saved_scheduler = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
    expected = []
    for _ in range(20):
        optimizer.step()
        scheduler.step()
        expected.append(scheduler.get_last_lr()[0])
    restored_optimizer, restored_scheduler = build_optimizer_and_scheduler(args, [parameter], 330)
    restored_optimizer.load_state_dict(saved_optimizer)
    restored_scheduler.load_state_dict(saved_scheduler)
    actual = []
    for _ in range(20):
        restored_optimizer.step()
        restored_scheduler.step()
        actual.append(restored_scheduler.get_last_lr()[0])
    assert actual == expected
    assert actual[-1] == pytest.approx(0.001, abs=1e-12)
