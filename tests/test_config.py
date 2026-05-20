"""Hygiene tests for MyTrainingArguments."""

from __future__ import annotations

import dataclasses

import pytest

from compression_horizon.train.arguments import MyTrainingArguments


@pytest.mark.parametrize(
    "field_name, expected_default",
    [
        ("max_optimization_steps_per_sample", 1_000),
        ("max_optimization_steps_per_token", 1_000),
        ("random_seed", 42),
        ("fix_position_ids", False),
    ],
)
def test_my_training_arguments_no_duplicates(field_name: str, expected_default):
    """Each field is declared exactly once with the expected default."""
    field_names = [f.name for f in dataclasses.fields(MyTrainingArguments)]
    assert field_names.count(field_name) == 1, f"{field_name} is declared multiple times in MyTrainingArguments"
    f = MyTrainingArguments.__dataclass_fields__[field_name]
    assert f.default == expected_default
