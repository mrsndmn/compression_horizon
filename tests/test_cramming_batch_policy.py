"""User-required single-sample cramming is enforced before expensive setup."""

import importlib.util
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from compression_horizon.utils.launch import validate_cramming_batch_size

REPO = Path(__file__).parents[1]


def load_script(relative):
    spec = importlib.util.spec_from_file_location("policy_" + Path(relative).stem, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("batch_size", [2, 4])
def test_training_cli_rejects_large_cramming_before_output_setup(tmp_path, monkeypatch, batch_size):
    original = subprocess.check_output

    def check_output(command, *args, **kwargs):
        return b"" if command == ["nvidia-smi"] else original(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", check_output)
    output = tmp_path / "no-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "activation_distillation.py",
            "--per_device_train_batch_size",
            str(batch_size),
            "--model_checkpoint",
            str(tmp_path / "nonexistent"),
            "--output_dir",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="per_device_train_batch_size=1"):
        runpy.run_path(str(REPO / "scripts/activation_distillation.py"), run_name="__main__")
    assert not output.exists()


@pytest.mark.parametrize("batch,head", [(1, False), (8, True)])
def test_valid_execution_modes_remain_available(batch, head):
    validate_cramming_batch_size(SimpleNamespace(per_device_train_batch_size=batch, train_compression_head=head))


def test_sps_both_checkpoints_remeasure_original_effective_text_subset():
    module = load_script("scripts/jobs/run_jobs_sps.py")
    assert len(module.experiment_configs) == 2
    for exp in module.experiment_configs:
        script = module.build_config(exp, "test")["script"]
        argv = shlex.split(script)
        assert argv[argv.index("--per_device_train_batch_size") + 1] == "1"
        assert argv[argv.index("--limit_dataset_items") + 1] == "48"
        assert exp["checkpoint"].endswith("ckpt_tokens_1000341504_final.pt")


def test_model_grid_emits_single_sample_cramming():
    module = load_script("scripts/jobs/run_training.py")
    for config in module.build_experiment_configs():
        argv = shlex.split(config["command"])
        assert argv[argv.index("--per_device_train_batch_size") + 1] == "1"


@pytest.mark.parametrize("script", ["scripts/jobs/run_jobs.py", "scripts/jobs/run_jobs_low_dim.py"])
def test_legacy_launchers_reject_explicit_large_batch_before_submission(script, monkeypatch):
    monkeypatch.setattr(sys, "argv", [script, "--per_device_train_batch_size", "4"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(REPO / script), run_name="__main__")
    assert raised.value.code == 2


@pytest.mark.parametrize("early_eos", [False, True])
def test_single_sample_audit_uses_full_target_length(early_eos):
    import torch
    from scripts.validation.validate_sps_batch1_run import check_sample
    from tests.test_sps_generation_semantics import FixedLogitLM, SmallTokenizer

    model = FixedLogitLM()
    tokenizer = SmallTokenizer()
    if early_eos:
        tokenizer.eos_token_id = 2
    result = check_sample(model, tokenizer, torch.tensor([[2, 2, 2]]), torch.zeros(1, 1, 4))
    assert result["greedy_accuracy"] == (1 / 3 if early_eos else 1.0)
    assert result["greedy_exact"] is (not early_eos)
