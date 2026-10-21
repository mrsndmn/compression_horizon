"""MLS single-line script contract and conditional success marker."""

import importlib.util
import shlex
import subprocess
from pathlib import Path

spec = importlib.util.spec_from_file_location("sps_jobs", Path(__file__).parents[1] / "scripts/jobs/run_jobs_sps.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_mls_shell_contract(tmp_path, monkeypatch):
    # Replace only the remote mount and interpreter so the actual emitted script
    # can run locally without training or contacting MLS.
    monkeypatch.setattr(launcher, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(launcher, "cluster_path", str)
    python = tmp_path / "fake-python"
    python.write_text("#!/bin/sh\nexit 3\n")
    python.chmod(0o755)
    monkeypatch.setattr(launcher, "PYTHON", str(python))
    cfg = launcher.build_config(launcher.experiment_configs[0], "test")
    cfg["stage"].mkdir(parents=True)
    cfg["output"].mkdir()
    assert "\n" not in cfg["script"], "MLS rejects scripts containing literal newlines"
    assert subprocess.run(["bash", "-c", cfg["script"]]).returncode == 3
    assert not (cfg["output"] / "completed").exists()
    python.write_text("#!/bin/sh\nexit 0\n")
    assert subprocess.run(["bash", "-c", cfg["script"]]).returncode == 0
    assert (cfg["output"] / "completed").exists()


def test_generated_job_paths_use_cluster_mount():
    cfg = launcher.build_config(launcher.experiment_configs[1], "test")
    assert "\n" not in cfg["script"]
    assert "/mnt/virtual_ai" not in cfg["script"]
    assert "/workspace-SR004.nfs2/d.tarasov/sps-artifacts/" in cfg["script"]
    assert shlex.split(cfg["script"])[0] == "cd"


def test_post_training_audit_failure_prevents_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(launcher, "cluster_path", str)
    python = tmp_path / "fake-python"
    python.write_text('#!/bin/sh\ncase "$1" in\n  *validate_sps_batch1_run.py) exit 7;;\n  *) exit 0;;\nesac\n')
    python.chmod(0o755)
    monkeypatch.setattr(launcher, "PYTHON", str(python))
    cfg = launcher.build_config(launcher.experiment_configs[0], "test")
    cfg["stage"].mkdir(parents=True)
    cfg["output"].mkdir()
    assert subprocess.run(["bash", "-c", cfg["script"]]).returncode == 7
    assert not (cfg["output"] / "completed").exists()


def test_actual_validator_imports_with_staged_environment(monkeypatch):
    import os
    import sys

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(launcher, "cluster_path", str)
    env = dict(os.environ, PYTHONPATH=launcher.staged_pythonpath(repo), OMP_NUM_THREADS="2")
    result = subprocess.run(
        [sys.executable, "scripts/validation/validate_sps_batch1_run.py", "--help"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--run-dir" in result.stdout
