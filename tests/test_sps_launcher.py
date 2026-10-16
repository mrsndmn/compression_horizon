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
