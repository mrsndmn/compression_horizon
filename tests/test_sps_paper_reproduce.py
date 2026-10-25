"""Publication measurements must be derived from portable, aligned evidence."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "sps_paper/reproduce.py"


def module():
    assert SCRIPT.exists(), "Missing offline evidence/reproduction entrypoint"
    spec = importlib.util.spec_from_file_location("sps_paper_reproduce", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sample(target, generated, sid=0):
    return dict(
        sample_id=sid,
        horizon=len(target),
        target_ids=target,
        generated_ids=generated,
        teacher_replay_correct_tokens=len(target),
        teacher_replay_accuracy=1.0,
        greedy_exact=target == generated,
        greedy_accuracy=sum(a == b for a, b in zip(target, generated)) / len(target),
    )


def test_short_generation_and_later_recovery_do_not_inflate_initial_prefix():
    api = module()
    row = api.score_sample(sample([10, 20, 30, 40], [10, 99, 30]))
    assert row == dict(
        horizon=4, greedy_exact=False, greedy_initial_prefix=1, greedy_correct_tokens=2, greedy_accuracy=0.5, teacher_exact=True
    )


def test_ids_are_authoritative_and_inconsistent_saved_metrics_are_rejected():
    api = module()
    row = sample([10, 20], [10])
    row["greedy_exact"] = True
    with pytest.raises(ValueError, match="[Gg]reedy"):
        api.score_sample(row)


def test_nonexact_teacher_replay_is_distinct_from_selected_horizon():
    api = module()
    row = sample([10, 20], [10, 20])
    row.update(teacher_replay_correct_tokens=1, teacher_replay_accuracy=0.5)
    assert api.score_sample(row)["teacher_exact"] is False


def test_text_paired_interval_uses_aligned_differences():
    api = module()
    assert api.paired_interval([3, 3, 3]) == {"mean": 3.0, "ci95": [3.0, 3.0]}
    assert api.paired_interval([0, 2])["mean"] == 1.0


def test_snapshot_rejects_corrupt_file_before_measurement(tmp_path):
    api = module()
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {"runs.json.gz": "0" * 64}}))
    (tmp_path / "runs.json.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="[Hh]ash|digest"):
        api.load_snapshot(tmp_path)


def test_export_rejects_missing_completion_evidence(tmp_path):
    api = module()
    with pytest.raises(ValueError, match="[Ii]ncomplete|completed|[Mm]issing"):
        api.export_snapshot(tmp_path, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot/manifest.json").exists()


def test_default_snapshot_covers_exact_cohort_and_derives_results():
    api = module()
    data = api.load_snapshot(ROOT / "sps_paper/data")
    metrics = api.derive(data)
    assert len(data["cohort"]) == 48
    assert metrics["runs"]["standard_ce_raw"]["horizon"]["mean"] == pytest.approx(32.3125)
    assert metrics["runs"]["sps_hinge_strict"]["horizon"]["mean"] == pytest.approx(31.5208333333)
    assert metrics["runs"]["predict_hinge_native"]["horizon"]["mean"] == pytest.approx(132.1041666667)
    assert metrics["runs"]["predict_fixed"]["horizon"]["mean"] == pytest.approx(133.8541666667)
    assert metrics["terminal"]["median_margin_violations"] == 10
    assert metrics["terminal"]["range_margin_violations"] == [2, 92]
    assert metrics["terminal"]["previous_converged"] == 48


@pytest.mark.parametrize("fault", ["duplicate", "target", "counter", "uuid"])
def test_unaggregated_evidence_guards_alignment_provenance_and_counters(fault):
    api = module()
    data = copy.deepcopy(api.load_snapshot(ROOT / "sps_paper/data"))
    run = data["runs"]["predict_hinge_native"]
    if fault == "duplicate":
        run["samples"][1]["sample_id"] = 0
    elif fault == "target":
        run["samples"][0]["target_ids"][0] += 1
    elif fault == "counter":
        run["stages"][0]["steps_taken"] = -1
    else:
        run["jobs"][0]["job_id"] = "lm-mpi-job-425f508b"
    with pytest.raises(ValueError):
        api.derive(data)


def test_render_works_outside_repo_and_does_not_import_training(tmp_path):
    module()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "render", "--output-root", str(tmp_path)], cwd=tmp_path, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("main_results", "objective_factorial", "budget_control", "terminal_constraints", "causal_routes", "rescue"):
        table = (tmp_path / "tables" / f"{name}.tex").read_text()
        assert "\\begin{tabular}" in table and "\\begin{table}" not in table
        assert "nan" not in table.lower()
    for name in ("site_objective", "budget_control", "routes", "terminal_constraints"):
        assert (tmp_path / "figures" / f"{name}.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "data/derived_metrics.json").exists()
    assert (
        "torch"
        not in subprocess.check_output(
            [sys.executable, "-c", f"import runpy,sys;runpy.run_path({str(SCRIPT)!r});print('torch' in sys.modules)"], text=True
        )
        .strip()
        .lower()
    )


def test_replay_plan_is_pinned_and_never_submits():
    api = module()
    plan = api.replay_plan(api.load_snapshot(ROOT / "sps_paper/data"))
    assert "git archive" in plan and "PYTHONPATH" in plan and "--phase probe" in plan
    assert "--suite fixed_schedule" in plan
    assert "Historical" in plan and "not executed" in plan
    assert "--phase submit" not in plan and "mls job submit" not in plan
    assert "2683216db54e0ae8f51a69fa467c2dc07f5509f9" in plan


def test_diagnostic_and_position_cost_claims_derive_from_raw_measurements():
    api = module()
    metrics = api.derive(api.load_snapshot(ROOT / "sps_paper/data"))
    assert metrics["geometry"]["sps_w0/optimized"]["full_row_rank"] == 8
    assert metrics["gradients"]["hinge_perfect_but_higher_ce_than_incorrect_ce03"] == 4
    assert metrics["ablation"]["standard/float32/prefix/nll"][-1] == pytest.approx(4.4798529657)
    assert metrics["trajectory_audit"]["pre_boundary_vectors_equal"] == 6342
    blocks = metrics["runs"]["predict_hinge_native"]["fixed_cohort_blocks"]
    assert next(b for b in blocks if b["end"] == 50)["n_samples"] == 48
    assert next(b for b in blocks if b["end"] == 50)["median"] == 32


def test_replay_heredocs_are_executable_python_and_use_fresh_outputs():
    import ast
    import re

    api = module()
    plan = api.replay_plan(api.load_snapshot(ROOT / "sps_paper/data"))
    blocks = re.findall(r"python - <<'PY'\n(.*?)\nPY", plan, re.S)
    assert len(blocks) == 25
    for block in blocks:
        ast.parse(block)
    assert "out.mkdir(parents=True,exist_ok=False)" in plan
    assert "p.OUTPUT=out" in plan and "jobs.OUTPUT=out" in plan
    assert "p.ROOT=out" in plan
