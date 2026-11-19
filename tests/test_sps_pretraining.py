import pytest
from scripts import analyze_sps_pretraining as sweep


def test_merge_reports_audited_anchor_and_common_update_cutoffs(tmp_path, monkeypatch):
    snapshot = sweep.load_snapshot(sweep.jobs.REPO / "sps_paper/data")
    configs = [
        dict(name=f"{tokens}_{site}", tokens=tokens, site=site, checkpoint="checkpoint.pt")
        for tokens in sweep.PRETRAINING_TOKENS[:-1]
        for site in sweep.SITES
    ]
    monkeypatch.setattr(sweep, "ROOT", tmp_path)
    monkeypatch.setattr(sweep, "experiment_configs", configs)
    cache = [dict(s, attention_mask=[1] * len(s["input_ids"])) for s in snapshot["cohort"]]
    monkeypatch.setattr(sweep, "load_from_disk", lambda path: cache)
    monkeypatch.setattr(sweep, "historical_controls", lambda data: (snapshot, []))
    monkeypatch.setattr(sweep, "memory_initializations", lambda: {})
    monkeypatch.setattr(sweep, "_job", lambda path: dict(source_commit="test", job_id=path.name))

    def audit_shard(exp, *args):
        run = snapshot["runs"][sweep.SITES[exp["site"]]]
        return run["samples"], run["stages"]

    monkeypatch.setattr(sweep, "audit_shard", audit_shard)
    sweep.merge()
    report = sweep.json_read(tmp_path / "summary.json")
    assert [r["pretraining_tokens"] for r in report["rows"]] == list(sweep.PRETRAINING_TOKENS)
    final = report["rows"][-1]
    assert final["memory"]["mean_horizon"] == pytest.approx(31.52, abs=0.005)
    assert final["predict"]["mean_horizon"] == pytest.approx(132.10, abs=0.005)
    assert final["gain_change_from_earliest"]["mean"] == 0
    cutoffs = final["paired_gain_at_update_cutoff"]
    assert cutoffs[str(sweep.UPDATE_CUTOFFS[0])]["mean_horizons"]["predict"] < final["predict"]["mean_horizon"]
    assert cutoffs[str(sweep.UPDATE_CUTOFFS[-1])]["paired_gain"] == final["paired_horizon_gain"]
    assert (tmp_path / "comparison.png").is_file()


@pytest.mark.parametrize("threshold, error", [(0.5, "threshold mismatch"), (1.0, "Unexplained terminal stage")])
def test_audit_rejects_invalid_threshold_and_truncated_valid_prefix(threshold, error):
    stage = dict(
        sample_id=0,
        stage_index=0,
        stage_seq_len=1,
        num_input_tokens=1,
        steps_taken=10,
        final_convergence=1.0,
        convergence_threshold=threshold,
    )
    sample = dict(
        sample_id=0,
        horizon=1,
        target_ids=[1],
        generated_ids=[1],
        teacher_replay_accuracy=1.0,
        greedy_accuracy=1.0,
        greedy_exact=True,
        stored_convergence=1.0,
    )
    cache = [dict(input_ids=[1, 2], attention_mask=[1, 1])]
    with pytest.raises(ValueError, match=error):
        sweep.audit_run([sample], [stage], cache, sweep.CONTROL_SETTINGS)
