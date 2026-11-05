import copy

import pytest
from scripts.analyze_pretrained_sps import audit_sample


def sample_record(sid=0, ids=None, generated=None):
    ids = [2, 3] if ids is None else ids
    generated = list(ids) if generated is None else generated
    return dict(
        sample_id=sid,
        selected_length=len(ids),
        horizon=len(ids),
        steps_taken=2,
        stored_convergence=1.0,
        target_ids=ids,
        generated_ids=generated,
        greedy_accuracy=sum(a == b for a, b in zip(ids, generated)) / len(ids),
        greedy_exact=ids == generated,
        teacher_replay_correct_tokens=len(ids),
        teacher_replay_accuracy=1.0,
    )


def raw_stages(sid=0):
    return [
        dict(
            sample_id=sid,
            stage_index=i,
            stage_seq_len=i + 1,
            num_input_tokens=i + 1,
            steps_taken=steps,
            final_convergence=conv,
            convergence_threshold=1.0,
        )
        for i, (steps, conv) in enumerate([(1, 1.0), (2, 1.0), (1002, 0.5)])
    ]


def settings():
    return dict(max_optimization_steps_per_sample=10000, max_optimization_steps_per_token=1000, max_sequence_length=4096)


@pytest.mark.parametrize("generated,expected_prefix", [([2, 3], 2), ([2], 1), ([7, 3], 0)])
def test_stage_and_generation_audit_retains_short_output_and_first_error(generated, expected_prefix):
    result = audit_sample(raw_stages(), sample_record(generated=generated), [2, 3, 4], settings(), valid_length=3)
    assert result["horizon"] == 2 and result["greedy_initial_prefix"] == expected_prefix
    assert result["total_steps"] == 1002 and result["selected_steps"] == 2
    assert result["stop_reason"] == "stage_budget" and result["first_failed_length"] == 3


@pytest.mark.parametrize(
    "corruption", ["missing_stage", "target", "horizon", "updates", "accuracy", "after_failure", "fractional_updates"]
)
def test_inconsistent_selected_or_raw_evidence_is_rejected(corruption):
    rows, sample = raw_stages(), sample_record()
    if corruption == "missing_stage":
        rows.pop(0)
    elif corruption == "target":
        sample["target_ids"][0] = 9
    elif corruption == "horizon":
        sample["horizon"] += 1
    elif corruption == "updates":
        rows[-1]["steps_taken"] += 1
    elif corruption == "accuracy":
        sample["greedy_accuracy"] = 0.5
    elif corruption == "fractional_updates":
        rows[-1]["steps_taken"] = 1001.5
    else:
        rows[0]["final_convergence"] = 0.0
    with pytest.raises(ValueError):
        audit_sample(rows, sample, [2, 3, 4], settings(), valid_length=3)


def test_raw_stage_cannot_extend_past_the_valid_document():
    with pytest.raises(ValueError, match="Stage identity"):
        audit_sample(raw_stages(), sample_record(), [2, 3, 4], settings(), valid_length=2)


def test_total_budget_and_margin_failure_are_separate_from_argmax_accuracy():
    rows = raw_stages()
    rows[-1]["steps_taken"] = 3
    cfg = dict(settings(), max_optimization_steps_per_sample=3)
    assert audit_sample(rows, sample_record(), [2, 3, 4], cfg, valid_length=3)["stop_reason"] == "total_budget"
    rows = [dict(rows[0], steps_taken=1000, final_convergence=0.0)]
    sample = dict(sample_record(ids=[2]), horizon=0, steps_taken=1000, stored_convergence=0.0)
    result = audit_sample(rows, sample, [2, 3, 4], settings(), valid_length=3)
    assert result["horizon"] == 0 and result["teacher_exact"] and result["greedy_exact"]


def test_successful_post_update_measurement_still_stops_at_stage_budget():
    rows = raw_stages()
    rows[-1]["final_convergence"] = 1.0
    sample = dict(sample_record(ids=[2, 3, 4]), steps_taken=1002)
    result = audit_sample(rows, sample, [2, 3, 4, 5], settings(), valid_length=4)
    assert result["stop_reason"] == "stage_budget"
    assert result["horizon"] == 3 and result["first_failed_length"] is None
    assert result["teacher_exact"] and result["total_steps"] == 1002


@pytest.mark.parametrize("budget", ["total", "stage"])
def test_successful_measurements_cannot_continue_after_budget_exhaustion(budget):
    rows = raw_stages()[:2]
    cfg = settings()
    if budget == "total":
        cfg["max_optimization_steps_per_sample"] = 1
        rows[-1]["steps_taken"] = 1
    else:
        cfg["max_optimization_steps_per_token"] = 1
    sample = dict(sample_record(), steps_taken=rows[-1]["steps_taken"])
    with pytest.raises(ValueError, match="Unexpected stages after budget exhaustion"):
        audit_sample(rows, sample, [2, 3, 4], cfg, valid_length=3)


@pytest.fixture(params=["ar", "ce"])
def cohort(tmp_path, monkeypatch, request):
    from datasets import Dataset
    from scripts import analyze_pretrained_sps as analysis
    from scripts import pretrained_sps_worker as worker
    from scripts.pretrained_sps_experiments import TRAINING_ARGS, write_json

    exp = dict(
        name="test_" + request.param,
        family="test",
        arm=request.param,
        source="test/source",
        revision="pinned",
        source_path=str(tmp_path / "hf"),
        checkpoint=None if request.param == "ar" else "final.pt",
    )
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "COHORT", 2)
    monkeypatch.setattr(analysis, "SHARD_SIZE", 1)
    monkeypatch.setattr(analysis, "code_digest", lambda _: "b" * 64)
    data = Dataset.from_dict(dict(input_ids=[[2, 3, 9], [4, 5, 9]], attention_mask=[[1, 1, 1], [1, 1, 1]]))
    manifest = dict(document_hashes=["c" * 64, "d" * 64])
    monkeypatch.setattr(worker, "validate_data", lambda _: (data, manifest))
    write_json(tmp_path / "hf/config.json", dict(bos_token_id=29, hidden_size=16))
    paths = []
    for sid in range(2):
        path = tmp_path / "runs" / ("b" * 12) / exp["name"] / f"shard{sid:02d}"
        paths.append(path)
        write_json(
            path / "job.json", dict(commit="a" * 40, result=dict(job_name=f"lm-mpi-job-00000000-0000-0000-0000-{sid:012d}"))
        )
        samples = [sample_record(sid=sid, ids=list(data[sid]["input_ids"][:2]))]
        cfg = dict(
            TRAINING_ARGS, model_checkpoint=exp["checkpoint"] or exp["source"], offset_dataset_items=sid, limit_dataset_items=1
        )
        role = "ordinary_memory" if request.param == "ar" else "shared_predict"
        report = dict(
            settings=cfg,
            experiment=exp,
            code_digest="b" * 64,
            source_commit="a" * 40,
            data_manifest=manifest,
            embedding_role=role,
            bos_token_id=29,
            elapsed_seconds=2.0,
            samples=samples,
        )
        write_json(path / "single_sample_validation.json", report)
        rows = raw_stages(sid)
        for row in rows:
            row.update(
                {
                    k: cfg[k]
                    for k in [
                        "loss_type",
                        "token_objective",
                        "loss_margin",
                        "convergence_margin",
                        "dtype",
                        "model_checkpoint",
                        "max_optimization_steps_per_sample",
                    ]
                }
            )
            row.update(num_compression_tokens=1, hidden_size=16)
            if request.param != "ar":
                row.update(embedding_role=role, fixed_bos_token_id=29, predict_init="native")
        folder = "progressive_prefixes" if request.param == "ar" else "predict_prefixes"
        Dataset.from_list(rows).save_to_disk(str(path / folder))
        (path / "completed").touch()
    return analysis, exp, paths


def test_complete_cohort_export(cohort, tmp_path):
    analysis, exp, _ = cohort
    result = analysis.audit_experiment(exp, output_dir=tmp_path / "output")
    assert result["n_samples"] == 2 and result["mean_horizon"] == 2
    assert result["mean_greedy_initial_prefix"] == 2 and result["stopping_counts"] == {"stage_budget": 2}
    assert (tmp_path / "output/per_sample.csv").exists()


@pytest.mark.parametrize("corruption", ["incomplete", "duplicate_job", "provenance", "sample"])
def test_inconsistent_cohort_is_never_exported(cohort, tmp_path, corruption):
    analysis, exp, paths = cohort
    from scripts.pretrained_sps_experiments import read_json, write_json

    if corruption == "incomplete":
        (paths[1] / "completed").unlink()
    elif corruption == "duplicate_job":
        write_json(paths[1] / "job.json", read_json(paths[0] / "job.json"))
    else:
        report = copy.deepcopy(read_json(paths[1] / "single_sample_validation.json"))
        if corruption == "provenance":
            report["source_commit"] = "f" * 40
        else:
            report["samples"][0]["sample_id"] = 0
        write_json(paths[1] / "single_sample_validation.json", report)
    with pytest.raises(ValueError):
        analysis.audit_experiment(exp, output_dir=tmp_path / "output")
    assert not (tmp_path / "output").exists()
