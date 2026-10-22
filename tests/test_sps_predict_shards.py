import json
import shlex
import shutil
from pathlib import Path

import pytest
from datasets import Dataset, load_from_disk


def test_shard_launch_covers_original_cohort_exactly_once(monkeypatch, capsys):
    from scripts import analyze_sps_cramming as jobs
    from scripts import analyze_sps_predict_embedding as predict
    from scripts import analyze_sps_predict_shards as shards

    monkeypatch.setattr(jobs.subprocess, "check_output", lambda args, **kwargs: "a" * 40 if "rev-parse" in args else b"")
    shards.submit(dry=True)
    remaining = capsys.readouterr().out.strip()
    payloads = []
    while remaining:
        payload, end = json.JSONDecoder().raw_decode(remaining)
        payloads.append(payload)
        remaining = remaining[end:].lstrip()
    assert len(payloads) == 8
    names = []
    for payload in payloads:
        argv = shlex.split(payload["script"])
        assert payload["instance_type"] == "a100.1gpu"
        names.append(argv[argv.index("--experiment") + 1])
    assert set(names) == {exp["name"] for exp in shards.experiment_configs}
    ids = []
    for exp in shards.experiment_configs:
        settings = predict.resolved_training_args(exp)
        assert exp["init"] == "native" and settings["token_objective"] == "squared_margin"
        assert settings["per_device_train_batch_size"] == 1 and settings["limit_dataset_items"] == 6
        assert settings["max_optimization_steps_per_sample"] == 30000
        start = settings["offset_dataset_items"]
        ids.extend(range(start, start + settings["limit_dataset_items"]))
    assert sorted(ids) == list(range(48))


@pytest.fixture
def saved_shards(tmp_path, monkeypatch):
    from scripts import analyze_sps_cramming as jobs
    from scripts import analyze_sps_predict_embedding as predict
    from scripts import analyze_sps_predict_shards as shards

    monkeypatch.setattr(shards, "ROOT", tmp_path / "shards")
    monkeypatch.setattr(jobs, "CACHE_ROOT", tmp_path / "cache")
    control = tmp_path / "control"
    monkeypatch.setattr(shards, "experiment_configs", [dict(e, baseline=control) for e in shards.experiment_configs])
    cache_path = jobs.CACHE_ROOT / Path(predict.TRAINING_ARGS["tokenized_dataset_path"]).name
    Dataset.from_dict({"input_ids": [[sid + 1, sid + 101] + [1] * 4094 for sid in range(48)]}).save_to_disk(cache_path)
    common = dict(embedding_role="shared_predict", fixed_bos_token_id=50256, predict_token_id=50257, trainable_scalars=16)
    controls = []
    for index, exp in enumerate(shards.experiment_configs):
        path = shards.ROOT / exp["name"]
        path.mkdir(parents=True)
        settings = predict.resolved_training_args(exp)
        rows, samples = [], []
        start = settings["offset_dataset_items"]
        for sid in range(start, start + settings["limit_dataset_items"]):
            target = [sid + 1, sid + 101]
            generated = target if sid else [sid + 1, 999]
            samples.append(
                dict(
                    sample_id=sid,
                    horizon=2,
                    stored_convergence=1.0,
                    target_ids=target,
                    generated_ids=generated,
                    teacher_replay_correct_tokens=2,
                    teacher_replay_accuracy=1.0,
                    greedy_accuracy=1.0 if sid else 0.5,
                    greedy_exact=bool(sid),
                )
            )
            controls.append(
                dict(
                    sample_id=sid,
                    horizon=1,
                    target_ids=target[:1],
                    generated_ids=target[:1],
                    teacher_replay_correct_tokens=1,
                    teacher_replay_accuracy=1.0,
                    greedy_accuracy=1.0,
                    greedy_exact=True,
                )
            )
            for stage in range(2):
                rows.append(
                    dict(
                        sample_id=sid,
                        stage_index=stage,
                        stage_seq_len=stage + 1,
                        num_input_tokens=stage + 1,
                        steps_taken=stage + 1,
                        final_convergence=1.0,
                        convergence_threshold=1.0,
                        embedding=[[0.1] * 16],
                        initialization_embedding=[[0.01] * 16],
                        embedding_role="shared_predict",
                        fixed_bos_token_id=50256,
                        predict_init="native",
                        hidden_size=16,
                        num_compression_tokens=1,
                        token_objective="squared_margin",
                        loss_type="cross_entropy",
                        loss_margin=0.05,
                        convergence_margin=0.05,
                        max_optimization_steps_per_sample=30000,
                        dtype="bfloat16",
                        model_checkpoint=jobs.cluster_path(predict.CHECKPOINT),
                    )
                )
        Dataset.from_list(rows).save_to_disk(path / "predict_prefixes")
        report = dict(
            common,
            source_commit="a" * 40,
            experiment_name=exp["name"],
            training_args=settings,
            n_samples=6,
            mean_horizon=2.0,
            greedy_exact_samples=sum(s["greedy_exact"] for s in samples),
            teacher_replay_exact_samples=6,
            samples=samples,
        )
        (path / "single_sample_validation.json").write_text(json.dumps(report))
        (path / "job.json").write_text(
            json.dumps(dict(commit="a" * 40, result=dict(job_name=f"lm-mpi-job-00000000-0000-0000-0000-{index:012d}")))
        )
        (path / "completed").touch()
    control.mkdir()
    (control / "single_sample_validation.json").write_text(
        json.dumps(
            dict(
                common,
                n_samples=48,
                mean_horizon=1.0,
                greedy_exact_samples=48,
                teacher_replay_exact_samples=48,
                samples=controls,
            )
        )
    )
    (control / "completed").touch()
    return shards


def test_merge_keeps_global_ids_all_stages_and_recomputes_metrics(saved_shards):
    shards = saved_shards
    path = shards.merge()
    report = json.loads((path / "single_sample_validation.json").read_text())
    assert [s["sample_id"] for s in report["samples"]] == list(range(48))
    assert report["n_samples"] == 48 and report["mean_horizon"] == 2
    assert report["greedy_exact_samples"] == 47
    summary = json.loads((path / "summary.json").read_text())
    assert summary["mean_greedy_initial_prefix"] == pytest.approx(95 / 48)
    assert summary["paired_horizon_gain"]["mean"] == 1
    raw = load_from_disk(path / "predict_prefixes")
    assert len(raw) == 96 and sorted(set(raw["sample_id"])) == list(range(48))
    assert (path / "comparison.png").is_file() and (path / "completed").exists()
    before = (path / "single_sample_validation.json").stat().st_mtime_ns
    assert shards.merge() == path
    assert (path / "single_sample_validation.json").stat().st_mtime_ns == before


def test_existing_submissions_remain_idempotent_after_documentation_commit(saved_shards, monkeypatch):
    shards = saved_shards
    monkeypatch.setattr(shards, "source_commit", lambda: "b" * 40)

    def unexpected_submission(**kwargs):
        raise AssertionError("Existing shards must not submit new jobs")

    monkeypatch.setattr(shards.jobs, "submit", unexpected_submission)
    shards.submit()
    manifest = json.loads((shards.ROOT / "submission_manifest.json").read_text())
    assert manifest["source_commit"] == "a" * 40
    assert len(manifest["shards"]) == 8


def test_archived_shard_submission_uses_its_snapshot_even_after_head_changes(monkeypatch, capsys):
    from scripts import analyze_sps_cramming as jobs
    from scripts import analyze_sps_predict_shards as shards

    monkeypatch.setattr(shards, "source_commit", lambda: "a" * 40)

    def git_output(argv, **kwargs):
        if "rev-parse" in argv:
            return "b" * 40 if "HEAD" in argv else "a" * 40
        return b""

    monkeypatch.setattr(jobs.subprocess, "check_output", git_output)
    shards.submit(dry=True)
    remaining = capsys.readouterr().out.strip()
    while remaining:
        payload, end = json.JSONDecoder().raw_decode(remaining)
        remaining = remaining[end:].lstrip()
        assert "/code/" + "a" * 40 in payload["script"]
        assert "/code/" + "b" * 40 not in payload["script"]


@pytest.mark.parametrize(
    "corruption", ["incomplete", "duplicate_id", "wrong_target", "settings", "source_commit", "missing_stage", "shifted_stages"]
)
def test_merge_rejects_incomplete_or_inconsistent_shards(saved_shards, corruption):
    shards = saved_shards
    path = shards.ROOT / shards.experiment_configs[1]["name"]
    report_path = path / "single_sample_validation.json"
    report = json.loads(report_path.read_text())
    if corruption == "incomplete":
        (path / "completed").unlink()
    elif corruption == "duplicate_id":
        report["samples"][0]["sample_id"] = 0
    elif corruption == "wrong_target":
        report["samples"][0]["target_ids"][0] += 1
    elif corruption == "settings":
        report["training_args"]["learning_rate"] = 0.1
    elif corruption == "source_commit":
        report["source_commit"] = "c" * 40
    elif corruption in ("missing_stage", "shifted_stages"):
        raw_path = path / "predict_prefixes"
        rows = load_from_disk(raw_path).to_list()
        if corruption == "missing_stage":
            rows = rows[1:]
        else:
            for row in rows:
                row["stage_index"] += 200
                row["stage_seq_len"] += 200
        shutil.rmtree(raw_path)
        Dataset.from_list(rows).save_to_disk(raw_path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        shards.merge()
    assert not (shards.ROOT / "merged").exists()
