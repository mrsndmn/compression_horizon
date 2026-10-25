import json

import pytest
import torch
from tests.test_sps_causal_lm import tiny_args
from tests.test_sps_generation_semantics import SmallTokenizer


def experiment():
    import importlib

    try:
        return importlib.import_module("scripts.analyze_standard_additive")
    except ModuleNotFoundError as error:
        pytest.fail(f"Additive cramming implementation is missing: {error}")


def test_shared_vector_is_added_once_and_gradient_accumulates_all_positions():
    e = experiment()
    z = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    tokens = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    inputs = e.condition_inputs(z, tokens)
    torch.testing.assert_close(inputs, torch.tensor([[[1.0, 2.0], [11.0, 22.0], [31.0, 42.0]]]))
    inputs.sum().backward()
    torch.testing.assert_close(z.grad, torch.tensor([[[3.0, 3.0]]]))
    assert tokens.tolist() == [[[10.0, 20.0], [30.0, 40.0]]]


def test_generation_matches_independent_additive_history_and_teacher_forward():
    from compression_horizon.inference.generation import mask_native_generation_logits
    from compression_horizon.models.sps_causal_lm import SPSCausalLM

    e = experiment()
    torch.manual_seed(51)
    model = SPSCausalLM(tiny_args(), sps=False).eval().requires_grad_(False)
    tokenizer = SmallTokenizer()
    z = torch.randn(1, 1, 16)
    ids = torch.tensor([[1, 2, 3, 4]])
    result = e.check_sample(model, tokenizer, ids, z)
    teacher = model(inputs_embeds=torch.cat([z, model.get_input_embeddings()(ids) + z], 1)).logits[:, :-1]
    assert result["teacher_replay_correct_tokens"] == int((teacher.argmax(-1) == ids).sum())
    history = torch.empty(1, 0, dtype=torch.long)
    for _ in range(4):
        inputs = torch.cat([z, model.get_input_embeddings()(history) + z], 1)
        logits = model(inputs_embeds=inputs).logits[:, -1]
        token = mask_native_generation_logits(model, tokenizer, logits).argmax(-1)
        history = torch.cat([history, token[:, None]], 1)
        if token.item() == tokenizer.eos_token_id:
            break
    assert result["generated_ids"] == history[0].tolist()


def test_training_loss_and_gradient_match_explicit_additive_forward():
    from tests.trainer_helpers import _make_args

    from compression_horizon.models.sps_causal_lm import SPSCausalLM

    e = experiment()
    torch.manual_seed(72)
    model = SPSCausalLM(tiny_args(), sps=False).eval().requires_grad_(False)
    trainer = e.AdditiveCrammingTrainer(model=model, args=_make_args(), initializations={})
    ids = torch.tensor([[1, 2, 3]])
    tokens = model.get_input_embeddings()(ids)
    z = torch.randn(1, 1, 16, requires_grad=True)
    actual = trainer.forward_and_compute_loss(
        model,
        ids,
        tokens,
        torch.ones_like(ids),
        torch.cat([z, tokens], 1),
        torch.ones(1, 4, dtype=torch.long),
        1,
    )[0]
    logits = model(inputs_embeds=torch.cat([z, tokens + z], 1)).logits[:, :-1]
    expected = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids.flatten())
    torch.testing.assert_close(actual.squeeze(), expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), z)[0], torch.autograd.grad(expected, z)[0])


def test_early_eos_counts_missing_tokens_as_errors():
    from compression_horizon.models.sps_causal_lm import SPSCausalLM

    e = experiment()
    model = SPSCausalLM(tiny_args(), sps=False).eval().requires_grad_(False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    tokenizer = SmallTokenizer()
    tokenizer.eos_token_id = 0
    result = e.check_sample(model, tokenizer, torch.tensor([[0, 1, 2]]), torch.zeros(1, 1, 16))
    assert result["generated_ids"] == [0]
    assert result["greedy_accuracy"] == pytest.approx(1 / 3)
    assert result["teacher_replay_correct_tokens"] == 1
    assert not result["greedy_exact"]


@pytest.mark.parametrize("offset", [0, 6])
def test_trainer_keeps_weights_frozen_and_saves_paired_initialization(tmp_path, monkeypatch, offset):
    from datasets import load_from_disk
    from tests.trainer_helpers import _collate_batch, _make_args

    from compression_horizon.models.sps_causal_lm import SPSCausalLM

    e = experiment()
    monkeypatch.setattr(
        "compression_horizon.train.trainers.base.get_device",
        lambda: torch.device("cpu"),
    )
    torch.manual_seed(34)
    model = SPSCausalLM(tiny_args(), sps=False).eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    initial = torch.full((1, 16), 0.02)
    args = _make_args(
        output_dir=str(tmp_path),
        progressive_train=True,
        dtype="float32",
        progressive_min_seq_len=2,
        progressive_max_stages=1,
        max_optimization_steps_per_sample=2,
        max_optimization_steps_per_token=2,
        offset_dataset_items=offset,
    )
    trainer = e.AdditiveCrammingTrainer(
        model=model,
        processing_class=SmallTokenizer(),
        args=args,
        train_dataset=[
            dict(
                input_ids=torch.tensor([1, 2, 3]),
                attention_mask=torch.ones(3, dtype=torch.long),
            )
        ],
        data_collator=_collate_batch,
        initializations={offset: initial},
    )
    path = trainer.train()
    assert path == str(tmp_path / "additive_prefixes")
    row = load_from_disk(path)[0]
    assert row["embedding_role"] == "prefix_and_additive" and row["sample_id"] == offset
    torch.testing.assert_close(torch.tensor(row["initialization_embedding"]), initial)
    assert not torch.equal(torch.tensor(row["embedding"]), initial)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_dry_shards_cover_cohort_and_preserve_control_settings(monkeypatch, capsys):
    from scripts import analyze_sps_cramming as jobs

    e = experiment()
    monkeypatch.setattr(
        jobs.subprocess,
        "check_output",
        lambda cmd, **kw: "a" * 40 if "rev-parse" in cmd else b"",
    )
    e.submit(dry=True)
    remaining = capsys.readouterr().out.strip()
    payloads = []
    while remaining:
        payload, end = json.JSONDecoder().raw_decode(remaining)
        payloads.append(payload)
        remaining = remaining[end:].lstrip()
    assert len(payloads) == 8 and all(p["instance_type"] == "a100.1gpu" for p in payloads)
    ids = []
    for exp in e.experiment_configs:
        settings = e.resolved_training_args(exp)
        assert settings["per_device_train_batch_size"] == 1
        assert settings["token_objective"] == "squared_margin"
        start, n = settings["offset_dataset_items"], settings["limit_dataset_items"]
        ids.extend(range(start, start + n))
    assert sorted(ids) == list(range(48))


@pytest.fixture
def saved_additive(tmp_path, monkeypatch):
    from datasets import Dataset

    e = experiment()
    root = tmp_path / "shards"
    baseline = tmp_path / "control"
    monkeypatch.setattr(e, "ROOT", root)
    monkeypatch.setattr(e, "BASELINE", baseline)
    monkeypatch.setattr(e.jobs, "CACHE_ROOT", tmp_path / "cache")
    cache = e.jobs.CACHE_ROOT / e.Path(e.ordinary.TRAINING_ARGS["tokenized_dataset_path"]).name
    Dataset.from_list([dict(input_ids=[i + 1, 100 + i], attention_mask=[1, 1]) for i in range(48)]).save_to_disk(cache)
    baseline.mkdir()
    settings = dict(
        e.ordinary.TRAINING_ARGS,
        **e.CONTROL["training_args"],
        tokenized_dataset_path=str(cache),
    )
    (baseline / "design.json").write_text(
        json.dumps(dict(training_args=settings, experiment=dict(checkpoint=str(e.CHECKPOINT))))
    )
    initial = torch.full((1, 16), 0.02).tolist()
    Dataset.from_list([dict(sample_id=i, stage_index=0, initialization_embedding=initial) for i in range(48)]).save_to_disk(
        baseline / "progressive_prefixes"
    )
    controls = []
    for exp in e.experiment_configs:
        path = root / exp["name"]
        path.mkdir(parents=True)
        rows, samples = [], []
        args = e.MyTrainingArguments(**e.resolved_training_args(exp))
        for sid in range(exp["offset"], exp["offset"] + exp["count"]):
            for stage in range(2):
                rows.append(
                    dict(
                        sample_id=sid,
                        stage_index=stage,
                        stage_seq_len=stage + 1,
                        num_input_tokens=stage + 1,
                        embedding=[[0.1] * 16],
                        initialization_embedding=initial,
                        embedding_role=e.ROLE,
                        num_compression_tokens=1,
                        hidden_size=16,
                        model_checkpoint=str(e.CHECKPOINT),
                        steps_taken=stage + 1,
                        final_convergence=1.0,
                        convergence_threshold=1.0,
                        **{
                            key: getattr(args, key)
                            for key in (
                                "token_objective",
                                "loss_type",
                                "loss_margin",
                                "convergence_margin",
                                "dtype",
                                "max_optimization_steps_per_sample",
                            )
                        },
                    )
                )
            target = [sid + 1, sid + 100]
            sample = dict(
                sample_id=sid,
                horizon=2,
                stored_convergence=1.0,
                target_ids=target,
                generated_ids=target,
                teacher_replay_correct_tokens=2,
                teacher_replay_accuracy=1.0,
                greedy_accuracy=1.0,
                greedy_exact=True,
            )
            samples.append(sample)
            controls.append(
                dict(
                    sample,
                    horizon=1,
                    target_ids=target[:1],
                    generated_ids=target[:1],
                    teacher_replay_correct_tokens=1,
                )
            )
        Dataset.from_list(rows).save_to_disk(path / "additive_prefixes")
        report = dict(
            source_commit="a" * 40,
            experiment_name=exp["name"],
            training_args=e.resolved_training_args(exp),
            embedding_role=e.ROLE,
            trainable_scalars=16,
            initialization_source=str(baseline),
            n_samples=6,
            mean_horizon=2.0,
            greedy_exact_samples=6,
            teacher_replay_exact_samples=6,
            samples=samples,
        )
        (path / "single_sample_validation.json").write_text(json.dumps(report))
        (path / "job.json").write_text(
            json.dumps(
                dict(
                    commit="a" * 40,
                    result=dict(job_name=f"lm-mpi-job-00000000-0000-4000-8000-{exp['offset']:012x}"),
                )
            )
        )
        (path / "completed").touch()
    (baseline / "single_sample_validation.json").write_text(
        json.dumps(
            dict(
                n_samples=48,
                mean_horizon=1.0,
                greedy_exact_samples=48,
                teacher_replay_exact_samples=48,
                samples=controls,
            )
        )
    )
    (baseline / "completed").touch()
    return e, root


def test_merge_checks_complete_cohort_and_compares_with_control(saved_additive):
    from datasets import load_from_disk

    e, root = saved_additive
    merged = e.merge()
    report = json.loads((merged / "single_sample_validation.json").read_text())
    summary = json.loads((merged / "summary.json").read_text())
    assert [s["sample_id"] for s in report["samples"]] == list(range(48))
    assert len(load_from_disk(merged / "additive_prefixes")) == 96
    assert summary["paired_horizon_gain"]["mean"] == 1
    assert summary["mean_greedy_initial_prefix"] == 2
    assert (merged / "completed").exists() and (merged / "comparison.png").exists()


@pytest.mark.parametrize(
    "fault", ["incomplete", "snapshot", "target", "initialization", "counter", "role", "job_id", "duplicate_job"]
)
def test_merge_rejects_inconsistent_shards(saved_additive, fault):
    from datasets import Dataset, load_from_disk

    e, root = saved_additive
    path = root / e.experiment_configs[-1]["name"]
    report_path = path / "single_sample_validation.json"
    report = json.loads(report_path.read_text())
    if fault == "incomplete":
        (path / "completed").unlink()
    elif fault == "snapshot":
        report["source_commit"] = "b" * 40
    elif fault == "target":
        report["samples"][0]["target_ids"][0] += 1
    elif fault == "role":
        report["embedding_role"] = "ordinary_memory"
    elif fault in ("job_id", "duplicate_job"):
        job_path = path / "job.json"
        job = json.loads(job_path.read_text())
        job["result"]["job_name"] = "lm-mpi-job-123" if fault == "job_id" else "lm-mpi-job-00000000-0000-4000-8000-000000000000"
        job_path.write_text(json.dumps(job))
    else:
        raw_path = path / "additive_prefixes"
        rows = list(load_from_disk(raw_path))
        if fault == "initialization":
            rows[0]["initialization_embedding"][0][0] += 1
        else:
            rows[0]["steps_taken"] = -1
        import shutil

        shutil.rmtree(raw_path)
        Dataset.from_list(rows).save_to_disk(raw_path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        e.merge()
    assert not (root / "merged/completed").exists()
