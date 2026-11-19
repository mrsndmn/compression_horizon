"""Submit and analyze matched memory/shared-prediction cramming across SPS checkpoints."""

import argparse
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scripts import analyze_sps_cramming as jobs
from scripts import analyze_sps_predict_embedding as predict
from scripts.activation_distillation import load_or_create_tokenized_dataset
from scripts.analyze_pretrained_sps import audit_sample
from scripts.analyze_sps_predict_shards import check_raw, checked_samples
from scripts.sps_mechanism_common import source_commit
from scripts.validation.validate_sps_batch1_run import validate_run
from sps_paper.reproduce import STAGE_FIELDS, _job, digest, json_read, load_snapshot, paired_interval
from transformers import DataCollatorForLanguageModeling

from compression_horizon.models.sps_causal_lm import load_sps_checkpoint, load_sps_tokenizer
from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.train.trainers.progressive_cramming import ProgressiveCrammingTrainer

ROOT = jobs.ROOT / "pretraining_progress_20261120"
CONTROL = next(exp for exp in predict.experiment_configs if exp["name"] == "predict_hinge_native")
CONTROL_SETTINGS = predict.resolved_training_args(CONTROL)
COHORT_SIZE = CONTROL_SETTINGS["limit_dataset_items"]
SHARD_SIZE = 12
PRETRAINING_TOKENS = (250085376, 500170752, 750256128, 1000341504)
UPDATE_CUTOFFS = (1000, 2500, 5000, 7500, CONTROL_SETTINGS["max_optimization_steps_per_sample"])
SITES = {"memory": "sps_hinge_strict", "predict": "predict_hinge_native"}
experiment_configs = [
    dict(
        name=f"pretraining_{tokens}_{site}_{offset:02d}",
        tokens=tokens,
        site=site,
        checkpoint=str(predict.CHECKPOINT.with_name(f"ckpt_tokens_{tokens}.pt")),
        init=CONTROL["init"] if site == "predict" else "paired_memory",
        training_args=dict(
            CONTROL["training_args"],
            offset_dataset_items=offset,
            limit_dataset_items=min(SHARD_SIZE, COHORT_SIZE - offset),
        ),
    )
    for tokens in PRETRAINING_TOKENS[:-1]
    for site in SITES
    for offset in range(0, COHORT_SIZE, SHARD_SIZE)
]


def memory_initializations():
    baseline = CONTROL["baseline"]
    design = json_read(baseline / "design.json")
    settings = dict(design["training_args"])
    settings["tokenized_dataset_path"] = str(jobs.cluster_path(settings["tokenized_dataset_path"]))
    if (
        not (baseline / "completed").is_file()
        or settings != CONTROL_SETTINGS
        or jobs.cluster_path(design["experiment"]["checkpoint"]) != jobs.cluster_path(predict.CHECKPOINT)
        or design["commit"] != _job(baseline)["source_commit"]
    ):
        raise ValueError("Historical memory control does not match the pilot settings")
    raw = load_from_disk(str(baseline / "progressive_prefixes"))
    initial_rows = [
        row for row in raw.select_columns(["sample_id", "stage_index", "initialization_embedding"]) if row["stage_index"] == 0
    ]
    if sorted(row["sample_id"] for row in initial_rows) != list(range(COHORT_SIZE)):
        raise ValueError("Historical memory control has incomplete initializations")
    return {row["sample_id"]: row["initialization_embedding"] for row in initial_rows}


def historical_controls(cache):
    snapshot = load_snapshot(jobs.REPO / "sps_paper/data")
    if snapshot["cohort"] != [dict(sample_id=sid, input_ids=cache[sid]["input_ids"]) for sid in range(COHORT_SIZE)]:
        raise ValueError("The experiment cache differs from the audited cohort")
    anchors = []
    for site, path in (("memory", CONTROL["baseline"]), ("predict", predict.ROOT / CONTROL["name"])):
        run = snapshot["runs"][SITES[site]]
        job = _job(path)
        report_path = path / "single_sample_validation.json"
        report = json_read(report_path)
        relative_report = str(report_path.relative_to(jobs.ROOT))
        if (
            not (path / "completed").is_file()
            or run["source"] != str(path.relative_to(jobs.ROOT))
            or run["model"] != "sps"
            or run["site"] != {"memory": "ordinary_memory", "predict": "shared_predict"}[site]
            or run["source_commit"] != report.get("training_commit", report.get("source_commit"))
            or [job] != [{key: record[key] for key in job} for record in run["jobs"]]
            or digest(report_path) != snapshot["manifest"]["artifact_sha256"][relative_report]
            or (
                site == "predict"
                and (
                    run["selector"] != CONTROL["name"]
                    or report["embedding_role"] != "shared_predict"
                    or report.get("training_args", CONTROL_SETTINGS) != CONTROL_SETTINGS
                )
            )
        ):
            raise ValueError(f"Historical control provenance mismatch for {path}")
        anchors.append(dict(site=site, checkpoint=str(predict.CHECKPOINT), **job))
    return snapshot, anchors


class PairedMemoryTrainer(ProgressiveCrammingTrainer):
    def __init__(self, *args, initializations, **kwargs):
        super().__init__(*args, **kwargs)
        self.initializations = initializations

    def _setup_batch(self, batch, ctx, *, sample_id_counter=0):
        result = super()._setup_batch(batch, ctx, sample_id_counter=sample_id_counter)
        sample_id = sample_id_counter + self.args.offset_dataset_items
        parameter = result.parametrization.parameters[0]
        initial = torch.as_tensor(self.initializations[sample_id], device=parameter.device, dtype=parameter.dtype)
        if initial.shape != parameter.shape[1:] or not torch.isfinite(initial).all():
            raise ValueError("Invalid historical memory initialization")
        with torch.no_grad():
            parameter.copy_(initial.unsqueeze(0))
        result.initialization_embeddings = result.parametrization.materialize().detach().clone().cpu()
        return result


def probe(exp, root=ROOT):
    output = root / exp["name"]
    if (output / "completed").exists():
        raise FileExistsError(output / "completed")
    if exp["site"] == "predict":
        predict.probe(exp, output_root=root)
        return
    torch.set_num_threads(8)
    settings = predict.resolved_training_args(exp)
    checkpoint = str(jobs.cluster_path(exp["checkpoint"]))
    args = MyTrainingArguments(**settings, model_checkpoint=checkpoint, output_dir=str(output))
    model = load_sps_checkpoint(checkpoint, torch_dtype=torch.bfloat16)
    tokenizer = load_sps_tokenizer()
    dataset = load_or_create_tokenized_dataset(
        cache_dir="artifacts/cache/tokenized_datasets",
        dataset_name=args.dataset_name,
        split="test",
        tokenizer=tokenizer,
        max_sequence_length=args.max_sequence_length,
        model_checkpoint=checkpoint,
        no_bos_token=args.no_bos_token,
        limit_dataset_items=args.limit_dataset_items,
        offset_dataset_items=args.offset_dataset_items,
        tokenized_dataset_path=args.tokenized_dataset_path,
    )
    trainer = PairedMemoryTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        initializations=memory_initializations(),
    )
    jobs.write_json(output / "design.json", dict(commit=source_commit(), experiment=exp, training_args=settings))
    trainer.train()
    del trainer, model
    torch.cuda.empty_cache()
    validate_run(output)
    (output / "completed").touch()


def submit(*, dry=False):
    records = [ROOT / exp["name"] / "job.json" for exp in experiment_configs]
    earlier_commits = {json_read(record)["commit"] for record in records if record.is_file()}
    commit = source_commit()
    if not dry and (len(earlier_commits) > 1 or (not all(p.is_file() for p in records) and earlier_commits - {commit})):
        raise ValueError("Existing shards use a different source snapshot")
    if not dry:
        historical_controls(load_from_disk(CONTROL_SETTINGS["tokenized_dataset_path"]))
        memory_initializations()
    for exp, record in zip(experiment_configs, records, strict=True):
        if record.exists() and not dry:
            continue
        jobs.OUTPUT = record.parent
        jobs.submit(
            dry=dry,
            entrypoint="scripts/analyze_sps_pretraining.py",
            job_kind=exp["name"],
            probe_args=("--experiment", exp["name"]),
            snapshot_commit=commit,
        )
    if not dry:
        jobs.write_json(ROOT / "submission_manifest.json", dict(shards=[_job(record.parent) for record in records]))


def audit_shard(exp, output, cache, initializations, commit):
    settings = predict.resolved_training_args(exp)
    start = settings["offset_dataset_items"]
    report = json_read(output / "single_sample_validation.json")
    checkpoint = jobs.cluster_path(exp["checkpoint"])
    if not (output / "completed").is_file() or jobs.cluster_path(report["model_checkpoint"]) != checkpoint:
        raise ValueError(f"Missing completion or wrong checkpoint for {output}")
    samples = checked_samples(report, range(start, start + settings["limit_dataset_items"]), cache)
    if exp["site"] == "predict":
        if report["experiment_name"] != exp["name"] or report["training_args"] != settings or report["source_commit"] != commit:
            raise ValueError(f"Shared-prediction report provenance mismatch for {output}")
        raw = load_from_disk(str(output / "predict_prefixes"))
        check_raw(raw, report, exp, samples)
        return samples, list(raw.select_columns(STAGE_FIELDS))
    design = dict(commit=commit, experiment=exp, training_args=settings)
    if json_read(output / "design.json") != design or report["training_commit"] != commit:
        raise ValueError(f"Memory report provenance mismatch for {output}")
    raw = load_from_disk(str(output / "progressive_prefixes"))
    for row in raw.select_columns(
        [*STAGE_FIELDS, "initialization_embedding", "model_checkpoint", "token_objective", "loss_margin", "convergence_margin"]
    ):
        if (
            jobs.cluster_path(row["model_checkpoint"]) != checkpoint
            or any(row[key] != settings[key] for key in ("token_objective", "loss_margin", "convergence_margin"))
            or (
                row["stage_index"] == 0
                and not np.array_equal(row["initialization_embedding"], initializations[start + row["sample_id"]])
            )
        ):
            raise ValueError(f"Memory stage provenance mismatch for {output}")
    return samples, [dict(row, sample_id=start + row["sample_id"]) for row in raw.select_columns(STAGE_FIELDS)]


def audit_run(samples, stages, cache, settings):
    grouped = defaultdict(list)
    for row in stages:
        grouped[row["sample_id"]].append(row)
    if sorted(grouped) != sorted(s["sample_id"] for s in samples):
        raise ValueError("Stage IDs differ from the validated samples")
    scores = []
    for sample in sorted(samples, key=lambda s: s["sample_id"]):
        sid = sample["sample_id"]
        rows = sorted(grouped[sid], key=lambda r: r["stage_index"])
        converged = [r for r in rows if r["final_convergence"] >= r["convergence_threshold"]]
        selected = (converged or rows)[-1]
        targets = cache[sid]
        replay = dict(
            sample,
            selected_length=len(sample["target_ids"]),
            steps_taken=selected["steps_taken"],
            stored_convergence=sample.get("stored_convergence", sample.get("stored_teacher_accuracy")),
        )
        replay.setdefault("teacher_replay_correct_tokens", round(sample["teacher_replay_accuracy"] * len(sample["target_ids"])))
        scores.append(audit_sample(rows, replay, targets["input_ids"], settings, valid_length=sum(targets["attention_mask"])))
    return scores


def summary_record(memory, shared, first_gaps):
    memory_scores, shared_scores = memory["scores"], shared["scores"]
    gaps = np.asarray([b["horizon"] - a["horizon"] for a, b in zip(memory_scores, shared_scores, strict=True)])
    if first_gaps is None:
        first_gaps = gaps
    common_lengths = [min(a["horizon"], b["horizon"]) for a, b in zip(memory_scores, shared_scores, strict=True)]
    result = dict(
        paired_horizon_gain=paired_interval(gaps),
        gain_change_from_earliest=paired_interval(gaps - first_gaps),
    )
    per_site = dict(memory=memory, predict=shared)
    for site, run in per_site.items():
        scores = run["scores"]
        # Use the saved final vectors and truncate their greedy outputs to a shared length.
        comparable = [(s, length) for s, length in zip(scores, common_lengths, strict=True) if length]
        result[site] = dict(
            mean_horizon=float(np.mean([s["horizon"] for s in scores])),
            mean_greedy_prefix=float(np.mean([min(s["greedy_initial_prefix"], s["horizon"]) for s in scores])),
            greedy_exact_at_common_length=sum(s["greedy_initial_prefix"] >= length for s, length in comparable),
            n_common_lengths=len(comparable),
            mean_updates=float(np.mean([s["total_steps"] for s in scores])),
            total_budget_stops=sum(s["stop_reason"] == "total_budget" for s in scores),
            stage_budget_stops=sum(s["stop_reason"] == "stage_budget" for s in scores),
        )
    result["paired_gain_at_update_cutoff"] = {}
    for cutoff in UPDATE_CUTOFFS:
        horizons = {}
        for site, run in per_site.items():
            reached = np.zeros(COHORT_SIZE, dtype=int)
            for row in run["stages"]:
                if row["steps_taken"] <= cutoff and row["final_convergence"] >= row["convergence_threshold"]:
                    sid = row["sample_id"]
                    reached[sid] = max(reached[sid], row["stage_seq_len"])
            horizons[site] = reached
        result["paired_gain_at_update_cutoff"][cutoff] = dict(
            paired_gain=paired_interval(horizons["predict"] - horizons["memory"]),
            mean_horizons={site: float(values.mean()) for site, values in horizons.items()},
        )
    return result, first_gaps


def merge():
    cache = load_from_disk(CONTROL_SETTINGS["tokenized_dataset_path"])
    snapshot, anchors = historical_controls(cache)
    limits = dict(
        total_updates=CONTROL_SETTINGS["max_optimization_steps_per_sample"],
        stage_updates=CONTROL_SETTINGS["max_optimization_steps_per_token"],
    )
    runs = defaultdict(lambda: dict(samples=[], stages=[], measurement_limits=limits))
    initializations = memory_initializations()
    provenance = []
    for exp in experiment_configs:
        output = ROOT / exp["name"]
        job = _job(output)
        samples, stages = audit_shard(exp, output, cache, initializations, job["source_commit"])
        run = runs[exp["tokens"], exp["site"]]
        run["samples"].extend(samples)
        run["stages"].extend(stages)
        provenance.append(dict(experiment=exp["name"], checkpoint=exp["checkpoint"], **job))
    if len({job["source_commit"] for job in provenance}) != 1:
        raise ValueError("The sweep mixes different training snapshots")
    for site, key in SITES.items():
        runs[PRETRAINING_TOKENS[-1], site] = snapshot["runs"][key]
    for run in runs.values():
        samples = sorted(run["samples"], key=lambda s: s["sample_id"])
        if [s["sample_id"] for s in samples] != list(range(COHORT_SIZE)) or run["measurement_limits"] != limits:
            raise ValueError("Misaligned cohort or cramming budget")
        run["scores"] = audit_run(samples, run["stages"], cache, CONTROL_SETTINGS)
    first_gaps = None
    rows = []
    for tokens in PRETRAINING_TOKENS:
        row, first_gaps = summary_record(runs[tokens, "memory"], runs[tokens, "predict"], first_gaps)
        rows.append(dict(pretraining_tokens=tokens, **row))
    jobs.write_json(ROOT / "summary.json", dict(rows=rows, jobs=provenance, anchors=anchors, merge_commit=source_commit()))
    plot(rows)


def plot(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (horizon, progress) = plt.subplots(1, 2, figsize=(10, 4))
    tokens = [row["pretraining_tokens"] / 1e6 for row in rows]
    for site, label in (("memory", "Ordinary memory"), ("predict", "Shared prediction")):
        horizon.plot(tokens, [row[site]["mean_horizon"] for row in rows], marker="o", label=label)
    horizon.set(xlabel="Pretraining tokens (millions)", ylabel="Mean teacher-forced horizon (tokens)")
    horizon.legend()
    for row, count in zip(rows, tokens, strict=True):
        values = row["paired_gain_at_update_cutoff"]
        progress.plot(
            UPDATE_CUTOFFS,
            [values[steps]["paired_gain"]["mean"] for steps in UPDATE_CUTOFFS],
            marker="o",
            label=f"{count:.0f}M",
        )
    progress.set(xlabel="Optimizer updates", ylabel="Shared prediction minus memory (tokens)")
    progress.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "comparison.png", dpi=160)
    plt.close(fig)


def smoke():
    with tempfile.TemporaryDirectory(prefix="sps-pretraining-smoke-") as directory:
        for site in SITES:
            exp = next(
                e
                for e in experiment_configs
                if e["tokens"] == PRETRAINING_TOKENS[0] and e["site"] == site and e["training_args"]["offset_dataset_items"] > 0
            )
            settings = dict(
                exp["training_args"],
                limit_dataset_items=1,
                warmup_steps=0,
                max_optimization_steps_per_sample=6,
                max_optimization_steps_per_token=6,
            )
            exp = dict(exp, training_args=settings)
            probe(exp, Path(directory))
            cache = load_from_disk(CONTROL_SETTINGS["tokenized_dataset_path"])
            samples, stages = audit_shard(exp, Path(directory) / exp["name"], cache, memory_initializations(), source_commit())
            audit_run(samples, stages, cache, predict.resolved_training_args(exp))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dry", "probe", "smoke", "submit", "merge"), default="dry")
    parser.add_argument("--experiment", choices=[e["name"] for e in experiment_configs])
    cli = parser.parse_args()
    if cli.phase == "probe":
        if not cli.experiment:
            parser.error("--experiment is required for probe")
        probe(next(e for e in experiment_configs if e["name"] == cli.experiment))
    else:
        {"dry": lambda: submit(dry=True), "smoke": smoke, "submit": submit, "merge": merge}[cli.phase]()
