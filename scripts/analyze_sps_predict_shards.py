"""Native-init hinge cramming in disjoint single-GPU shards, then audited merge.

The durable Arkhip group resumes the session after the jobs finish; run --phase
merge then. No polling process or GPU allocation is needed to wait for shards.
"""

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import concatenate_datasets, load_from_disk
from scripts import analyze_sps_cramming as jobs
from scripts import analyze_sps_predict_embedding as predict
from scripts.analyze_sps_objective_factorial import leading_correct, paired_interval
from scripts.greedy_reconstruction_eval import select_rows
from scripts.sps_mechanism_common import source_commit

ROOT = jobs.ROOT / "predict_native_hinge_shards_20261022"
SOURCE = next(exp for exp in predict.budget_experiment_configs if exp["init"] == "native")
COHORT_SIZE = predict.TRAINING_ARGS["limit_dataset_items"]
SHARD_SIZE = 6
experiment_configs = [
    dict(
        SOURCE,
        name=SOURCE["name"] + f"_shard{index:02d}",
        training_args=dict(
            SOURCE["training_args"],
            offset_dataset_items=offset,
            limit_dataset_items=min(SHARD_SIZE, COHORT_SIZE - offset),
        ),
    )
    for index, offset in enumerate(range(0, COHORT_SIZE, SHARD_SIZE))
]
FIXED_SCHEDULE_ROOT = jobs.ROOT / "predict_native_hinge_fixed_schedule_20261023"
fixed_schedule_experiment_configs = [
    dict(
        exp,
        name=exp["name"].replace("_budget30000", "_fixed_schedule_budget30000"),
        training_args=dict(
            exp["training_args"],
            lr_scheduler_decay_steps=predict.TRAINING_ARGS["max_optimization_steps_per_sample"],
        ),
    )
    for exp in experiment_configs
]
SUITES = ("budget", "fixed_schedule")
META_FIELDS = ("embedding_role", "fixed_bos_token_id", "predict_token_id", "trainable_scalars")


def suite_settings(suite):
    if suite == "budget":
        return ROOT, experiment_configs
    if suite == "fixed_schedule":
        return FIXED_SCHEDULE_ROOT, fixed_schedule_experiment_configs
    raise ValueError(f"Unknown shard suite: {suite}")


def submit(dry=False, *, suite="budget"):
    root, configs = suite_settings(suite)
    manifest = []
    commit = source_commit()
    # A partial retry must use the same immutable training snapshot.
    if not dry:
        records = [root / exp["name"] / "job.json" for exp in configs]
        prior_commits = {json.loads(path.read_text())["commit"] for path in records if path.exists()}
        if len(prior_commits) > 1 or (
            not all(path.exists() for path in records) and prior_commits and prior_commits != {commit}
        ):
            raise ValueError("Existing shards use another commit; resume from their original snapshot")
    for exp in configs:
        jobs.OUTPUT = root / exp["name"]
        path = jobs.OUTPUT / "job.json"
        if not dry and path.exists():
            print("Existing submission:", jobs.OUTPUT, flush=True)
        else:
            jobs.submit(
                dry=dry,
                entrypoint="scripts/analyze_sps_predict_shards.py",
                job_kind="sps_" + exp["name"],
                probe_args=("--experiment", exp["name"], "--suite", suite),
                snapshot_commit=commit,
            )
        if not dry:
            manifest.append(dict(experiment=exp["name"], **json.loads(path.read_text())))
    if not dry:
        jobs.write_json(root / "submission_manifest.json", dict(source_commit=manifest[0]["commit"], shards=manifest))


def checked_samples(report, expected_ids, cache):
    samples = sorted(report["samples"], key=lambda sample: sample["sample_id"])
    if report["n_samples"] != len(expected_ids) or [s["sample_id"] for s in samples] != list(expected_ids):
        raise ValueError("Missing, duplicated or misassigned sample IDs")
    for sample in samples:
        target, generated = sample["target_ids"], sample["generated_ids"]
        if not target or target != cache[sample["sample_id"]]["input_ids"][: len(target)]:
            raise ValueError("Target IDs differ from the pinned cohort")
        correct = sum(a == b for a, b in zip(target, generated))
        if sample["greedy_exact"] != (target == generated) or not np.isclose(sample["greedy_accuracy"], correct / len(target)):
            raise ValueError("Greedy metrics disagree with saved IDs")
        replay = sample["teacher_replay_correct_tokens"]
        if not 0 <= replay <= len(target) or not np.isclose(sample["teacher_replay_accuracy"], replay / len(target)):
            raise ValueError("Invalid teacher replay metrics")
        if sample["horizon"] not in (0, len(target)):
            raise ValueError("Invalid selected horizon")
    if not np.isclose(report["mean_horizon"], np.mean([s["horizon"] for s in samples])):
        raise ValueError("Reported horizon mean mismatch")
    if report["greedy_exact_samples"] != sum(s["greedy_exact"] for s in samples):
        raise ValueError("Reported exact-generation count mismatch")
    if report["teacher_replay_exact_samples"] != sum(
        s["teacher_replay_correct_tokens"] == len(s["target_ids"]) for s in samples
    ):
        raise ValueError("Reported teacher replay count mismatch")
    return samples


def check_raw(raw, report, exp, samples):
    settings = predict.resolved_training_args(exp)
    args = predict.MyTrainingArguments(**settings)
    expected = {s["sample_id"] for s in samples}
    fields = [
        "sample_id",
        "stage_index",
        "stage_seq_len",
        "num_input_tokens",
        "steps_taken",
        "final_convergence",
        "convergence_threshold",
        "embedding_role",
        "fixed_bos_token_id",
        "predict_init",
        "hidden_size",
        "num_compression_tokens",
        "token_objective",
        "loss_type",
        "loss_margin",
        "convergence_margin",
        "max_optimization_steps_per_sample",
        "dtype",
        "model_checkpoint",
    ]
    seen, counters, next_stages = set(), defaultdict(int), defaultdict(int)
    for row in raw.select_columns(fields):
        key = (row["sample_id"], row["stage_index"])
        if row["sample_id"] not in expected or key in seen:
            raise ValueError("Raw stage IDs overlap or leave the assigned shard")
        seen.add(key)
        expected_stage = next_stages[row["sample_id"]]
        expected_length = min(args.progressive_min_seq_len + expected_stage * args.progressive_step, args.max_sequence_length)
        if (
            row["stage_index"] != expected_stage
            or row["stage_seq_len"] != expected_length
            or row["num_input_tokens"] != expected_length
            or row["convergence_threshold"] != args.progressive_convergence_threshold
        ):
            raise ValueError("Raw stages must be contiguous and follow the registered prefix progression")
        next_stages[row["sample_id"]] += 1
        delta = row["steps_taken"] - counters[row["sample_id"]]
        if (
            not 0 <= delta <= args.max_optimization_steps_per_token
            or row["steps_taken"] > args.max_optimization_steps_per_sample
        ):
            raise ValueError("Raw optimizer counters violate the registered budgets")
        counters[row["sample_id"]] = row["steps_taken"]
        for name in (
            "token_objective",
            "loss_type",
            "loss_margin",
            "convergence_margin",
            "max_optimization_steps_per_sample",
            "dtype",
        ):
            if row[name] != getattr(args, name):
                raise ValueError(f"Raw training setting mismatch: {name}")
        if (
            row["embedding_role"] != "shared_predict"
            or row["fixed_bos_token_id"] != report["fixed_bos_token_id"]
            or row["predict_init"] != "native"
            or row["hidden_size"] != report["trainable_scalars"]
            or row["num_compression_tokens"] != 1
            or jobs.cluster_path(row["model_checkpoint"]) != jobs.cluster_path(predict.CHECKPOINT)
            or not math.isfinite(row["final_convergence"])
        ):
            raise ValueError("Raw embedding role/model metadata mismatch")
    selected = select_rows(raw, "progr")
    if [row["sample_id"] for row in selected] != sorted(expected):
        raise ValueError("Incomplete raw sample cohort")
    for row, sample in zip(selected, samples):
        length = row["num_input_tokens"]
        horizon = length if row["final_convergence"] >= row["convergence_threshold"] else 0
        if (
            len(sample["target_ids"]) != length
            or sample["horizon"] != horizon
            or sample["stored_convergence"] != row["final_convergence"]
        ):
            raise ValueError("Selected raw stage disagrees with validation")
        vector = np.asarray(row["embedding"])
        if vector.shape != (1, report["trainable_scalars"]) or not np.isfinite(vector).all():
            raise ValueError("Invalid saved predict vector")


def merge(*, suite="budget"):
    root, configs = suite_settings(suite)
    output = root / "merged"
    if (output / "completed").exists():
        print("Already merged:", output, flush=True)
        return output
    if output.exists():
        raise ValueError("Existing incomplete merge destination; inspect before replacing")
    paths = [root / exp["name"] for exp in configs]
    if not all((path / "completed").exists() for path in paths):
        raise ValueError("Incomplete shards; merge requires every successful completion marker")
    cache = load_from_disk(predict.resolved_training_args(configs[0])["tokenized_dataset_path"])
    control_path = configs[0]["baseline"]
    if not (control_path / "completed").exists():
        raise ValueError("Incomplete original-budget control")
    control = json.loads((control_path / "single_sample_validation.json").read_text())
    controls = checked_samples(control, range(COHORT_SIZE), cache)
    metadata = {name: control[name] for name in META_FIELDS}
    if metadata["embedding_role"] != "shared_predict":
        raise ValueError("Control must contain shared predict vectors")
    samples, datasets, provenance = [], [], []
    training_commit = None
    for exp, path in zip(configs, paths):
        report = json.loads((path / "single_sample_validation.json").read_text())
        settings = predict.resolved_training_args(exp)
        if report["experiment_name"] != exp["name"] or report["training_args"] != settings:
            raise ValueError("Shard settings differ from the experiment registry")
        if any(report[name] != metadata[name] for name in META_FIELDS):
            raise ValueError("Shard model/embedding metadata mismatch")
        job = json.loads((path / "job.json").read_text())
        if training_commit is None:
            training_commit = report["source_commit"]
        if report["source_commit"] != training_commit or job["commit"] != training_commit:
            raise ValueError("Shards must share one committed training snapshot")
        start, count = settings["offset_dataset_items"], settings["limit_dataset_items"]
        part = checked_samples(report, range(start, start + count), cache)
        raw = load_from_disk(path / "predict_prefixes")
        check_raw(raw, report, exp, part)
        samples.extend(part)
        datasets.append(raw)
        provenance.append(
            dict(
                experiment=exp["name"],
                path=str(path),
                job_id=job["result"]["job_name"],
                source_commit=training_commit,
                training_args=settings,
            )
        )
    samples.sort(key=lambda sample: sample["sample_id"])
    if [s["sample_id"] for s in samples] != list(range(COHORT_SIZE)):
        raise ValueError("Merged cohort has missing or overlapping samples")
    report = dict(
        metadata,
        source_commit=training_commit,
        merge_commit=source_commit(),
        n_samples=len(samples),
        mean_horizon=float(np.mean([s["horizon"] for s in samples])),
        greedy_exact_samples=sum(s["greedy_exact"] for s in samples),
        teacher_replay_exact_samples=sum(s["teacher_replay_correct_tokens"] == len(s["target_ids"]) for s in samples),
        samples=samples,
    )
    checked_samples(report, range(COHORT_SIZE), cache)
    prefixes = [leading_correct(s["target_ids"], s["generated_ids"]) for s in samples]
    control_prefixes = [leading_correct(s["target_ids"], s["generated_ids"]) for s in controls]
    summary = dict(
        suite=suite,
        n_samples=len(samples),
        mean_horizon=report["mean_horizon"],
        baseline_horizon=control["mean_horizon"],
        mean_greedy_initial_prefix=float(np.mean(prefixes)),
        baseline_greedy_initial_prefix=float(np.mean(control_prefixes)),
        greedy_exact_samples=report["greedy_exact_samples"],
        paired_horizon_gain=paired_interval([a["horizon"] - b["horizon"] for a, b in zip(samples, controls)]),
        paired_greedy_prefix_gain=paired_interval(np.asarray(prefixes) - control_prefixes),
    )
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=root) as temporary:
        staged = Path(temporary) / "merged"
        staged.mkdir()
        concatenate_datasets(datasets).sort(["sample_id", "stage_index"]).save_to_disk(staged / "predict_prefixes")
        jobs.write_json(staged / "single_sample_validation.json", report)
        jobs.write_json(staged / "summary.json", summary)
        jobs.write_json(staged / "manifest.json", dict(shards=provenance, control=str(control_path)))
        with (staged / "per_sample.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "horizon", "greedy_initial_prefix", "greedy_exact"])
            writer.writeheader()
            for sample, prefix in zip(samples, prefixes):
                writer.writerow(
                    dict(
                        sample_id=sample["sample_id"],
                        horizon=sample["horizon"],
                        greedy_initial_prefix=prefix,
                        greedy_exact=sample["greedy_exact"],
                    )
                )
        plot_comparison(summary, staged / "comparison.png")
        (staged / "completed").touch()
        staged.rename(output)
    print(json.dumps(summary, indent=2), flush=True)
    return output


def plot_comparison(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    ax.bar(x - 0.18, [summary["baseline_horizon"], summary["baseline_greedy_initial_prefix"]], 0.36, label="Original budget")
    label = "Larger budget / merged shards"
    if summary.get("suite") == "fixed_schedule":
        label = "Larger budget / original LR then constant"
    ax.bar(x + 0.18, [summary["mean_horizon"], summary["mean_greedy_initial_prefix"]], 0.36, label=label)
    ax.set_xticks(x, ["Teacher-forced horizon", "Initial correct greedy prefix"])
    ax.set(ylabel="Mean tokens", title="SPS / shared predict / native init / hinge / batch 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["dry", "submit", "probe", "merge"], default="dry")
    parser.add_argument("--suite", choices=SUITES, default="budget")
    parser.add_argument("--experiment")
    cli = parser.parse_args()
    root, configs = suite_settings(cli.suite)
    if cli.phase == "probe":
        if cli.experiment not in {exp["name"] for exp in configs}:
            parser.error("--experiment must name an experiment in the selected suite")
        predict.probe(next(exp for exp in configs if exp["name"] == cli.experiment), output_root=root)
    else:
        {
            "dry": lambda: submit(dry=True, suite=cli.suite),
            "submit": lambda: submit(suite=cli.suite),
            "merge": lambda: merge(suite=cli.suite),
        }[cli.phase]()
