"""Audit one complete pretrained cramming cohort without inference or submission.

The executable experiment design remains pretrained_sps_experiments.py. Reports
are checked against raw stages and pinned native target IDs, including failed
stages and short generations. Partial model cohorts are never summarized.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from scripts.pretrained_sps_experiments import (
    COHORT,
    ROOT,
    SHARD_SIZE,
    TRAINING_ARGS,
    code_digest,
    experiment_configs,
    read_json,
    write_json,
)
from sps_paper.reproduce import JOB, SHA, score_sample

REPO = Path(__file__).resolve().parents[1]
STAGE_FIELDS = (
    "sample_id",
    "stage_index",
    "stage_seq_len",
    "num_input_tokens",
    "final_convergence",
    "convergence_threshold",
    "steps_taken",
    "num_compression_tokens",
    "hidden_size",
    "loss_type",
    "token_objective",
    "loss_margin",
    "convergence_margin",
    "dtype",
    "model_checkpoint",
    "max_optimization_steps_per_sample",
)


def canonical(value):
    if isinstance(value, str):
        return value.replace("/mnt/virtual_ai0001053-00054_SR004-nfs2", "/workspace-SR004.nfs2")
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def audit_sample(stages, sample, target_ids, settings, *, valid_length):
    """Pure audit of one sample; no model, tokenizer or GPU is needed."""
    if not stages:
        raise ValueError("No saved stages")
    if any(type(sample[key]) is not int for key in ("sample_id", "horizon", "selected_length", "steps_taken")):
        raise ValueError("Selected identity/length/update fields must be integers")
    stages = sorted(stages, key=lambda r: r["stage_index"])
    previous = 0
    successful = []
    first_failed = None
    for index, stage in enumerate(stages):
        if any(
            type(stage[key]) is not int
            for key in ("sample_id", "stage_index", "stage_seq_len", "num_input_tokens", "steps_taken")
        ):
            raise ValueError("Stage identity/length/update fields must be integers")
        if (
            stage["stage_index"] != index
            or stage["stage_seq_len"] != index + 1
            or stage["num_input_tokens"] != index + 1
            or stage["sample_id"] != sample["sample_id"]
            or stage["convergence_threshold"] != 1.0
            or stage["num_input_tokens"] > valid_length
        ):
            raise ValueError("Stage identity, order or reconstruction threshold mismatch")
        updates = stage["steps_taken"]
        if not (
            0 <= updates - previous <= settings["max_optimization_steps_per_token"]
            and updates <= settings["max_optimization_steps_per_sample"]
        ):
            raise ValueError("Stage update counters violate the registered budget")
        if index != len(stages) - 1 and (
            updates == settings["max_optimization_steps_per_sample"]
            or updates - previous == settings["max_optimization_steps_per_token"]
        ):
            raise ValueError("Unexpected stages after budget exhaustion")
        previous = updates
        if not 0 <= stage["final_convergence"] <= 1:
            raise ValueError("Invalid convergence value")
        if stage["final_convergence"] >= stage["convergence_threshold"]:
            successful.append(stage)
        elif first_failed is None:
            first_failed = stage["stage_seq_len"]
        if first_failed is not None and index != len(stages) - 1:
            raise ValueError("Unexpected stages after a single-sample failed stage")
    selected = successful[-1] if successful else stages[-1]
    length = selected["num_input_tokens"]
    horizon = length if successful else 0
    if (
        sample["selected_length"] != length
        or sample["horizon"] != horizon
        or sample["steps_taken"] != selected["steps_taken"]
        or sample["stored_convergence"] != selected["final_convergence"]
        or sample["target_ids"] != list(target_ids[:length])
        or length > valid_length
        or len(sample["generated_ids"]) > length
    ):
        raise ValueError("Selected validation disagrees with saved stages or pinned target IDs")
    scores = score_sample(sample)
    if horizon and not scores["teacher_exact"]:
        raise ValueError("A converged selected vector does not replay exactly")
    terminal = stages[-1]
    terminal_delta = terminal["steps_taken"] - (stages[-2]["steps_taken"] if len(stages) > 1 else 0)
    if terminal["steps_taken"] == settings["max_optimization_steps_per_sample"]:
        stop = "total_budget"
    elif terminal_delta == settings["max_optimization_steps_per_token"]:
        # The trainer remeasures after the last update without updating its
        # convergence state. Even a successful final measurement stops here.
        stop = "stage_budget"
    elif terminal["final_convergence"] < terminal["convergence_threshold"]:
        raise ValueError("Failed terminal stage did not consume its stage budget")
    elif terminal["num_input_tokens"] == valid_length:
        stop = "sequence_cap" if valid_length == settings["max_sequence_length"] else "document_end"
    else:
        raise ValueError("Unexplained terminal stage")
    return dict(
        sample_id=sample["sample_id"],
        **scores,
        selected_length=length,
        selected_steps=selected["steps_taken"],
        total_steps=terminal["steps_taken"],
        first_failed_length=first_failed,
        terminal_stage_steps=terminal_delta,
        stop_reason=stop,
        n_stages=len(stages),
    )


def audit_experiment(exp, *, output_dir=None):
    from datasets import load_from_disk
    from scripts.pretrained_sps_worker import validate_data

    fingerprint = code_digest(REPO)
    data, manifest = validate_data(exp)
    config = read_json(Path(exp["source_path"]) / "config.json")
    records, source_shards, seen_jobs, seen_commits = [], [], set(), set()
    for offset in range(0, COHORT, SHARD_SIZE):
        path = ROOT / "runs" / fingerprint[:12] / exp["name"] / f"shard{offset // SHARD_SIZE:02d}"
        if not (path / "completed").is_file():
            raise ValueError(f"Incomplete cohort: {path}")
        job = read_json(path / "job.json")
        job_id, commit = job["result"]["job_name"], job["commit"]
        if not JOB.fullmatch(job_id) or job_id in seen_jobs or not SHA.fullmatch(commit):
            raise ValueError("Invalid or duplicate job UUID/source commit")
        seen_jobs.add(job_id)
        if commit not in seen_commits:
            if code_digest(ROOT / "code" / commit) != fingerprint:
                raise ValueError("Archived worker code differs from the audit registry")
            seen_commits.add(commit)
        report_path = path / "single_sample_validation.json"
        report = read_json(report_path)
        if not math.isfinite(report["elapsed_seconds"]) or report["elapsed_seconds"] < 0:
            raise ValueError("Invalid worker runtime")
        expected = dict(
            TRAINING_ARGS,
            model_checkpoint=exp["checkpoint"] or exp["source"],
            offset_dataset_items=offset,
            limit_dataset_items=SHARD_SIZE,
        )
        settings = {k: v for k, v in report["settings"].items() if k not in {"output_dir", "logging_dir"}}
        role = "ordinary_memory" if exp["arm"] == "ar" else "shared_predict"
        if (
            canonical(report["experiment"]) != canonical(exp)
            or canonical(settings) != canonical(expected)
            or report["code_digest"] != fingerprint
            or report["source_commit"] != commit
            or report["data_manifest"] != manifest
            or report["embedding_role"] != role
            or report["bos_token_id"] != config["bos_token_id"]
        ):
            raise ValueError("Shard design, provenance, data identity or conditioning site mismatch")
        expected_ids = set(range(offset, offset + SHARD_SIZE))
        samples = report["samples"]
        if len(samples) != SHARD_SIZE or {r["sample_id"] for r in samples} != expected_ids:
            raise ValueError("Incomplete or duplicate validation sample set")
        dirname = "progressive_prefixes" if exp["arm"] == "ar" else "predict_prefixes"
        stages = load_from_disk(str(path / dirname)).select_columns(
            STAGE_FIELDS + (() if exp["arm"] == "ar" else ("embedding_role", "fixed_bos_token_id", "predict_init"))
        )
        grouped = defaultdict(list)
        for stage in stages:
            if (
                stage["sample_id"] not in expected_ids
                or stage["num_compression_tokens"] != 1
                or stage["hidden_size"] != config["hidden_size"]
            ):
                raise ValueError("Raw-stage sample identity or vector dimensions mismatch")
            for key in (
                "loss_type",
                "token_objective",
                "loss_margin",
                "convergence_margin",
                "dtype",
                "model_checkpoint",
                "max_optimization_steps_per_sample",
            ):
                if canonical(stage[key]) != canonical(expected[key]):
                    raise ValueError(f"Raw-stage training argument mismatch: {key}")
            if exp["arm"] != "ar" and (
                stage["embedding_role"] != role
                or stage["predict_init"] != "native"
                or stage["fixed_bos_token_id"] != config["bos_token_id"]
            ):
                raise ValueError("Raw-stage prediction interface mismatch")
            grouped[stage["sample_id"]].append(stage)
        for sample in samples:
            sid = sample["sample_id"]
            record = audit_sample(
                grouped[sid], sample, data[sid]["input_ids"], expected, valid_length=sum(data[sid]["attention_mask"])
            )
            record["document_sha256"] = manifest["document_hashes"][sid]
            records.append(record)
        source_shards.append(
            dict(
                job_id=job_id,
                source_commit=commit,
                path=str(path),
                validation_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                elapsed_seconds=report["elapsed_seconds"],
            )
        )
    records.sort(key=lambda r: r["sample_id"])
    if [r["sample_id"] for r in records] != list(range(COHORT)):
        raise ValueError("Incomplete or duplicate cohort")
    analysis_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    committed_source = subprocess.run(
        ["git", "show", analysis_commit + ":" + str(Path(__file__).relative_to(REPO))], cwd=REPO, capture_output=True
    )
    if committed_source.returncode or committed_source.stdout != Path(__file__).read_bytes():
        analysis_commit = None
    summary = dict(
        schema=1,
        experiment=exp,
        code_digest=fingerprint,
        n_samples=COHORT,
        mean_horizon=statistics.mean(r["horizon"] for r in records),
        median_horizon=statistics.median(r["horizon"] for r in records),
        mean_greedy_initial_prefix=statistics.mean(r["greedy_initial_prefix"] for r in records),
        greedy_exact_samples=sum(r["greedy_exact"] for r in records),
        teacher_exact_samples=sum(r["teacher_exact"] for r in records),
        mean_total_steps=statistics.mean(r["total_steps"] for r in records),
        stopping_counts=dict(Counter(r["stop_reason"] for r in records)),
        worker_seconds=dict(
            min=min(r["elapsed_seconds"] for r in source_shards),
            max=max(r["elapsed_seconds"] for r in source_shards),
            sum=sum(r["elapsed_seconds"] for r in source_shards),
        ),
        source_shards=source_shards,
        samples=records,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_source_commit=analysis_commit,
    )
    destination = Path(output_dir) if output_dir is not None else ROOT / "analysis" / fingerprint[:12] / exp["name"]
    write_json(destination / "summary.json", summary)
    with (destination / "per_sample.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=[exp["name"] for exp in experiment_configs], required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = audit_experiment(next(e for e in experiment_configs if e["name"] == args.experiment), output_dir=args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k not in {"samples", "source_shards"}}, indent=2))


if __name__ == "__main__":
    main()
