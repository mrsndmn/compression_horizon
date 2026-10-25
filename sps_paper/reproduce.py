#!/usr/bin/env python
"""Audit completed SPS evidence, render offline, or print historical GPU replay commands.

No CLI command submits jobs. Only export imports the existing experiment registries
and datasets; rendering needs numpy/matplotlib and the versioned measurements.
Experiment design belongs to the original Python registries, never this module.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 20000
SHA = re.compile(r"^[0-9a-f]{40}$")
JOB = re.compile(r"^lm-mpi-job-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
PROBES = {
    "routes": "mechanism_routes_20261022",
    "geometry": "mechanism_margin_geometry_20261022",
    "rescue": "mechanism_margin_rescue_20261022",
    "gradients": "mechanism_objective_gradients_20261022",
    "ablation": "memory_ablation_20261022",
}
STAGE_FIELDS = (
    "sample_id",
    "stage_index",
    "stage_seq_len",
    "num_input_tokens",
    "steps_taken",
    "final_convergence",
    "convergence_threshold",
    "max_optimization_steps_per_sample",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_read(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def score_sample(sample):
    """Count short outputs as errors; a recovery after the first error is not a prefix."""
    target, generated = sample["target_ids"], sample["generated_ids"]
    if not target or sample["horizon"] not in (0, len(target)):
        raise ValueError("Invalid selected horizon")
    if any(type(token) is not int or token < 0 for token in target + generated):
        raise ValueError("Invalid token IDs")
    correct = sum(a == b for a, b in zip(target, generated))
    exact = target == generated
    prefix = next((i for i, (a, b) in enumerate(zip(target, generated)) if a != b), min(len(target), len(generated)))
    if sample["greedy_exact"] != exact or not math.isclose(sample["greedy_accuracy"], correct / len(target), abs_tol=1e-7):
        raise ValueError("Greedy metrics disagree with token IDs")
    replay = sample["teacher_replay_correct_tokens"]
    if type(replay) is not int or not 0 <= replay <= len(target):
        raise ValueError("Invalid teacher replay count")
    if not math.isclose(sample["teacher_replay_accuracy"], replay / len(target), abs_tol=1e-7):
        raise ValueError("Teacher replay fraction disagrees with count")
    return dict(
        horizon=sample["horizon"],
        greedy_exact=exact,
        greedy_initial_prefix=prefix,
        greedy_correct_tokens=correct,
        greedy_accuracy=correct / len(target),
        teacher_exact=replay == len(target),
    )


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite aligned text measurements")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    return dict(mean=float(values.mean()), ci95=np.quantile(values[samples].mean(axis=1), [0.025, 0.975]).tolist())


def _job(path):
    record = json_read(path / "job.json")
    job_id = record["result"]["job_name"]
    if not JOB.fullmatch(job_id) or not SHA.fullmatch(record["commit"]):
        raise ValueError(f"Invalid full job UUID or source SHA: {path}")
    return dict(job_id=job_id, source_commit=record["commit"])


def _complete(path):
    if not (path / "completed").is_file():
        raise ValueError(f"Incomplete run: missing completed marker: {path}")


def export_snapshot(artifacts_root, destination):
    """Export actual completed runs selected by existing source registries.

    Snapshots keep measurements and their provenance, not a second experiment
    registry. Exact token IDs, all scalar stage counters and diagnostic raw data
    permit independent offline metrics. Model weights/optimized vectors stay external.
    """
    root = Path(artifacts_root).resolve()
    # Fail early and do not write a publication manifest for partial exports.
    _complete(root / PROBES["routes"])
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from datasets import load_from_disk
    from scripts import analyze_sps_predict_embedding as predict
    from scripts import analyze_sps_predict_shards as shards
    from scripts.jobs import run_jobs_sps as ordinary

    sources = {}

    def read(path):
        sources[str(path.relative_to(root))] = digest(path)
        return json_read(path)

    def completed(path):
        _complete(path)
        sources[str((path / "completed").relative_to(root))] = digest(path / "completed")

    def dataset(path):
        for file in sorted(path.iterdir()):
            if file.is_file() and file.suffix in (".arrow", ".json"):
                sources[str(file.relative_to(root))] = digest(file)
        return load_from_disk(str(path))

    cache_path = Path(ordinary.TRAINING_ARGS["tokenized_dataset_path"])
    if not cache_path.exists():
        # Preserve the pinned cache basename and immutable source SHA while relocating mount.
        cache_path = root / "code" / cache_path.parts[-5] / "artifacts/cache/tokenized_datasets" / cache_path.name
    cache = load_from_disk(str(cache_path))
    cohort = [
        dict(sample_id=i, input_ids=list(cache[i]["input_ids"])) for i in range(ordinary.TRAINING_ARGS["limit_dataset_items"])
    ]
    cache_sources = {p.name: digest(p) for p in sorted(cache_path.iterdir()) if p.is_file()}
    runs = {}

    def add(key, path, *, model, site, cell, settings, registry, selector, jobs=None):
        completed(path)
        report = read(path / "single_sample_validation.json")
        if site == "ordinary_memory":
            design = read(path / "design.json")
            expected_checkpoint = next(
                e["checkpoint"]
                for e in ordinary.experiment_configs
                if e["name"].startswith("standard" if model == "standard" else "sps_w0")
            )
            if (
                design["training_args"] != settings
                or design["commit"] != report["training_commit"]
                or design["experiment"]["name"] != selector
                or design["experiment"]["checkpoint"] != expected_checkpoint
            ):
                raise ValueError("Historical ordinary-memory design differs from registry/report")
        if jobs is None:
            read(path / "job.json")
            jobs = [_job(path)]
        commit = report.get("training_commit", report.get("source_commit"))
        if {j["source_commit"] for j in jobs} != {commit}:
            raise ValueError(f"Validation/source commit mismatch: {path}")
        subdir = "predict_prefixes" if site == "shared_predict" else "progressive_prefixes"
        raw = dataset(path / subdir)
        if site == "shared_predict" and report["embedding_role"] != "shared_predict":
            raise ValueError("Wrong latent-vector role")
        for field, default in (("token_objective", "cross_entropy"), ("loss_margin", 0.0), ("convergence_margin", 0.0)):
            expected_value = settings.get(field, default)
            if field in raw.column_names:
                if any(value != expected_value for value in raw[field]):
                    raise ValueError("Raw objective differs from original registry")
            elif expected_value != default:
                raise ValueError("Missing raw objective evidence")
        stages = list(raw.select_columns(STAGE_FIELDS))
        # Hash the vectors that matter for trajectory equivalence, rather than
        # publishing thousands of unnecessary 768-dimensional optimization states.
        vector_hashes = []
        for row in (
            raw.select_columns(["sample_id", "stage_index", "embedding", "initialization_embedding"])
            if key in ("predict_hinge_native", "predict_fixed")
            else []
        ):
            vector_hashes.append(
                dict(
                    sample_id=row["sample_id"],
                    stage_index=row["stage_index"],
                    embedding_sha256=hashlib.sha256(np.asarray(row["embedding"], dtype="<f4").tobytes()).hexdigest(),
                    initialization_sha256=hashlib.sha256(
                        np.asarray(row["initialization_embedding"], dtype="<f4").tobytes()
                    ).hexdigest(),
                )
            )
        if "training_args" in report:
            # Single-shard reports contain local limits/offsets; merged reports do not.
            expected = dict(settings)
            if report["training_args"] != expected:
                raise ValueError("Saved settings differ from original registry")
        samples = report["samples"]
        for sample in samples:
            if "teacher_replay_correct_tokens" not in sample:
                count = sample["teacher_replay_accuracy"] * len(sample["target_ids"])
                # Legacy reports store a float32 mean rather than the integer
                # count; account for its rounding error scaled by target length.
                tolerance = max(1e-6, len(sample["target_ids"]) * np.finfo(np.float32).eps)
                if not math.isclose(count, round(count), abs_tol=tolerance):
                    raise ValueError("Historical teacher count is not integral")
                sample["teacher_replay_correct_tokens"] = round(count)
        if report["n_samples"] != len(cohort) or not math.isclose(
            report["mean_horizon"], np.mean([s["horizon"] for s in samples])
        ):
            raise ValueError("Saved report aggregate disagrees with samples")
        if report["greedy_exact_samples"] != sum(score_sample(s)["greedy_exact"] for s in samples):
            raise ValueError("Saved greedy exact aggregate disagrees with IDs")
        runs[key] = dict(
            model=model,
            site=site,
            cell=cell,
            source=str(path.relative_to(root)),
            jobs=jobs,
            registry=registry,
            selector=selector,
            source_commit=commit,
            measurement_limits=dict(
                total_updates=settings["max_optimization_steps_per_sample"],
                stage_updates=settings["max_optimization_steps_per_token"],
            ),
            samples=samples,
            stages=stages,
            vector_hashes=vector_hashes,
        )

    for base in ordinary.experiment_configs:
        model = "standard" if base["name"].startswith("standard") else "sps"
        add(
            model + "_ce_baseline",
            root / ordinary.BASELINE_RUNS[base["name"]].name,
            model=model,
            site="ordinary_memory",
            cell="ce_baseline",
            settings=dict(ordinary.TRAINING_ARGS),
            registry="ordinary:experiment_configs",
            selector=base["name"],
        )
        basepath = root / ordinary.OBJECTIVE_BASELINES[base["name"]].name
        exp = next(e for e in ordinary.lr_sweep_experiment_configs if e["name"] == base["name"] + "_lr0.3")
        add(
            model + "_ce_raw",
            basepath,
            model=model,
            site="ordinary_memory",
            cell="ce_raw",
            settings=dict(ordinary.TRAINING_ARGS, **exp["training_args"]),
            registry="ordinary:lr_sweep_experiment_configs",
            selector=exp["name"],
        )
        for exp in ordinary.objective_experiment_configs:
            if exp["checkpoint"] != base["checkpoint"]:
                continue
            paths = [p for p in root.glob(exp["name"] + "_*") if (p / "design.json").is_file()]
            if len(paths) != 1:
                raise ValueError(f"Missing or ambiguous completed objective run: {exp['name']}")
            design = read(paths[0] / "design.json")
            expected = dict(ordinary.TRAINING_ARGS, **exp["training_args"])
            if design["training_args"] != expected or design["experiment"] != exp:
                raise ValueError("Historical objective design differs from registry")
            cell = exp["name"].removeprefix(base["name"] + "_")
            add(
                model + "_" + cell,
                paths[0],
                model=model,
                site="ordinary_memory",
                cell=cell,
                settings=expected,
                registry="ordinary:objective_experiment_configs",
                selector=exp["name"],
            )
        for exp in ordinary.lr_sweep_experiment_configs:
            if exp["checkpoint"] != base["checkpoint"] or exp["training_args"]["learning_rate"] == 0.3:
                continue
            paths = [p for p in root.glob(exp["name"] + "_*") if (p / "design.json").is_file()]
            if len(paths) != 1:
                raise ValueError("Incomplete historical LR sweep")
            add(
                model + "_lr" + str(exp["training_args"]["learning_rate"]),
                paths[0],
                model=model,
                site="ordinary_memory",
                cell="ce_raw",
                settings=dict(ordinary.TRAINING_ARGS, **exp["training_args"]),
                registry="ordinary:lr_sweep_experiment_configs",
                selector=exp["name"],
            )
    for exp in predict.experiment_configs:
        add(
            exp["name"],
            root / predict.ROOT.name / exp["name"],
            model="sps",
            site="shared_predict",
            cell="ce_raw" if "_ce_" in exp["name"] else "hinge_strict",
            settings=predict.resolved_training_args(exp),
            registry="predict:experiment_configs",
            selector=exp["name"],
        )
    for suite, key in (("budget", "predict_stretched"), ("fixed_schedule", "predict_fixed")):
        suite_root, exps = shards.suite_settings(suite)
        merge_path = root / suite_root.name / "merged"
        manifest = read(merge_path / "manifest.json")
        jobs = []
        for exp, source in zip(exps, manifest["shards"], strict=True):
            shardpath = root / suite_root.name / exp["name"]
            completed(shardpath)
            record = read(shardpath / "job.json")
            if record["result"]["job_name"] != source["job_id"] or record["commit"] != source["source_commit"]:
                raise ValueError("Shard provenance differs from merged manifest")
            settings = predict.resolved_training_args(exp)
            if source["training_args"] != settings:
                raise ValueError("Merged settings differ from source registry")
            jobs.append(dict(job_id=source["job_id"], source_commit=source["source_commit"], selector=exp["name"]))
        add(
            key,
            merge_path,
            model="sps",
            site="shared_predict",
            cell="hinge_strict",
            settings=predict.resolved_training_args(exps[0]),
            registry="shards:" + suite,
            selector=suite,
            jobs=jobs,
        )
    probes = {}
    for name, dirname in PROBES.items():
        path = root / dirname
        completed(path)
        raw = read(path / "probe.json")
        read(path / "job.json")
        job = _job(path)
        if raw["source_commit"] != job["source_commit"]:
            raise ValueError("Probe/source commit mismatch")
        probes[name] = dict(data=raw, job=job, source=dirname)
    evidence = dict(cohort=cohort, runs=runs, probes=probes)
    derive(evidence)  # Audit before publishing any usable manifest.
    destination = Path(destination)
    files = {}
    for name, payload in evidence.items():
        file = destination / (name + ".json.gz")
        json_write(file, payload)
        files[file.name] = digest(file)
    dependencies = {}
    for package in ("numpy", "matplotlib", "torch", "datasets", "transformers", "accelerate", "scipy"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not installed"
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        files=files,
        artifact_sha256=sources,
        cache=dict(
            source=str(cache_path),
            source_sha256=cache_sources,
            immutable_cache_commit="2683216db54e0ae8f51a69fa467c2dc07f5509f9",
            scope="Exact original first 48 cached input_ids; no retokenization",
        ),
        dependencies=dependencies,
        dependencies_scope="Exporter environment; historical lockfiles are in each pinned source commit",
        provenance="Measurements from existing completed artifacts; no retraining or inference during export",
        excluded="Checkpoints, optimized vectors and training optimizer states remain external; vector digests audit equality only",
    )
    json_write(destination / "manifest.json", manifest)
    return manifest


def load_snapshot(data_root):
    root = Path(data_root)
    manifest = json_read(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported evidence schema")
    for file, expected in manifest["files"].items():
        path = root / file
        if path.parent != root or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Evidence hash mismatch: {file}")
    result = {name: json_read(root / (name + ".json.gz")) for name in ("cohort", "runs", "probes")}
    result["manifest"] = manifest
    return result


def _stages(run, scores):
    groups = defaultdict(list)
    for row in run["stages"]:
        groups[row["sample_id"]].append(row)
    if sorted(groups) != list(range(len(scores))):
        raise ValueError("Missing or duplicated stage sample IDs")
    stops, positions, per_text, correlations = [], defaultdict(list), [], []
    for sid, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["stage_index"])
        last_steps, last_horizon = 0, 0
        successful = []
        failed = False
        for index, row in enumerate(rows):
            if (
                row["stage_index"] != index
                or row["stage_seq_len"] != index + 1
                or row["num_input_tokens"] != index + 1
                or failed
            ):
                raise ValueError("Stages must grow one token without gaps, duplicates or post-failure rows")
            steps = row["steps_taken"]
            delta = steps - last_steps
            if (
                not 0 <= delta <= run["measurement_limits"]["stage_updates"]
                or steps > run["measurement_limits"]["total_updates"]
            ):
                raise ValueError("Invalid stage optimizer counter")
            if row["max_optimization_steps_per_sample"] != run["measurement_limits"]["total_updates"]:
                raise ValueError("Inconsistent stage budget")
            fraction, threshold = row["final_convergence"], row["convergence_threshold"]
            if (
                not 0 <= fraction <= 1
                or not 0 <= threshold <= 1
                or not math.isclose(fraction * (index + 1), round(fraction * (index + 1)), abs_tol=1e-4)
            ):
                raise ValueError("Convergence is not an integer-token fraction")
            converged = fraction >= threshold
            if converged:
                last_horizon = row["stage_seq_len"]
                successful.append((row["stage_seq_len"], delta))
                positions[row["stage_seq_len"]].append(delta)
            last_steps = steps
            failed = not converged
        if last_horizon != scores[sid]["horizon"]:
            raise ValueError("Selected horizon disagrees with converged raw stages")
        stops.append(
            dict(
                sample_id=sid,
                final_length=rows[-1]["stage_seq_len"],
                steps=last_steps,
                last_stage_updates=delta,
                total_budget_hit=last_steps == run["measurement_limits"]["total_updates"],
                stage_cap_hit=delta == run["measurement_limits"]["stage_updates"],
                failed=failed,
                final_fraction=rows[-1]["final_convergence"],
                previous_converged=len(rows) > 1 and rows[-2]["final_convergence"] >= rows[-2]["convergence_threshold"],
            )
        )
        per_text.append(successful)
        if len(successful) >= 5:
            increments = np.asarray([delta for _, delta in successful])
            _, inverse, counts = np.unique(increments, return_inverse=True, return_counts=True)
            ranks = (np.cumsum(counts) - counts / 2 - 0.5)[inverse]
            if np.std(ranks) > 0:
                correlations.append(float(np.corrcoef(np.arange(len(ranks)), ranks)[0, 1]))
    blocks = []
    for start, end in ((1, 10), (11, 25), (26, 50), (51, 75), (76, 100), (101, 125)):
        eligible = [i for i, s in enumerate(scores) if s["horizon"] >= end]
        values = [delta for sid in eligible for length, delta in per_text[sid] if start <= length <= end]
        if values:
            blocks.append(
                dict(
                    start=start, end=end, n_samples=len(eligible), mean=float(np.mean(values)), median=float(np.median(values))
                )
            )
    return dict(
        stops=stops,
        mean_final_updates=float(np.mean([s["steps"] for s in stops])),
        total_budget_hit=sum(s["total_budget_hit"] for s in stops),
        exclusive_stage_cap=sum(s["stage_cap_hit"] and not s["total_budget_hit"] for s in stops),
        position_cost=[
            dict(
                position=p,
                n=len(v),
                median=float(np.median(v)),
                q25=float(np.quantile(v, 0.25)),
                q75=float(np.quantile(v, 0.75)),
            )
            for p, v in sorted(positions.items())
        ],
        fixed_cohort_blocks=blocks,
        median_within_text_spearman=float(np.median(correlations)) if correlations else None,
        positive_within_text_fraction=float(np.mean(np.asarray(correlations) > 0)) if correlations else None,
    )


def derive(data):
    cohort, results, vectors = data["cohort"], {}, {}
    if [s["sample_id"] for s in cohort] != list(range(48)):
        raise ValueError("Expected the exact original cohort of 48 texts")
    for key, run in data["runs"].items():
        if not SHA.fullmatch(run["source_commit"]) or not run["jobs"]:
            raise ValueError("Missing immutable source/job provenance")
        for job in run["jobs"]:
            if not JOB.fullmatch(job["job_id"]) or job["source_commit"] != run["source_commit"]:
                raise ValueError("Invalid full job UUID or source SHA")
        samples = sorted(run["samples"], key=lambda row: row["sample_id"])
        if [s["sample_id"] for s in samples] != list(range(48)):
            raise ValueError("Missing or duplicated sample IDs")
        scores = []
        for sid, sample in enumerate(samples):
            if sample["target_ids"] != cohort[sid]["input_ids"][: len(sample["target_ids"])]:
                raise ValueError("Target IDs differ from the pinned cohort")
            scores.append(score_sample(sample))
        horizon = np.array([s["horizon"] for s in scores])
        vectors[key] = horizon
        results[key] = dict(
            horizon=paired_interval(horizon),
            greedy_initial_prefix=paired_interval([s["greedy_initial_prefix"] for s in scores]),
            greedy_exact=sum(s["greedy_exact"] for s in scores),
            teacher_exact=sum(s["teacher_exact"] for s in scores),
            n=48,
            **_stages(run, scores),
        )
    paired = {}
    for cell in ("ce_raw", "ce_strict", "hinge_raw", "hinge_strict"):
        paired["standard_minus_sps_" + cell] = paired_interval(vectors["standard_" + cell] - vectors["sps_" + cell])
    for a, b in (
        ("predict_fixed", "predict_hinge_native"),
        ("predict_stretched", "predict_hinge_native"),
        ("predict_fixed", "predict_stretched"),
        ("predict_hinge_native", "standard_hinge_strict"),
        ("predict_hinge_native", "sps_hinge_strict"),
    ):
        delta = vectors[a] - vectors[b]
        paired[a + "_minus_" + b] = dict(
            **paired_interval(delta),
            improved=int((delta > 0).sum()),
            unchanged=int((delta == 0).sum()),
            worse=int((delta < 0).sum()),
        )
    stops = results["predict_fixed"]["stops"]
    violations = [s["final_length"] - round(s["final_fraction"] * s["final_length"]) for s in stops]
    terminal = dict(
        n=48,
        median_margin_violations=float(np.median(violations)),
        range_margin_violations=[min(violations), max(violations)],
        mean_fraction_meeting_margin=float(np.mean([s["final_fraction"] for s in stops])),
        previous_converged=sum(s["previous_converged"] for s in stops),
        all_transitions_add_one_token=all(
            s["final_length"] == int(vectors["predict_fixed"][s["sample_id"]]) + 1 for s in stops
        ),
        minimum_previous_position_violations=min(violations) - 1,
        median_previous_position_violations_lower_bound=float(np.median(violations)) - 1,
        samples=[dict(**s, margin_violations=n) for s, n in zip(stops, violations)],
        interpretation="Positive-margin violations in separately reoptimized final vectors, not argmax errors or positional attribution",
    )
    routes = {}
    route_data = data["probes"]["routes"]["data"]
    for model, measured in route_data["models"].items():
        rows = measured["samples"]
        baseline = [np.mean(s["baseline"]["nll"][1:]) for s in rows]
        routes[model] = dict(
            n=len(rows),
            controls={
                name: float(np.mean([np.mean(s["controls"][name]["nll"][1:]) - b for s, b in zip(rows, baseline)]))
                for name in rows[0]["controls"]
            },
            layers={
                name: np.asarray([[np.mean(v["nll"][1:]) - b for v in s["knockout"][name]] for s, b in zip(rows, baseline)])
                .mean(0)
                .tolist()
                for name in rows[0]["knockout"]
            },
            query_change=np.mean([s["query_context_change"] for s in rows], axis=0).tolist(),
        )
    rescue_rows = data["probes"]["rescue"]["data"]["samples"]
    rescue = {}
    for name in rescue_rows[0]["arms"]:
        rescue[name] = {
            dtype: sum(
                score_sample(
                    dict(sample_id=s["sample_id"], horizon=s["target_length"], **s["arms"][name][dtype + "_validation"])
                )["greedy_exact"]
                for s in rescue_rows
            )
            for dtype in ("fp32", "bf16")
        }
    trajectory = {}
    old = {(s["sample_id"], s["stage_index"]): s for s in data["runs"]["predict_hinge_native"]["stages"]}
    new = {(s["sample_id"], s["stage_index"]): s for s in data["runs"]["predict_fixed"]["stages"]}
    old_hashes = {(s["sample_id"], s["stage_index"]): s for s in data["runs"]["predict_hinge_native"]["vector_hashes"]}
    new_hashes = {(s["sample_id"], s["stage_index"]): s for s in data["runs"]["predict_fixed"]["vector_hashes"]}
    common = [k for k in old.keys() & new.keys() if old[k]["steps_taken"] < 10000 and new[k]["steps_taken"] < 10000]
    trajectory["initializations_equal"] = sum(
        old_hashes[(sid, 0)]["initialization_sha256"] == new_hashes[(sid, 0)]["initialization_sha256"] for sid in range(48)
    )
    trajectory["first_stage_vectors_equal"] = sum(
        old_hashes[(sid, 0)]["embedding_sha256"] == new_hashes[(sid, 0)]["embedding_sha256"] for sid in range(48)
    )
    trajectory["pre_boundary_pairs"] = len(common)
    trajectory["pre_boundary_vectors_equal"] = sum(
        old_hashes[k]["embedding_sha256"] == new_hashes[k]["embedding_sha256"] for k in common
    )
    trajectory["pre_boundary_steps_equal"] = sum(old[k]["steps_taken"] == new[k]["steps_taken"] for k in common)
    trajectory["pre_boundary_convergence_equal"] = sum(
        old[k]["final_convergence"] == new[k]["final_convergence"] for k in common
    )
    geometry = {}
    for model, measured in data["probes"]["geometry"]["data"]["models"].items():
        for state in ("warm_start", "optimized"):
            rows = [s["states"][state] for s in measured["samples"]]
            geometry[model + "/" + state] = dict(
                n=len(rows),
                full_row_rank=sum(
                    min(r["normalized_singular_values"]) > max(r["normalized_singular_values"]) * 1e-6 for r in rows
                ),
                mean_free_ratio=float(np.mean([r["new_gradient_free_ratio"] for r in rows])),
                mean_negative_cosine_fraction=float(np.mean([r["negative_cosine_fraction"] for r in rows])),
                mean_new_margin=float(np.mean([r["margins"][-1] for r in rows])),
                median_required_relative_step=float(np.median([r["required_linear_step_l2"] / r["memory_norm"] for r in rows])),
            )
    gradient_rows = data["probes"]["gradients"]["data"]["samples"]
    gradients = {
        point: {
            key: float(np.mean([s["points"][point][key] for s in gradient_rows]))
            for key in ("ce", "wrong_tokens", "cancellation_ratio")
        }
        for point in gradient_rows[0]["points"]
    }
    gradients["hinge_perfect_but_higher_ce_than_incorrect_ce03"] = sum(
        s["points"]["adam_hinge_lr03"]["wrong_tokens"] == 0
        and s["points"]["adam_ce_lr03"]["wrong_tokens"] > 0
        and s["points"]["adam_hinge_lr03"]["ce"] > s["points"]["adam_ce_lr03"]["ce"]
        for s in gradient_rows
    )
    ablation = {}
    for model, measured in data["probes"]["ablation"]["data"]["models"].items():
        for arithmetic, rows in measured["arithmetic"].items():
            for family in ("single", "prefix", "suffix"):
                for metric in ("nll", "correct"):
                    base = [np.mean(s["baseline"][metric][1:]) for s in rows]
                    values = np.asarray(
                        [[np.mean(v[metric][1:]) - b for v in s["interventions"][family]] for s, b in zip(rows, base)]
                    )
                    ablation["/".join((model, arithmetic, family, metric))] = (
                        values.mean(0) * (100 if metric == "correct" else 1)
                    ).tolist()
    return dict(
        runs=results,
        paired=paired,
        terminal=terminal,
        routes=routes,
        rescue=rescue,
        trajectory_audit=trajectory,
        geometry=geometry,
        gradients=gradients,
        ablation=ablation,
        intervals="95% percentile bootstrap, 20000 fixed-seed text resamples; paired contrasts align sample IDs; fixed checkpoints/seed, not seed uncertainty",
    )


def _interval(row):
    return f"{row['mean']:.2f} [{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}]"


def _table(path, headers, rows):
    lines = [
        "% Generated by sps_paper/reproduce.py; edit measurements or generator.",
        r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(str(value) for value in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def render(data_root=HERE / "data", output_root=HERE):
    data, output = load_snapshot(data_root), Path(output_root)
    metrics = derive(data)
    runs = metrics["runs"]
    table_root, figure_root = output / "tables", output / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    main = [
        ("Standard / memory", "CE / argmax", "standard_ce_raw"),
        ("Standard / memory", "Hinge / margin", "standard_hinge_strict"),
        ("SPS / memory", "CE / argmax", "sps_ce_raw"),
        ("SPS / memory", "Hinge / margin", "sps_hinge_strict"),
        ("SPS / predict (native)", "CE / argmax", "predict_ce_native"),
        ("SPS / predict (native)", "Hinge / margin", "predict_hinge_native"),
        ("SPS / predict (random)", "Hinge / margin", "predict_hinge_random"),
    ]
    _table(
        table_root / "main_results.tex",
        ["Model / site", "Objective / stop", "TF horizon [95\\% CI]", "Exact", "Initial prefix", "Updates"],
        [
            (
                label,
                objective,
                _interval(runs[k]["horizon"]),
                f"{runs[k]['greedy_exact']}/48",
                f"{runs[k]['greedy_initial_prefix']['mean']:.2f}",
                f"{runs[k]['mean_final_updates']:.0f}",
            )
            for label, objective, k in main
        ],
    )
    cells = [
        ("CE / argmax", "ce_raw"),
        ("CE / margin", "ce_strict"),
        ("Hinge / argmax", "hinge_raw"),
        ("Hinge / margin", "hinge_strict"),
    ]
    _table(
        table_root / "objective_factorial.tex",
        ["Objective / stop", "Standard TF [95\\% CI]", "SPS TF [95\\% CI]", r"Paired gap [95\% CI]"],
        [
            (
                label,
                _interval(runs["standard_" + cell]["horizon"]),
                _interval(runs["sps_" + cell]["horizon"]),
                _interval(metrics["paired"]["standard_minus_sps_" + cell]),
            )
            for label, cell in cells
        ],
    )
    budgets = [
        ("Original", "predict_hinge_native"),
        ("Stretched LR", "predict_stretched"),
        ("Fixed LR + floor", "predict_fixed"),
    ]
    _table(
        table_root / "budget_control.tex",
        ["Schedule", "TF horizon [95\\% CI]", "Initial prefix", "Exact", "Budget stop", "Stage stop"],
        [
            (
                label,
                _interval(runs[k]["horizon"]),
                f"{runs[k]['greedy_initial_prefix']['mean']:.2f}",
                f"{runs[k]['greedy_exact']}/48",
                runs[k]["total_budget_hit"],
                runs[k]["exclusive_stage_cap"],
            )
            for label, k in budgets
        ],
    )
    terminal = metrics["terminal"]
    _table(
        table_root / "terminal_constraints.tex",
        ["Final-stage measurement", "Value"],
        [
            ("Median margin violations", f"{terminal['median_margin_violations']:.0f}"),
            ("Range of margin violations", "--".join(map(str, terminal["range_margin_violations"]))),
            ("Mean fraction meeting margin", f"{terminal['mean_fraction_meeting_margin']:.3f}"),
            ("Previous stage converged", f"{terminal['previous_converged']}/48"),
            ("Minimum old-position violations", terminal["minimum_previous_position_violations"]),
            ("Median old-position lower bound", f"{terminal['median_previous_position_violations_lower_bound']:.0f}"),
        ],
    )
    routes = metrics["routes"]
    controls = [
        ("All memory reads removed", "all_layers_all"),
        ("Context reads removed", "all_layers_context"),
        ("Predict reads removed", "all_layers_predict"),
        ("Donor keys, all layers", "all_layers_key"),
        ("Donor values, all layers", "all_layers_value"),
        ("Donor keys + values", "all_layers_both"),
        ("Self patch, keys + values", "self_patch_both"),
    ]
    _table(
        table_root / "causal_routes.tex",
        ["Intervention", r"Standard $\Delta$NLL", r"SPS $\Delta$NLL"],
        [
            (
                label,
                f"{routes['standard']['controls'][key]:.3f}" if key in routes["standard"]["controls"] else "--",
                f"{routes['sps_w0']['controls'][key]:.3f}",
            )
            for label, key in controls
        ],
    )
    _table(
        table_root / "rescue.tex",
        ["Restart", "FP32 exact", "BF16 exact"],
        [
            (label, f"{metrics['rescue'][key]['fp32']}/4", f"{metrics['rescue'][key]['bf16']}/4")
            for label, key in [
                ("CE Adam (0.3)", "adam_ce_lr03"),
                ("CE Adam (1.0)", "adam_ce_lr1"),
                ("Hinge Adam", "adam_hinge_lr03"),
                ("Trust-region margin", "trust_margin"),
            ]
        ],
    )
    _table(
        table_root / "lr_sweep.tex",
        ["Learning rate", "Standard TF", "SPS TF"],
        [
            (
                f"{lr:g}",
                f"{runs['standard_' + suffix]['horizon']['mean']:.2f}",
                f"{runs['sps_' + suffix]['horizon']['mean']:.2f}",
            )
            for lr, suffix in [(0.01, "lr0.01"), (0.03, "lr0.03"), (0.3, "ce_raw"), (1.0, "lr1.0")]
        ],
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )

    def save(fig, name):
        fig.tight_layout(pad=0.6)
        fig.savefig(figure_root / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    labels = ["Standard\nmemory", "SPS\nmemory", "SPS\npredict"]
    for offset, keys, color, label in [
        (-0.17, ["standard_ce_raw", "sps_ce_raw", "predict_ce_native"], "#3775ad", "CE / argmax"),
        (0.17, ["standard_hinge_strict", "sps_hinge_strict", "predict_hinge_native"], "#db813b", "Hinge / margin"),
    ]:
        means = np.array([runs[k]["horizon"]["mean"] for k in keys])
        cis = np.array([runs[k]["horizon"]["ci95"] for k in keys]).T
        axes[0].bar(
            np.arange(3) + offset,
            means,
            0.34,
            label=label,
            color=color,
            yerr=np.maximum(np.stack([means - cis[0], cis[1] - means]), 0),
            capsize=2,
        )
    axes[0].set(xticks=np.arange(3), xticklabels=labels, ylabel="Mean TF horizon (tokens)", ylim=(0, 160))
    axes[0].legend(loc="upper left", frameon=False)
    for key, label, color in [("predict_hinge_native", "TF horizon", "#db813b")]:
        values = np.array([s["horizon"] for s in data["runs"][key]["samples"]])
        other = np.array([s["horizon"] for s in data["runs"]["standard_hinge_strict"]["samples"]])
        axes[1].scatter(other, values, s=13, alpha=0.65, color=color)
    axes[1].plot([0, 190], [0, 190], "--", color=".5", linewidth=0.8)
    axes[1].set(xlabel="Standard memory / hinge", ylabel="SPS shared predict / hinge", xlim=(0, 190), ylim=(0, 190))
    save(fig, "site_objective")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for i, (label, key) in enumerate(budgets):
        axes[0].bar(i, runs[key]["horizon"]["mean"], color=["#3775ad", "#db813b", "#32977f"][i])
        axes[0].text(i, runs[key]["horizon"]["mean"] + 3, f"{runs[key]['horizon']['mean']:.1f}", ha="center", fontsize=7)
    axes[0].set(xticks=range(3), xticklabels=["Original", "Stretch", "Fixed"], ylabel="TF horizon", ylim=(0, 165))
    axes[1].bar(range(3), [runs[k]["total_budget_hit"] for _, k in budgets], label="Total budget", color=".5")
    axes[1].bar(
        range(3),
        [runs[k]["exclusive_stage_cap"] for _, k in budgets],
        bottom=[runs[k]["total_budget_hit"] for _, k in budgets],
        label="Stage cap",
        color="#b47ba4",
    )
    axes[1].set(xticks=range(3), xticklabels=["Original", "Stretch", "Fixed"], ylabel="Texts stopping", ylim=(0, 58))
    axes[1].legend(frameon=False, fontsize=6)
    for key, label, color in [
        ("predict_hinge_native", "Native hinge", "#db813b"),
        ("predict_hinge_random", "Random hinge", "#32977f"),
    ]:
        curve = [p for p in runs[key]["position_cost"] if p["n"] >= 8]
        x = [p["position"] for p in curve]
        axes[2].plot(x, [p["median"] for p in curve], label=label, color=color, lw=1)
        axes[2].fill_between(x, [p["q25"] for p in curve], [p["q75"] for p in curve], alpha=0.12, color=color)
    axes[2].set(xlabel="Token position (survivors)", ylabel="Extra updates: median / IQR", yscale="symlog")
    axes[2].legend(frameon=False, fontsize=6)
    save(fig, "budget_control")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
    for route, label in [("all", "All memory reads"), ("context", "Context reads"), ("predict", "Predict reads")]:
        values = routes["sps_w0"]["layers"][route]
        axes[0].plot(range(1, len(values) + 1), values, label=label, marker="o", ms=2, lw=1)
    axes[0].set(xlabel="Layer", ylabel=r"SPS $\Delta$NLL, tokens 2--12")
    axes[0].legend(frameon=False)
    for model, label in [("standard", "Standard"), ("sps_w0", "SPS")]:
        axes[1].plot(range(1, 13), routes[model]["query_change"], marker="o", ms=2, lw=1, label=label)
    axes[1].set(xlabel="Layer", ylabel="Query change under context swap")
    axes[1].legend(frameon=False)
    save(fig, "routes")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    violations = [s["margin_violations"] for s in terminal["samples"]]
    axes[0].hist(violations, bins=np.arange(0, 101, 5), color="#81679c", edgecolor="white")
    axes[0].axvline(np.median(violations), color=".3", ls="--", lw=1)
    axes[0].set(xlabel="Positive-margin violations (final vector)", ylabel="Texts")
    axes[1].scatter([s["final_length"] for s in terminal["samples"]], violations, s=15, alpha=0.7, color="#81679c")
    axes[1].set(xlabel="Final failed target length", ylabel="Margin violations")
    save(fig, "terminal_constraints")
    json_write(output / "data/derived_metrics.json", metrics)
    return metrics


def replay_plan(data):
    """Return executable, isolated historical replays built from immutable registries.

    Run in this repository with external checkpoints available. Commands reconstruct
    the exact cached IDs, archive each historical source tree, and override only
    prerequisite/output locations; all design settings come from original code.
    """
    lines = [
        "#!/bin/sh",
        "set -eu",
        "# Historical experiment replay plan (GPU); not executed by render/export/plan.",
        "# CE/random arms are historical controls. Future launch policy is batch1, hinge/native.",
        "# Run from the repository root. No job submissions occur in these commands.",
        "# Checkpoint prerequisites: native Standard/SPS ckpt_tokens_1000341504_final.pt, GPT-2 tokenizer cache.",
        "# Supply STANDARD_CHECKPOINT and SPS_CHECKPOINT as absolute paths to the original checkpoints.",
        "# REPLAY_ROOT must be a fresh absolute directory. Install dependencies from archived pyproject.toml/uv.lock;",
        "# exporter versions in data/manifest.json describe measurement export, not historical runtime pinning.",
        ': "${STANDARD_CHECKPOINT:?set original native Standard checkpoint path}"',
        ': "${SPS_CHECKPOINT:?set original native SPS checkpoint path}"',
        ': "${REPLAY_ROOT:?set fresh absolute replay directory}"',
        "export STANDARD_CHECKPOINT SPS_CHECKPOINT REPLAY_ROOT",
        'test ! -e "$REPLAY_ROOT"',
        'mkdir -p "$REPLAY_ROOT/code" "$REPLAY_ROOT/outputs" "$REPLAY_ROOT/upstream"',
        "python - <<'PY'",
        "import gzip,json,os",
        "from pathlib import Path",
        "from datasets import Dataset",
        "p=Path(os.environ['REPLAY_ROOT'])/'exact_cached_cohort'",
        "rows=json.load(gzip.open('sps_paper/data/cohort.json.gz','rt'))",
        "Dataset.from_list([{'input_ids':r['input_ids'],'attention_mask':[1]*len(r['input_ids'])} for r in rows]).save_to_disk(str(p))",
        "PY",
        "# Exact original cache provenance: 2683216db54e0ae8f51a69fa467c2dc07f5509f9; no retokenization.",
    ]
    commits = sorted(
        {run["source_commit"] for run in data["runs"].values()} | {p["job"]["source_commit"] for p in data["probes"].values()}
    )
    for commit in commits:
        lines.extend(
            [f'mkdir -p "$REPLAY_ROOT/code/{commit}"', f'git archive {commit} | tar -x -C "$REPLAY_ROOT/code/{commit}"']
        )
    for key, run in data["runs"].items():
        commit = run["source_commit"]
        lines.extend(
            [
                f"# {key}; historical jobs: " + ", ".join(j["job_id"] for j in run["jobs"]),
                f"(cd \"$REPLAY_ROOT/code/{commit}\" && PYTHONPATH=.:src python - <<'PY'",
                "import os,subprocess,sys,inspect",
                "from pathlib import Path",
                f"out=Path(os.environ['REPLAY_ROOT'])/'outputs'/{key!r}",
                "cache=str(Path(os.environ['REPLAY_ROOT'])/'exact_cached_cohort')",
            ]
        )
        registry, selector = run["registry"].split(":", 1)
        if registry == "ordinary":
            lines.extend(
                [
                    "from scripts.jobs import run_jobs_sps as r",
                    f"exp=dict(next(e for e in r.{selector} if e['name']=={run['selector']!r}))",
                    f"exp['checkpoint']=os.environ[{'STANDARD_CHECKPOINT' if run['model'] == 'standard' else 'SPS_CHECKPOINT'!r}]",
                    "args=dict(r.TRAINING_ARGS,**exp.get('training_args',{}))",
                    f"args.update(model_checkpoint=os.environ[{'STANDARD_CHECKPOINT' if run['model'] == 'standard' else 'SPS_CHECKPOINT'!r}],tokenized_dataset_path=cache,output_dir=str(out))",
                    "out.mkdir(parents=True,exist_ok=False)",
                    f"(out/'design.json').write_text(__import__('json').dumps(dict(experiment=exp,training_args={{k:v for k,v in args.items() if k not in ('output_dir','model_checkpoint')}},commit={commit!r})))",
                    "cmd=[sys.executable,'scripts/activation_distillation.py']",
                    "for k,v in args.items(): cmd.extend(['--'+k,str(v)])",
                    "subprocess.run(cmd,check=True)",
                    "subprocess.run([sys.executable,'scripts/validation/validate_sps_batch1_run.py','--run-dir',str(out)],check=True)",
                    f"(Path(os.environ['REPLAY_ROOT'])/'upstream'/{run['source']!r}).symlink_to(out,target_is_directory=True)",
                ]
            )
        else:
            lines.extend(
                [
                    "from scripts import analyze_sps_predict_embedding as p",
                    "p.CHECKPOINT=os.environ['SPS_CHECKPOINT']",
                    "p.TRAINING_ARGS=dict(p.TRAINING_ARGS,tokenized_dataset_path=cache)",
                    "p.jobs.CACHE_ROOT=Path(cache).parent",
                    "out.mkdir(parents=True,exist_ok=False)",
                ]
            )
            if registry == "predict":
                lines.extend(
                    [
                        f"exp=next(e for e in p.experiment_configs if e['name']=={run['selector']!r})",
                        "p.ROOT=out",
                        "p.probe(exp)",
                    ]
                )
            else:
                lines.extend(
                    [
                        "from scripts import analyze_sps_predict_shards as s",
                        f"# Original CLI selection: --phase probe --suite {selector}",
                        f"exps=s.suite_settings({selector!r})[1] if hasattr(s,'suite_settings') else s.experiment_configs",
                        "p.ROOT=out",
                        "for exp in exps: p.probe(exp)",
                    ]
                )
        lines.extend(["PY", ")"])
    lines.extend(
        [
            "# Saved-stage diagnostic replay dependencies:",
            "# Fresh ordinary-memory runs above are linked under upstream/ using historical registry names.",
            "# Each diagnostic reads those newly trained vectors and the relocated checkpoint/cache paths.",
            "# Rescue precedes the gradient audit; its fresh result is linked under the original registry name.",
        ]
    )
    for name in ("ablation", "routes", "geometry", "rescue", "gradients"):
        probe = data["probes"][name]
        script = {
            "routes": "analyze_sps_memory_routes",
            "geometry": "analyze_sps_margin_geometry",
            "rescue": "analyze_sps_margin_rescue",
            "gradients": "audit_sps_objective_gradients",
            "ablation": "analyze_sps_memory_ablation",
        }[name]
        commit = probe["job"]["source_commit"]
        lines.extend(
            [
                f"# {name}: {probe['job']['job_id']}; original CLI --phase probe",
                f"(cd \"$REPLAY_ROOT/code/{commit}\" && PYTHONPATH=.:src python - <<'PY'",
                "import importlib,os",
                "from pathlib import Path",
                "from scripts import analyze_sps_cramming as jobs",
                "jobs.ROOT=Path(os.environ['REPLAY_ROOT'])/'upstream'",
                "# Fresh designs already contain the explicit absolute input paths; do not remap mounts.",
                "jobs.cluster_path=lambda path:str(path)",
                f"p=importlib.import_module('scripts.{script}')",
                f"out=Path(os.environ['REPLAY_ROOT'])/'outputs'/{name!r}",
                "out.mkdir(parents=True,exist_ok=False)",
                "p.OUTPUT=out",
                "jobs.OUTPUT=out",
                f"(jobs.ROOT/{PROBES[name]!r}).symlink_to(out,target_is_directory=True)",
                "p.probe()",
                "PY",
                ")",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="audit and snapshot existing completed artifact evidence")
    export.add_argument("--artifacts-root", type=Path, required=True)
    export.add_argument("--data-root", type=Path, default=HERE / "data")
    rendering = commands.add_parser("render", help="regenerate tables/figures from portable evidence")
    rendering.add_argument("--data-root", type=Path, default=HERE / "data")
    rendering.add_argument("--output-root", type=Path, default=HERE)
    plan = commands.add_parser("plan", help="print pinned historical GPU replay commands without executing")
    plan.add_argument("--data-root", type=Path, default=HERE / "data")
    args = parser.parse_args(argv)
    if args.command == "export":
        manifest = export_snapshot(args.artifacts_root, args.data_root)
        print(f"Exported audited evidence: {len(manifest['files'])} snapshots")
    elif args.command == "render":
        metrics = render(args.data_root, args.output_root)
        print(f"Rendered {len(metrics['runs'])} audited runs; fixed cohort of 48 texts")
    else:
        print(replay_plan(load_snapshot(args.data_root)), end="")


if __name__ == "__main__":
    main()
