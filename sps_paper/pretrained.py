"""Export audited pretrained measurements, render offline, or print replay commands.

The scientific design remains scripts/pretrained_sps_experiments.py. This module
never submits jobs; exported settings are measured provenance, not a new registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analyze_pretrained_sps import STAGE_FIELDS, audit_sample, canonical  # noqa: E402
from sps_paper.reproduce import JOB, SHA, digest, json_read, json_write, paired_interval  # noqa: E402

LABELS = {"qwen": "Qwen3-1.7B-Base", "smol": "SmolLM2-1.7B"}
ARMS = {"ar": "Original AR", "ce": "SPS-CE", "kd": "SPS-KD"}
HASH = re.compile(r"^[0-9a-f]{64}$")
NAMES = tuple(f"{family}_{arm}" for family in LABELS for arm in ARMS)


def export(destination=HERE / "data"):
    from datasets import load_from_disk
    from scripts.analyze_pretrained_sps import audit_experiment
    from scripts.pretrained_sps_experiments import ROOT, TRAINING_ARGS, UPSTREAM_COMMIT, experiment_configs
    from scripts.pretrained_sps_worker import validate_data

    families, runs, sources = {}, {}, {}
    # Full raw-stage auditing is mandatory before producing publication evidence.
    with tempfile.TemporaryDirectory() as temporary:
        for exp in experiment_configs:
            summary = audit_experiment(exp, output_dir=Path(temporary) / exp["name"])
            family = exp["family"]
            if family not in families:
                dataset, manifest = validate_data(exp)
                config = json_read(Path(exp["source_path"]) / "config.json")
                families[family] = dict(
                    manifest=manifest,
                    bos_token_id=config["bos_token_id"],
                    hidden_size=config["hidden_size"],
                    targets=[dict(input_ids=row["input_ids"], attention_mask=row["attention_mask"]) for row in dataset],
                )
            shards = []
            for source in summary["source_shards"]:
                path = Path(source["path"])
                dirname = "progressive_prefixes" if exp["arm"] == "ar" else "predict_prefixes"
                columns = STAGE_FIELDS + (
                    () if exp["arm"] == "ar" else ("embedding_role", "fixed_bos_token_id", "predict_init")
                )
                shards.append(
                    dict(
                        job=json_read(path / "job.json"),
                        report=json_read(path / "single_sample_validation.json"),
                        stages=list(load_from_disk(str(path / dirname)).select_columns(columns)),
                    )
                )
                for file in [
                    path / "job.json",
                    path / "completed",
                    path / "single_sample_validation.json",
                    *sorted((path / dirname).iterdir()),
                ]:
                    if file.is_file():
                        sources[str(file.relative_to(ROOT))] = digest(file)
            runs[exp["name"]] = dict(summary=summary, shards=shards)
            print("Audited/exported " + exp["name"], flush=True)
    bundle = dict(
        schema=1,
        registry="scripts/pretrained_sps_experiments.py",
        registry_sha256=digest(REPO / "scripts/pretrained_sps_experiments.py"),
        upstream_commit=UPSTREAM_COMMIT,
        training_args=TRAINING_ARGS,
        data_design=json_read(ROOT / "data/design.json"),
        raw_documents=json_read(ROOT / "data/raw.json"),
        families=families,
        runs=runs,
    )
    derive(bundle)  # Check compact evidence with the same offline path used by render.
    destination = Path(destination)
    json_write(destination / "pretrained.json.gz", bundle)
    json_write(
        destination / "pretrained_manifest.json",
        dict(schema=1, files={"pretrained.json.gz": digest(destination / "pretrained.json.gz")}, source_files=sources),
    )


def derive(bundle):
    """Recompute scores, stopping reasons and paired intervals from raw evidence."""
    if bundle["schema"] != 1 or set(bundle["runs"]) != set(NAMES) or set(bundle["families"]) != set(LABELS):
        raise ValueError("Incomplete pretrained evidence")
    if not HASH.fullmatch(bundle["registry_sha256"]) or not SHA.fullmatch(bundle["upstream_commit"]):
        raise ValueError("Invalid source provenance")
    results, all_jobs, fingerprints = {}, set(), set()
    raw = bundle["raw_documents"]
    documents = [r["sha256"] for r in raw]
    if [r["sample_id"] for r in raw] != list(range(len(raw))) or any(
        hashlib.sha256(r["text"].encode()).hexdigest() != r["sha256"] for r in raw
    ):
        raise ValueError("Raw document identity mismatch")
    for family in LABELS:
        data = bundle["families"][family]
        manifest = data["manifest"]
        ids = [t["input_ids"] for t in data["targets"]]
        masks = [t["attention_mask"] for t in data["targets"]]
        payload = dict(input_ids=ids, attention_mask=masks)
        if (
            hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            != manifest["targets_sha256"]
        ):
            raise ValueError("Native target IDs do not match pinned cache hash")
        if documents != manifest["document_hashes"] or len(ids) != len(documents) or len(set(documents)) != len(documents):
            raise ValueError("Raw document pairing mismatch")
        for target, mask in zip(ids, masks):
            if (
                not target
                or len(target) != len(mask)
                or any(type(t) is not int or t < 0 for t in target)
                or mask != [1] * sum(mask) + [0] * (len(mask) - sum(mask))
            ):
                raise ValueError("Invalid native target IDs or attention mask")
    for name in NAMES:
        run = bundle["runs"][name]
        summary = run["summary"]
        exp = summary["experiment"]
        family, arm = name.split("_")
        data = bundle["families"][family]
        if exp["name"] != name or exp["family"] != family or exp["arm"] != arm:
            raise ValueError("Wrong experiment identity")
        if exp["source"] != data["manifest"]["source"] or exp["revision"] != data["manifest"]["revision"]:
            raise ValueError("Wrong source model revision")
        fingerprint = summary["code_digest"]
        fingerprints.add(fingerprint)
        if not HASH.fullmatch(fingerprint):
            raise ValueError("Invalid worker fingerprint")
        records, jobs, commits = [], [], []
        for shard in run["shards"]:
            job, report = shard["job"], shard["report"]
            job_id, commit = job["result"]["job_name"], job["commit"]
            if not JOB.fullmatch(job_id) or job_id in all_jobs or not SHA.fullmatch(commit):
                raise ValueError("Invalid or reused job UUID/source commit")
            all_jobs.add(job_id)
            jobs.append(job_id)
            commits.append(commit)
            expected = dict(
                bundle["training_args"],
                model_checkpoint=exp["checkpoint"] or exp["source"],
                offset_dataset_items=min(s["sample_id"] for s in report["samples"]),
                limit_dataset_items=len(report["samples"]),
            )
            settings = {k: v for k, v in report["settings"].items() if k not in {"output_dir", "logging_dir"}}
            role = "ordinary_memory" if arm == "ar" else "shared_predict"
            if (
                canonical(settings) != canonical(expected)
                or canonical(report["experiment"]) != canonical(exp)
                or report["source_commit"] != commit
                or report["code_digest"] != fingerprint
                or report["data_manifest"] != data["manifest"]
                or report["embedding_role"] != role
                or report["bos_token_id"] != data["bos_token_id"]
            ):
                raise ValueError("Shard settings or provenance mismatch")
            grouped = defaultdict(list)
            for stage in shard["stages"]:
                if stage["hidden_size"] != data["hidden_size"] or stage["num_compression_tokens"] != 1:
                    raise ValueError("Wrong vector dimensions")
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
                        raise ValueError("Raw-stage settings mismatch")
                if arm != "ar" and (
                    stage["embedding_role"] != role
                    or stage["fixed_bos_token_id"] != data["bos_token_id"]
                    or stage["predict_init"] != "native"
                ):
                    raise ValueError("Wrong prediction conditioning interface")
                grouped[stage["sample_id"]].append(stage)
            if set(grouped) != {s["sample_id"] for s in report["samples"]}:
                raise ValueError("Raw-stage and selected sample IDs disagree")
            for sample in report["samples"]:
                sid = sample["sample_id"]
                target = data["targets"][sid]
                row = audit_sample(
                    grouped[sid], sample, target["input_ids"], expected, valid_length=sum(target["attention_mask"])
                )
                row["document_sha256"] = data["manifest"]["document_hashes"][sid]
                records.append(row)
        records.sort(key=lambda r: r["sample_id"])
        if records != summary["samples"] or [r["sample_id"] for r in records] != list(range(len(data["targets"]))):
            raise ValueError("Offline raw-stage audit disagrees with full-cohort audit")
        if jobs != [s["job_id"] for s in summary["source_shards"]] or commits != [
            s["source_commit"] for s in summary["source_shards"]
        ]:
            raise ValueError("Summary job provenance mismatch")
        results[name] = dict(
            n=len(records),
            horizon=paired_interval([r["horizon"] for r in records]),
            greedy_initial_prefix=paired_interval([r["greedy_initial_prefix"] for r in records]),
            teacher_exact=sum(r["teacher_exact"] for r in records),
            greedy_exact=sum(r["greedy_exact"] for r in records),
            mean_updates=float(np.mean([r["total_steps"] for r in records])),
            stopping_counts=dict(Counter(r["stop_reason"] for r in records)),
            samples=records,
        )
    if len(fingerprints) != 1:
        raise ValueError("Scientific worker fingerprints differ")
    paired = {}
    for family in LABELS:
        for arm in ("ce", "kd"):
            a, b = results[f"{family}_{arm}"]["samples"], results[f"{family}_ar"]["samples"]
            paired[f"{family}_{arm}_minus_ar"] = {
                metric: paired_interval([x[metric] - y[metric] for x, y in zip(a, b)])
                for metric in ("horizon", "greedy_initial_prefix")
            }
    return dict(runs=results, paired=paired, code_digest=next(iter(fingerprints)))


def load(data_root):
    root = Path(data_root)
    manifest = json_read(root / "pretrained_manifest.json")
    if manifest["schema"] != 1 or set(manifest["files"]) != {"pretrained.json.gz"}:
        raise ValueError("Invalid pretrained manifest")
    if digest(root / "pretrained.json.gz") != manifest["files"]["pretrained.json.gz"]:
        raise ValueError("Pretrained evidence checksum mismatch")
    return json_read(root / "pretrained.json.gz")


def render(data_root=HERE / "data", output_root=HERE):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sps_paper.reproduce import _table

    metrics = derive(load(data_root))
    output = Path(output_root)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    rows = []
    for name, run in metrics["runs"].items():
        family, arm = name.split("_")
        rows.append(
            (
                LABELS[family] + " / " + ARMS[arm],
                f"{run['horizon']['mean']:.2f}",
                f"{run['greedy_initial_prefix']['mean']:.2f}",
                f"{run['greedy_exact']}/{run['n']}",
                f"{run['mean_updates']:.0f}",
            )
        )
    _table(
        output / "tables/pretrained_results.tex",
        ["Model / interface", "TF horizon", "Greedy prefix", "Greedy exact", "Updates"],
        rows,
    )
    paired_rows = []
    for key, pair in metrics["paired"].items():
        family, arm, _, _ = key.split("_")
        values = []
        for metric in ("horizon", "greedy_initial_prefix"):
            value = pair[metric]
            lo, hi = value["ci95"]
            values.append(f"{value['mean']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        paired_rows.append((LABELS[family] + " / " + ARMS[arm] + " minus AR", *values))
    _table(
        output / "tables/pretrained_paired.tex",
        ["Paired comparison", r"$\Delta H$ [95\% CI]", r"$\Delta G$ [95\% CI]"],
        paired_rows,
    )
    for metric, filename, ylabel in (
        ("horizon", "pretrained_horizons", "Teacher-forced horizon (tokens)"),
        ("greedy_initial_prefix", "pretrained_greedy", "Initial correct greedy prefix (tokens)"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True, constrained_layout=True)
        for ax, family in zip(axes, LABELS):
            values = [metrics["runs"][f"{family}_{arm}"][metric] for arm in ARMS]
            means = np.array([r["mean"] for r in values])
            ci = np.array([r["ci95"] for r in values])
            ax.bar(range(3), means, color=["#486A95", "#C78E43", "#54846D"], width=0.6)
            ax.errorbar(
                range(3), means, yerr=np.stack([means - ci[:, 0], ci[:, 1] - means]), fmt="none", color="black", capsize=4
            )
            ax.set_xticks(range(3), list(ARMS.values()))
            ax.set_title(LABELS[family])
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(ylabel)
        fig.savefig(output / f"figures/{filename}.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(output / f"figures/{filename}.png", dpi=180)
        plt.close(fig)
    json_write(output / "data/pretrained_metrics.json", metrics)
    return metrics


def restore_data(bundle, output_root):
    """Materialize exact native caches in a fresh tree without a tokenizer/network."""
    from datasets import Dataset

    derive(bundle)
    dest = Path(output_root) / "data"
    dest.mkdir(parents=True, exist_ok=False)
    json_write(dest / "raw.json", bundle["raw_documents"])
    json_write(dest / "design.json", bundle["data_design"])
    for family, data in bundle["families"].items():
        Dataset.from_list(data["targets"]).save_to_disk(str(dest / family))
        json_write(dest / (family + ".json"), data["manifest"])
    (dest / "completed").touch()


def replay_one(bundle, archive, output_root, source_root, name, offset, *, preflight=False):
    """Run original GPU code directly, with only filesystem paths relocated."""
    from scripts.pretrained_sps_experiments import code_digest

    metrics = derive(bundle)
    archive, root, source_root = Path(archive).resolve(), Path(output_root).resolve(), Path(source_root).resolve()
    shard = next(s for s in bundle["runs"][name]["shards"] if min(r["sample_id"] for r in s["report"]["samples"]) == offset)
    if code_digest(archive) != metrics["code_digest"]:
        raise ValueError("Replay archive differs from original worker fingerprint")
    if not (root / "data/completed").is_file():
        raise ValueError("Restore bundled native caches before replay")
    output = root / ("preflight" if preflight else "runs") / name / str(offset)
    if output.exists():
        raise ValueError("Replay output must be fresh")
    config = dict(
        root=str(root),
        source_root=str(source_root),
        name=name,
        offset=offset,
        count=len(shard["report"]["samples"]),
        output=str(output),
        preflight=preflight,
    )
    driver = """
import json,sys
from pathlib import Path
from scripts import pretrained_sps_experiments as registry
from scripts import pretrained_sps_worker as worker
c=json.loads(sys.argv[1])
registry.ROOT=worker.ROOT=Path(c['root'])
def relocate(value):
    if value is None: return None
    return str(Path(c['source_root'])/Path(value).relative_to(registry.SOURCE_ROOT))
for exp in registry.experiment_configs:
    exp['source_path']=relocate(exp['source_path'])
    exp['checkpoint']=relocate(exp['checkpoint'])
exp=next(e for e in registry.experiment_configs if e['name']==c['name'])
output=Path(c['output'])
output.mkdir(parents=True,exist_ok=False)
if c['preflight']: worker.preflight(exp['family'],output)
else: worker.train_shard(exp,c['offset'],c['count'],output)
"""
    env = dict(
        os.environ, PYTHONPATH=str(archive / "src") + os.pathsep + str(archive), RUN_SOURCE_COMMIT=shard["job"]["commit"]
    )
    subprocess.run([sys.executable, "-c", driver, json.dumps(config)], cwd=archive, env=env, check=True)


def plan(data_root=HERE / "data"):
    """An executable serial GPU replay; print only, without job submission."""
    bundle = load(data_root)
    derive(bundle)
    commits = sorted({s["job"]["commit"] for r in bundle["runs"].values() for s in r["shards"]})
    script = shlex.quote(str(Path(__file__).resolve()))
    data_arg = shlex.quote(str(Path(data_root).resolve()))
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Printed plan, not executed. Requires GPU, archived dependencies and external weights.",
        ': "${REPLAY_ROOT:?Set a fresh absolute replay output path}"',
        ': "${PRETRAINED_SOURCE_ROOT:?Set the external sps-artifacts input root}"',
        'test "${REPLAY_ROOT#/}" != "$REPLAY_ROOT"',
        'test ! -e "$REPLAY_ROOT"',
        'mkdir -p "$REPLAY_ROOT/code"',
        "# Source registry: " + bundle["registry"],
        "# Upstream SPS implementation: " + bundle["upstream_commit"],
        "# Immutable worker snapshots: " + ", ".join(commits),
        f'python {script} restore-data --data-root {data_arg} --output-root "$REPLAY_ROOT"',
    ]
    for commit in commits:
        lines.extend(
            [
                f'mkdir "$REPLAY_ROOT/code/{commit}"',
                f'git -C {shlex.quote(str(REPO))} archive {commit} | tar -x -C "$REPLAY_ROOT/code/{commit}"',
            ]
        )
    for family in LABELS:
        commit = bundle["runs"][family + "_ar"]["shards"][0]["job"]["commit"]
        lines.append(
            f'python {script} replay --data-root {data_arg} --archive "$REPLAY_ROOT/code/{commit}" --output-root "$REPLAY_ROOT" --source-root "$PRETRAINED_SOURCE_ROOT" --experiment {family}_ar --offset 0 --preflight'
        )
    for name in NAMES:
        run = bundle["runs"][name]
        lines.append("# " + name + ":")
        for shard in run["shards"]:
            ids = [s["sample_id"] for s in shard["report"]["samples"]]
            lines.append(
                f"#   commit={shard['job']['commit']} offset={min(ids)} original_job={shard['job']['result']['job_name']}"
            )
            commit = shard["job"]["commit"]
            lines.append(
                f'python {script} replay --data-root {data_arg} --archive "$REPLAY_ROOT/code/{commit}" --output-root "$REPLAY_ROOT" --source-root "$PRETRAINED_SOURCE_ROOT" --experiment {name} --offset {min(ids)}'
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("export", "render", "verify", "plan", "restore-data", "replay"))
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--experiment", choices=NAMES)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.phase == "export":
        export(args.data_root)
    elif args.phase == "render":
        render(args.data_root, args.output_root)
    elif args.phase == "verify":
        metrics = derive(load(args.data_root))
        print(f"Verified {len(metrics['runs'])} complete pretrained cohorts")
    elif args.phase == "plan":
        print(plan(args.data_root), end="")
    elif args.phase == "restore-data":
        restore_data(load(args.data_root), args.output_root)
    else:
        if not all((args.archive, args.source_root, args.experiment)):
            parser.error("replay requires --archive, --source-root and --experiment")
        replay_one(
            load(args.data_root),
            args.archive,
            args.output_root,
            args.source_root,
            args.experiment,
            args.offset,
            preflight=args.preflight,
        )


if __name__ == "__main__":
    main()
