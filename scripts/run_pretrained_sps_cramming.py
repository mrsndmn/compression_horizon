"""Dry/prepare/preflight/advance/probe routes for the pretrained SPS experiment.

Run advance after a durable wait resolves. It submits only the next incomplete
model's shards and never holds a local dispatcher alive between waves.
"""

import argparse
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from scripts.analyze_sps_cramming import cluster_path
from scripts.pretrained_sps_experiments import (
    COHORT,
    FAMILIES,
    JOB_CONFIG,
    MAX_ACTIVE,
    PYTHON,
    ROOT,
    SHARD_SIZE,
    code_digest,
    experiment_configs,
    read_json,
    write_json,
)

REPO = Path(__file__).resolve().parents[1]
JOB_ID = re.compile(r"^lm-mpi-job-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def output_path(kind, name, offset=0):
    base = ROOT / kind / code_digest(REPO)[:12] / name
    return base if kind == "preflight" else base / f"shard{offset // SHARD_SIZE:02d}"


def payload(kind, name, offset=0):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    stage = ROOT / "code" / commit
    args = [
        PYTHON,
        "-m",
        "scripts.run_pretrained_sps_cramming",
        "--phase",
        "probe",
        "--kind",
        kind,
        "--name",
        name,
        "--offset",
        str(offset),
    ]
    return (
        dict(
            JOB_CONFIG,
            script=f"cd {shlex.quote(cluster_path(stage))} && exec {shlex.join(args)}",
            job_desc=f"CH: pretrained_sps_{kind}_{name}_{offset} {code_digest(REPO)[:12]} #d.tarasov",
            env_variables=dict(
                PYTHONPATH=f"{cluster_path(stage)}:{cluster_path(stage / 'src')}",
                OMP_NUM_THREADS="8",
                PYTHONUNBUFFERED="1",
                RUN_SOURCE_COMMIT=commit,
            ),
        ),
        commit,
        stage,
    )


def stage_code(commit, stage):
    if stage.exists():
        return
    stage.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=stage.parent, prefix="stage-") as directory:
        archive = subprocess.Popen(["git", "archive", commit], cwd=REPO, stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", directory], stdin=archive.stdout, check=True)
        archive.stdout.close()
        if archive.wait():
            raise RuntimeError("git archive failed")
        Path(directory).rename(stage)
        # TemporaryDirectory tolerates its renamed directory being absent.


def submit_one(kind, name, offset, client, options, active):
    output = output_path(kind, name, offset)
    if (output / "completed").exists():
        return None
    if (output / "job.json").exists():
        job = read_json(output / "job.json")["result"]["job_name"]
        if not JOB_ID.fullmatch(job):
            raise ValueError("Invalid recorded full MLS UUID")
        return job
    if (output / "submission-intent.json").exists():
        raise RuntimeError(f"Submission outcome uncertain; reconcile {output} before retry")
    plan, commit, stage = payload(kind, name, offset)
    description = plan["job_desc"].split(" #", 1)[0]
    if description in active:
        raise RuntimeError("Matching active job has no local submission record; reconcile first")
    if len(active) >= MAX_ACTIVE:
        raise RuntimeError("Pretrained SPS suite concurrency limit reached")
    stage_code(commit, stage)
    plan["region"] = options["region"]
    write_json(output / "submission-intent.json", dict(commit=commit, payload=plan))
    response = client.run_job(payload=plan)
    job = response["job_name"]
    if not JOB_ID.fullmatch(job):
        raise ValueError("Provider did not return a complete MLS job UUID")
    write_json(output / "job.json", dict(commit=commit, result=response))
    active.add(description)
    return job


def submit(phase):
    if subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=REPO):
        raise RuntimeError("Commit tracked changes before launching immutable code")
    if not (ROOT / "data/completed").exists():
        raise RuntimeError("Prepare the pinned model-native data first")
    if phase == "preflight":
        specs = [("preflight", family, 0) for family in FAMILIES]
    else:
        for family in FAMILIES:
            gate = output_path("preflight", family)
            if not (gate / "completed").exists() or read_json(gate / "preflight.json")["code_digest"] != code_digest(REPO):
                raise RuntimeError("Successful matching-code preflights required before main training")
        specs = []
        for exp in experiment_configs:
            pending = [
                ("runs", exp["name"], i)
                for i in range(0, COHORT, SHARD_SIZE)
                if not (output_path("runs", exp["name"], i) / "completed").exists()
            ]
            if pending:
                specs = pending
                break
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

    client, options = training_job_api_from_profile("default")
    active = {
        j["job_desc"].split(" #", 1)[0]
        for j in get_in_progress_jobs()
        if j.get("job_desc", "").startswith("CH: pretrained_sps_")
    }
    jobs = []
    for kind, name, offset in specs:
        job = submit_one(kind, name, offset, client, options, active)
        if job:
            jobs.append(job)
    record = dict(phase=phase, code_digest=code_digest(REPO), jobs=jobs)
    write_json(ROOT / f"latest-{phase}.json", record)
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dry", "prepare", "preflight", "advance", "probe"), required=True)
    parser.add_argument("--kind", choices=("preflight", "runs"), default="preflight")
    parser.add_argument("--name", choices=tuple(FAMILIES) + tuple(e["name"] for e in experiment_configs))
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.phase == "dry":
        names = list(FAMILIES) if args.kind == "preflight" else [e["name"] for e in experiment_configs]
        for name in names:
            for offset in [0] if args.kind == "preflight" else range(0, COHORT, SHARD_SIZE):
                print(json.dumps(payload(args.kind, name, offset)[0]))
        return
    if args.phase in {"preflight", "advance"}:
        submit(args.phase)
        return
    from scripts.pretrained_sps_worker import preflight, prepare_data, train_shard

    if args.phase == "prepare":
        prepare_data()
        return
    if args.name is None:
        parser.error("probe requires --name")
    if args.kind == "preflight":
        if args.name not in FAMILIES:
            parser.error("preflight requires a family")
        preflight(args.name, output_path("preflight", args.name))
    else:
        if args.offset not in range(0, COHORT, SHARD_SIZE):
            parser.error("offset must be a registered shard boundary")
        exp = next((e for e in experiment_configs if e["name"] == args.name), None)
        if exp is None:
            parser.error("runs requires an experiment")
        train_shard(exp, args.offset, SHARD_SIZE, output_path("runs", args.name, args.offset))


if __name__ == "__main__":
    main()
