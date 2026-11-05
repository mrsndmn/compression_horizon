"""Submit the two remaining SPS cohorts alongside the active AR cohort.

Scientific settings and workers come from pretrained_sps_experiments. This
operational entrypoint does not change their fingerprint; its own source SHA
and Git commit are recorded separately. New outputs and code archives live on
physical NFS2. Existing NFS1 outputs are never moved by this launcher.
"""

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

from scripts import run_pretrained_sps_cramming as base
from scripts.pretrained_sps_experiments import (
    COHORT,
    FAMILIES,
    SHARD_SIZE,
    code_digest,
    experiment_configs,
    read_json,
    write_json,
)

# User-authorized operational schedule; the training registry stays unchanged.
NAMES = ("smol_ce", "smol_kd")
MAX_SUITE_ACTIVE = 24
MIN_FREE_BYTES = 4 * 2**30
NFS2_ROOT = base.ROOT.parent / (base.ROOT.name + "_nfs2")


def physical_path(name, fingerprint):
    return NFS2_ROOT / "runs" / fingerprint[:12] / name


def require_ready():
    if not set(NAMES) <= {e["name"] for e in experiment_configs}:
        raise RuntimeError("Parallel model selection must come from experiment registry")
    if subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=base.REPO):
        raise RuntimeError("Commit tracked changes before launching immutable code")
    untracked = subprocess.check_output(["git", "ls-files", "--others", "-z", "--", "src"], cwd=base.REPO)
    if any(p.endswith(b".py") for p in untracked.split(b"\0")):
        raise RuntimeError("Untracked source Python would not be archived")
    fingerprint = code_digest(base.REPO)
    if not (base.ROOT / "data/completed").is_file():
        raise RuntimeError("Prepared data required")
    for family in FAMILIES:
        gate = base.output_path("preflight", family)
        if not (gate / "completed").is_file() or read_json(gate / "preflight.json")["code_digest"] != fingerprint:
            raise RuntimeError("Successful matching-code preflights required")
    if NFS2_ROOT.is_symlink() or not NFS2_ROOT.resolve().is_relative_to(base.ROOT.parent.resolve()):
        raise RuntimeError("Parallel output root must remain on original NFS2")
    if shutil.disk_usage(base.ROOT.parent).free < MIN_FREE_BYTES:
        raise RuntimeError("Insufficient NFS2 space; wait for cleanup without relocation")
    for name in NAMES:
        canonical = base.output_path("runs", name).parent
        target = physical_path(name, fingerprint)
        if canonical.is_symlink():
            if canonical.resolve() != target.resolve():
                raise RuntimeError("Existing output link points elsewhere")
        elif canonical.exists():
            raise RuntimeError("Existing output directory must not be relocated")
        elif target.exists() and any(target.iterdir()):
            raise RuntimeError("Unlinked physical output already contains artifacts; reconcile first")
        if target.is_symlink() or not target.resolve().is_relative_to(NFS2_ROOT.resolve()):
            raise RuntimeError("Physical cohort output points outside NFS2 root")
        for offset in range(0, COHORT, SHARD_SIZE):
            shard = target / f"shard{offset // SHARD_SIZE:02d}"
            if shard.is_symlink() or not shard.resolve().is_relative_to(target.resolve()):
                raise RuntimeError("Shard output points outside its physical cohort")
    return fingerprint


@contextmanager
def submission_lock():
    NFS2_ROOT.mkdir(parents=True, exist_ok=True)
    with (NFS2_ROOT / "submission.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another parallel submission is in progress") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_routes(fingerprint):
    for name in NAMES:
        target = physical_path(name, fingerprint)
        target.mkdir(parents=True, exist_ok=True)
        canonical = base.output_path("runs", name).parent
        canonical.parent.mkdir(parents=True, exist_ok=True)
        if not canonical.is_symlink():
            canonical.symlink_to(target, target_is_directory=True)
    _, commit, canonical_stage = base.payload("runs", NAMES[0])
    stage = NFS2_ROOT / "code" / commit
    if stage.is_symlink() or not stage.resolve().is_relative_to(NFS2_ROOT.resolve()):
        raise RuntimeError("Physical source archive points outside NFS2 root")
    base.stage_code(commit, stage)
    if code_digest(stage) != fingerprint:
        raise RuntimeError("Archived worker code differs from preflight")
    relative = Path(__file__).relative_to(base.REPO)
    if (stage / relative).read_bytes() != Path(__file__).read_bytes():
        raise RuntimeError("Parallel launcher must be committed in its source archive")
    canonical_stage.parent.mkdir(parents=True, exist_ok=True)
    if canonical_stage.exists():
        if canonical_stage.resolve() != stage.resolve():
            raise RuntimeError("Existing source archive points elsewhere")
    else:
        canonical_stage.symlink_to(stage, target_is_directory=True)
    return commit


def submit_with_client(client, options, active, fingerprint):
    specs = [(name, offset) for name in NAMES for offset in range(0, COHORT, SHARD_SIZE)]
    required = 0
    for name, offset in specs:
        output = base.output_path("runs", name, offset)
        if (output / "completed").is_file():
            continue
        description = base.payload("runs", name, offset)[0]["job_desc"].split(" #", 1)[0]
        if (output / "job.json").is_file():
            job = read_json(output / "job.json")["result"]["job_name"]
            if not base.JOB_ID.fullmatch(job) or description not in active:
                raise RuntimeError("Recorded submission is not active; reconcile terminal status first")
        elif (output / "submission-intent.json").exists():
            raise RuntimeError("Submission outcome uncertain; reconcile before retry")
        elif description in active:
            raise RuntimeError("Active job lacks local record; reconcile before retry")
        else:
            required += 1
    if len(active) + required > MAX_SUITE_ACTIVE:
        raise RuntimeError("Parallel suite concurrency limit reached")
    commit = prepare_routes(fingerprint)
    per_model = {name: {d for d in active if d.startswith(f"CH: pretrained_sps_runs_{name}_")} for name in NAMES}
    record = dict(
        source_commit=commit,
        submission_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        code_digest=fingerprint,
        experiments={name: [] for name in NAMES},
        physical_root=str(NFS2_ROOT),
    )
    for name, offset in specs:
        # Retain the base launcher's eight-job guard within each model. The
        # global 24-job limit above accounts for the simultaneously active AR.
        job = base.submit_one("runs", name, offset, client, options, per_model[name])
        if job:
            record["experiments"][name].append(job)
        write_json(NFS2_ROOT / "latest-parallel.json", record)
    return record


def submit():
    fingerprint = require_ready()
    with submission_lock():
        require_ready()
        from mls.manager.job.utils import (
            get_in_progress_jobs,
            training_job_api_from_profile,
        )

        client, options = training_job_api_from_profile("default")
        descriptions = [
            j["job_desc"].split(" #", 1)[0]
            for j in get_in_progress_jobs()
            if j.get("job_desc", "").startswith("CH: pretrained_sps_")
        ]
        if len(descriptions) != len(set(descriptions)):
            raise RuntimeError("Duplicate active descriptions; reconcile before submission")
        active = set(descriptions)
        return submit_with_client(client, options, active, fingerprint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if args.dry:
        fingerprint = code_digest(base.REPO)
        assert set(NAMES) <= {e["name"] for e in experiment_configs}
        for name in NAMES:
            for offset in range(0, COHORT, SHARD_SIZE):
                print(
                    json.dumps(
                        dict(
                            experiment=name,
                            offset=offset,
                            physical_output=str(physical_path(name, fingerprint) / f"shard{offset // SHARD_SIZE:02d}"),
                            payload=base.payload("runs", name, offset)[0],
                        )
                    )
                )
        return
    print(json.dumps(submit()), flush=True)


if __name__ == "__main__":
    main()
