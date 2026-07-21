"""Guarded launcher for the CE-temperature sweep.

Submits only the ``CE_TEMPERATURE_EXPERIMENTS`` from ``run_jobs_progressive.py``
(the baseline-CE configs in ``CE_TEMPERATURE_MODELS`` x the ``CE_TEMPERATURES`` grid
x {raw, t2} gradient arms, with T=1.0 deduped to a single control per model). The
grid and model set live in that module and may grow, so this driver derives the run
list from it rather than hard-coding one. Reuses that module's ``render_job`` so the
commands/out_dirs are byte-identical to the generic launcher -- this is just a *safe*
driver for the temperature subset.

Why a bespoke driver instead of ``run_jobs_progressive.py --model pythia``:
  * ``--model pythia`` would also (re)consider every other pythia experiment;
  * MLSpace's ``run_job`` can internally retry on a slow gateway and create a
    DUPLICATE job that shares one out_dir (corruption), and killing jobs needs a
    human grant. So each submission runs in a ``timeout``-guarded child process
    (this same file, ``--submit-index N``); the parent then verifies via a fresh
    in-progress fetch and never resubmits a desc that already landed.

Usage:
    python scripts/jobs/launch_ce_temperature.py            # submit all pending
    python scripts/jobs/launch_ce_temperature.py --dry      # print, do not submit
    python scripts/jobs/launch_ce_temperature.py --verify   # list temp jobs + dupes
    python scripts/jobs/launch_ce_temperature.py --submit-index N   # internal child
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jobs_progressive as rjp  # noqa: E402
from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile  # noqa: E402

PYTHON_PATH = "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python"
AUTHOR_NAME = "d.tarasov"
SUBMIT_TIMEOUT_SEC = 50  # kill a slow/retrying run_job child before it can duplicate.


def job_desc_for(exp_suffix: str) -> str:
    return f"CH: progressive {exp_suffix} #{AUTHOR_NAME} #multimodal #notify_completed @mrsndmn"


def build_payload(experiment, region: str):
    workdir = os.getcwd()
    cmd_args, exp_suffix, out_dir_name = rjp.render_job(experiment)
    script = f" cd {workdir} && {PYTHON_PATH} scripts/activation_distillation.py  {' '.join(cmd_args)}"
    payload = {
        "script": script,
        "job_desc": job_desc_for(exp_suffix),
        "env_variables": {
            "PYTHONPATH": "./src",
            "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface",
        },
        "instance_type": "a100.1gpu",
        "region": region,
        "type": "binary_exp",
        "shm_size_class": "medium",
        "base_image": "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41",
        "n_workers": 1,
        "processes_per_worker": 1,
    }
    return payload, exp_suffix, out_dir_name


def in_progress_descs() -> list:
    return [j.get("job_desc", "") for j in get_in_progress_jobs()]


def submit_one_child(index: int) -> int:
    """Child mode: submit exactly one job, then exit. Run under an outer timeout."""
    client, extra_options = training_job_api_from_profile("default")
    experiment = rjp.CE_TEMPERATURE_EXPERIMENTS[index]
    payload, exp_suffix, out_dir_name = build_payload(experiment, extra_options["region"])
    result = client.run_job(payload=payload)
    print(f"SUBMITTED index={index} out_dir={out_dir_name} result={result}", flush=True)
    return 0


def verify():
    descs = in_progress_descs()
    print(f"In-progress jobs total: {len(descs)}")
    seen = {}
    for experiment in rjp.CE_TEMPERATURE_EXPERIMENTS:
        _, exp_suffix, out_dir_name = build_payload(experiment, "-")
        desc = job_desc_for(exp_suffix)
        count = sum(1 for d in descs if d == desc)
        exists = os.path.isdir(out_dir_name)
        seen[exp_suffix] = count
        tag = "DUPLICATE!" if count > 1 else ("in-queue" if count == 1 else ("artifact" if exists else "MISSING"))
        print(f"  [{tag:>10}] queue={count} artifact={int(exists)}  {exp_suffix}")
    dupes = [k for k, v in seen.items() if v > 1]
    if dupes:
        print(f"\n!! {len(dupes)} DUPLICATE job_descs in queue -- needs manual killall: {dupes}")
    return dupes


def launch_all(dry: bool):
    _, extra_options = training_job_api_from_profile("default")
    region = extra_options["region"]
    descs = set(in_progress_descs())
    n = len(rjp.CE_TEMPERATURE_EXPERIMENTS)
    print(f"CE-temperature sweep: {n} experiments; {len(descs)} jobs currently in progress.")
    for index, experiment in enumerate(rjp.CE_TEMPERATURE_EXPERIMENTS):
        _, exp_suffix, out_dir_name = build_payload(experiment, region)
        desc = job_desc_for(exp_suffix)
        if os.path.isdir(out_dir_name):
            print(f"[skip: artifact exists] {exp_suffix}")
            continue
        if desc in descs:
            print(f"[skip: already in queue] {exp_suffix}")
            continue
        if dry:
            print(f"[dry] would submit index={index} {exp_suffix}")
            continue
        # Submit in a timeout-guarded child so a slow-gateway retry cannot duplicate.
        print(f"[submit index={index}] {exp_suffix} ...", flush=True)
        try:
            cp = subprocess.run(
                ["timeout", str(SUBMIT_TIMEOUT_SEC), sys.executable, os.path.abspath(__file__), "--submit-index", str(index)],
                capture_output=True,
                text=True,
                timeout=SUBMIT_TIMEOUT_SEC + 15,
            )
            if cp.stdout.strip():
                print("   " + cp.stdout.strip().replace("\n", "\n   "))
            if cp.returncode != 0 and cp.stderr.strip():
                print("   stderr: " + cp.stderr.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            print("   child timed out (submission may or may not have landed) -- will verify")
        # Verify via a fresh fetch; only resubmit once if it truly did not land.
        time.sleep(6)
        live = set(in_progress_descs())
        if desc in live:
            print("   verified: in queue")
            descs.add(desc)
        elif os.path.isdir(out_dir_name):
            print("   verified: artifact dir created")
            descs.add(desc)
        else:
            print("   NOT found after submit -- retrying once")
            subprocess.run(
                ["timeout", str(SUBMIT_TIMEOUT_SEC), sys.executable, os.path.abspath(__file__), "--submit-index", str(index)],
                text=True,
            )
            time.sleep(6)
            if job_desc_for(exp_suffix) in set(in_progress_descs()):
                print("   verified on retry: in queue")
                descs.add(desc)
            else:
                print("   STILL not found -- leaving for a later re-run")
        # Gentle stagger so we never hammer the gateway (duplicate-risk mitigation).
        time.sleep(4)
    print("\n--- post-launch verification ---")
    verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true", help="Print planned submissions, do not submit.")
    parser.add_argument("--verify", action="store_true", help="List temperature jobs in queue + flag duplicates.")
    parser.add_argument("--submit-index", type=int, default=None, help="Internal: submit exactly one experiment by index.")
    args = parser.parse_args()

    if args.submit_index is not None:
        sys.exit(submit_one_child(args.submit_index))
    if args.verify:
        dupes = verify()
        sys.exit(1 if dupes else 0)
    launch_all(dry=args.dry)


if __name__ == "__main__":
    main()
