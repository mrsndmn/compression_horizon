#!/usr/bin/env python3
"""Watcher for the T5Gemma encoder-inversion jobs: exits as soon as there is something to say.

Polls the MLS status of every job whose description carries the experiment tag and stops on the
first outcome that needs a human (or the agent that launched it):

* **any job failed / stopped / terminated** -> exit code 1, with that job's tail log so the failure
  can be diagnosed and the job relaunched;
* **every tracked job completed** -> exit code 0, with the per-arm artifact summary.

Anything else (Pending/Running) just prints a status line and sleeps. Run it as a background task
so its exit is the notification.

Usage:
    python scripts/jobs/watch_t5gemma_inversion.py --job lm-mpi-job-... lm-mpi-job-...
    python scripts/jobs/watch_t5gemma_inversion.py --interval 300
    python scripts/jobs/watch_t5gemma_inversion.py --once
    python scripts/jobs/watch_t5gemma_inversion.py --expect 9   # don't finish early: wait for N jobs
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from mls.manager.job.utils import training_job_api_from_profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jobs_t5gemma_inversion as launcher  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOB_DESC_TAG = "t5gemma-inversion"
FAILED_STATUSES = {"failed", "stopped", "deleted", "terminated"}
COMPLETED_STATUSES = {"completed"}


def tracked_jobs(client, region: str, job_names: list[str] | None = None) -> list[dict]:
    """Status of every tracked job.

    ``get_list_jobs`` drops jobs once they reach a final status, so listing alone can never see
    the all-completed case — it just reports an empty set and waits forever. Pass ``--job`` with
    the launcher's job names to poll each one by name instead; the listing is only a fallback for
    when the names were not recorded.
    """
    if job_names:
        jobs = []
        for name in job_names:
            try:
                jobs.append(client.get_job_status(name))
            except Exception as exc:
                print(f"[watch] could not read status of {name}: {exc}", flush=True)
        return jobs

    jobs = client.get_list_jobs(region=region, allocation_name=None, status=None, limit=200, offset=0)
    if isinstance(jobs, dict):
        jobs = jobs.get("jobs", jobs.get("items", []))
    return [job for job in jobs if JOB_DESC_TAG in (job.get("job_desc") or "")]


def status_of(job: dict) -> str:
    return str(job.get("status") or "").lower()


def arm_of(job: dict) -> str:
    """Short label for the arm, recovered from the job description."""
    desc = job.get("job_desc") or ""
    for token in ("inj_encoder_input", "inj_encoder_output"):
        if token in desc:
            return token.replace("inj_", "t5gemma/")
    if "gemma-2-2b" in desc:
        return "gemma2-baseline"
    return desc[:40]


def print_status(jobs: list[dict]) -> None:
    print(f"[watch] {time.strftime('%Y-%m-%d %H:%M:%S')} tracking {len(jobs)} job(s)", flush=True)
    for job in sorted(jobs, key=lambda j: j.get("job_desc") or ""):
        stage = "eval" if " eval " in (job.get("job_desc") or "") else "train"
        print(f"[watch]   {status_of(job):<10} {stage:<5} {arm_of(job):<24} {job.get('job_name')}", flush=True)


def artifact_summary() -> list[str]:
    """One line per expected output dir, saying whether its artifacts landed."""
    lines = []
    for stage in ("train", "eval"):
        for job in launcher.build_jobs(stage):
            out_dir = os.path.join(REPO, job["out_dir"])
            if stage == "train":
                landed = os.path.isdir(os.path.join(out_dir, "progressive_prefixes"))
            else:
                landed = os.path.isfile(os.path.join(out_dir, "results.json"))
            lines.append(f"  [{'ok ' if landed else 'MISS'}] {job['arm']:<28} {job['out_dir']}")
    return lines


def tail_logs(client, region: str, job_name: str, lines: int = 40) -> str:
    try:
        logs = client.get_job_logs(name=job_name, region=region, tail=lines)
    except Exception as exc:  # the failure report matters more than the log fetch
        return f"(could not fetch logs: {exc})"
    if isinstance(logs, (list, tuple)):
        logs = "\n".join(str(entry) for entry in logs)
    return str(logs)[-4000:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch the T5Gemma encoder-inversion jobs.")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between polls.")
    parser.add_argument("--once", action="store_true", help="Poll once and exit.")
    parser.add_argument(
        "--expect",
        type=int,
        default=0,
        help="Wait until at least this many jobs are tracked before declaring everything done. "
        "0 = use whatever is currently tagged.",
    )
    parser.add_argument(
        "--job",
        nargs="+",
        default=None,
        help="Full job names to poll by name (what the launcher printed). Strongly preferred: the "
        "job listing drops jobs once they finish, so name-based polling is the only way to observe "
        "the all-completed case.",
    )
    args = parser.parse_args()

    client, extra_options = training_job_api_from_profile("default")
    region = extra_options["region"]

    while True:
        jobs = tracked_jobs(client, region, args.job)
        print_status(jobs)

        failed = [job for job in jobs if status_of(job) in FAILED_STATUSES]
        if failed:
            print(f"\n[watch] FAILED: {len(failed)} job(s) did not finish cleanly.", flush=True)
            for job in failed:
                print(f"\n[watch] === {arm_of(job)} :: {job.get('job_name')} :: {status_of(job)} ===", flush=True)
                print(tail_logs(client, region, job.get("job_name")), flush=True)
            return 1

        done = [job for job in jobs if status_of(job) in COMPLETED_STATUSES]
        enough = len(jobs) >= args.expect if args.expect else bool(jobs)
        if enough and jobs and len(done) == len(jobs):
            print("\n[watch] ALL DONE: every tracked job completed.", flush=True)
            print("\n".join(artifact_summary()), flush=True)
            return 0

        if args.once:
            print("[watch] --once: still in progress.", flush=True)
            return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
