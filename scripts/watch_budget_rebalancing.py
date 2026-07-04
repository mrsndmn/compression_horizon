"""Watch the 12 Pythia-1.4B budget-rebalancing jobs; aggregate + greedy-eval as they land.

Runs as a background poller. Every ``--interval`` seconds it:

  1. For each finished run (has ``progressive_prefixes``) that lacks a greedy-reconstruction cache,
     runs ``scripts/greedy_reconstruction_eval.py`` (GPU) so the greedy_match column fills in.
  2. Re-runs ``scripts/aggregate_budget_rebalancing.py`` -> refreshes the CSV + a markdown table.
  3. Checks MLS for how many of the 12 budget-rebalance jobs are still in progress.

Exits when all 12 target runs have a ``progressive_prefixes`` dataset (done) OR none of the 12 jobs
remain in progress (so no further artifacts can appear) OR ``--max-hours`` elapses. Everything it does
is idempotent and re-runnable, so a later manual invocation of the aggregation script reproduces the
same table. It never launches or kills jobs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time

RESULTS_DIR = "artifacts/results"
CSV_PATH = os.path.join(RESULTS_DIR, "budget_rebalancing.csv")
MD_PATH = os.path.join(RESULTS_DIR, "budget_rebalancing_table.md")
LOG_PATH = os.path.join(RESULTS_DIR, "watch_budget_rebalancing.log")

# The 12 target run dirs (2 base x 3 eps x {cap,dual}); mirror aggregate_budget_rebalancing.BASES.
BASES = [
    "sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5",
    "sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.5",
]
EPSILONS = [0.5, 1.0, 2.0]
MODES = ["cap", "dual"]
TARGETS = [f"{b}_cm_{e}_brl_{m}" for b in BASES for e in EPSILONS for m in MODES]
ARTIFACT_ROOT = "artifacts/experiments_progressive"


def log(msg: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def has_dataset(run_dir: str) -> bool:
    return os.path.exists(os.path.join(run_dir, "progressive_prefixes", "dataset_info.json"))


def has_greedy(run_dir: str) -> bool:
    return os.path.exists(os.path.join(run_dir, "greedy_accuracy_cache.json"))


def in_progress_brl() -> int:
    """Count the budget-rebalance Pythia jobs still queued/running (0 on any MLS error)."""
    try:
        from mls.manager.job.utils import get_in_progress_jobs

        jobs = get_in_progress_jobs()
        return sum(1 for j in jobs if "brl_" in j.get("job_desc", "") and "pythia" in j.get("job_desc", ""))
    except Exception as e:  # noqa: BLE001 - transient gateway errors must not kill the watcher
        log(f"WARN could not query MLS jobs: {type(e).__name__}: {e}")
        return -1


def run_greedy(run_dir: str) -> None:
    log(f"greedy-eval {os.path.basename(run_dir)}")
    env = dict(os.environ, PYTHONPATH="./src:.")
    try:
        subprocess.run(
            ["python", "scripts/greedy_reconstruction_eval.py", "--run-dir", run_dir, "--dataset-type", "progr"],
            env=env,
            timeout=3600,
            check=False,
        )
    except Exception as e:  # noqa: BLE001
        log(f"WARN greedy-eval failed for {run_dir}: {type(e).__name__}: {e}")


def aggregate() -> None:
    env = dict(os.environ, PYTHONPATH="./src")
    res = subprocess.run(
        ["python", "scripts/aggregate_budget_rebalancing.py", "--csv", CSV_PATH],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    table = res.stdout
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(MD_PATH, "w") as f:
        f.write("# Budget-rebalancing aggregation (Pythia-1.4B)\n\n```\n" + table + "\n```\n")
    # Log only the trailing progress line to keep the watch log compact.
    for ln in table.strip().splitlines()[-1:]:
        log(f"aggregate: {ln}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=int, default=600, help="Seconds between polls (default 600).")
    ap.add_argument("--max-hours", type=float, default=23.0, help="Wall-clock cap before exiting with partial results.")
    ap.add_argument("--no-greedy", action="store_true", help="Skip the GPU greedy-reconstruction eval.")
    args = ap.parse_args()

    log(f"watch start: {len(TARGETS)} target runs, interval={args.interval}s, max_hours={args.max_hours}")
    start = time.monotonic()
    polls = 0
    while True:
        polls += 1
        dirs = [os.path.join(ARTIFACT_ROOT, t) for t in TARGETS]
        done = [d for d in dirs if has_dataset(d)]
        if not args.no_greedy:
            for d in done:
                if not has_greedy(d):
                    run_greedy(d)
        aggregate()
        remaining = in_progress_brl()
        log(f"poll #{polls}: {len(done)}/{len(TARGETS)} runs have datasets; {remaining} brl jobs still in progress")

        if len(done) == len(TARGETS):
            log("all target runs have datasets -> final aggregate + exit")
            aggregate()
            break
        if remaining == 0:
            log("no brl jobs in progress and not all datasets present -> some jobs ended without artifacts; exit")
            break
        if (time.monotonic() - start) / 3600.0 >= args.max_hours:
            log("max-hours reached -> exit with partial results")
            break
        time.sleep(args.interval)
    log("watch done")


if __name__ == "__main__":
    main()
