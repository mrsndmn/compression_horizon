"""Watcher: block until every compression-head progressive-EVAL job has finished, then exit.

Unlike ``watch_compression_head_eval.py`` (which *submits* an eval as each CH checkpoint saves),
this script waits for the already-submitted eval jobs to *complete*, so the caller can then render
``tab:ch_qformer_layer_ablation`` and write up the results.

Completion is detected via ``get_in_progress_jobs()`` (NOT a recent-N ``mls job list`` window): an
eval is "still running" iff its ``job_desc`` is in the in-progress set. This is robust to the global
job list aging out over a long (many-hour) wait, and naturally handles a failed job that the user
recreates (it reappears in-progress, so the watcher keeps waiting). The per-experiment
``progressive_prefixes`` sample count is logged as a progress/ETA signal and used only to flag a
finished-but-short (likely failed) eval in the final summary.

Usage:
    python scripts/jobs/watch_compression_head_eval_done.py --plan        # print states once, exit
    python scripts/jobs/watch_compression_head_eval_done.py --poll 600    # wait until all evals done
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jobs_compression_head as J  # noqa: E402
from mls.manager.job.utils import get_in_progress_jobs  # noqa: E402

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STATE_DIR = os.path.join(PROJ, ".omc", "ablation_watch", "compression_head_eval")
DONE_LOG = os.path.join(STATE_DIR, "eval_done.log")
TARGET_SAMPLES = J.PROG_LIMIT_DATASET_ITEMS  # progressive eval set size (50)


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(DONE_LOG, "a") as fh:
        fh.write(line + "\n")


def eval_targets() -> list[tuple[str, str, str]]:
    """Return (label, eval_job_desc, eval_out_dir) for every experiment's progressive eval."""
    targets = []
    for exp in J.EXPERIMENTS:
        _, ch_suffix, ch_out = J.render_ch_job(exp)
        _, eval_suffix, eval_out = J.render_eval_job(ch_out, ch_suffix)
        targets.append((os.path.basename(ch_out), J.eval_job_desc(eval_suffix), eval_out))
    return targets


def sample_count(eval_out: str) -> int:
    return len(glob.glob(os.path.join(PROJ, eval_out, "progressive_prefixes", "*")))


def snapshot() -> tuple[list[dict], bool]:
    """One poll: per-experiment {label, running, samples, dir_exists}; plus all_done flag."""
    in_progress = {str(j.get("job_desc", "")) for j in get_in_progress_jobs()}
    rows = []
    for label, desc, eval_out in eval_targets():
        running = desc in in_progress
        rows.append(
            {
                "label": label,
                "running": running,
                "samples": sample_count(eval_out),
                "dir_exists": os.path.isdir(os.path.join(PROJ, eval_out)),
            }
        )
    all_done = not any(r["running"] for r in rows)
    return rows, all_done


def _print(rows: list[dict]) -> None:
    for r in rows:
        state = "running" if r["running"] else ("done" if r["dir_exists"] else "missing")
        log(f"  [{state:8}] {r['label']}: {r['samples']}/{TARGET_SAMPLES} samples")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poll", type=int, default=600, help="Seconds between polls (default 600).")
    parser.add_argument("--plan", action="store_true", help="Print current states once and exit.")
    args = parser.parse_args()

    if args.plan:
        rows, all_done = snapshot()
        log("PLAN -- compression-head eval-completion watcher:")
        _print(rows)
        log(f"all_done={all_done}")
        return 0

    log(f"=== eval-completion watcher start (poll={args.poll}s, {len(J.EXPERIMENTS)} evals) ===")
    while True:
        rows, all_done = snapshot()
        n_running = sum(r["running"] for r in rows)
        _print(rows)
        log(f"--- {len(rows) - n_running}/{len(rows)} evals finished, {n_running} still running ---")
        if all_done:
            short = [r["label"] for r in rows if r["samples"] < TARGET_SAMPLES]
            log("All eval jobs finished.")
            if short:
                log(f"WARNING: finished but < {TARGET_SAMPLES} samples (likely failed/incomplete): {', '.join(short)}")
            return 1 if short else 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
