"""Watcher: launch progressive-cramming evaluation for each compression-head checkpoint
as soon as that head's training job finishes (independently, in any completion order).

For every experiment in ``run_jobs_compression_head.EXPERIMENTS`` it polls one of four states:
  * ``ch_running``    -- the training job is pending/running (logs live TB step/ETA).
  * ``ch_ready``      -- the checkpoint is saved (config.json + .safetensors) -> submit its eval now.
  * ``eval_submitted``-- the progressive-eval job is queued/running (or its output dir exists).
  * ``ch_failed``     -- the training job reached a terminal status without producing a checkpoint.

Submission reuses ``run_jobs_compression_head.py --stage eval`` (idempotent: it skips heads that are
not ready, evals already queued, and existing output dirs), so the watcher is safe to re-run and
resilient to a missed/failed submission. It exits once every experiment is ``eval_submitted`` or
``ch_failed``.

Usage:
    python scripts/jobs/watch_compression_head_eval.py --plan          # print states, don't wait
    python scripts/jobs/watch_compression_head_eval.py --poll 180      # watch + submit evals
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jobs_compression_head as J  # noqa: E402

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLS_BIN = os.environ.get("MLS_BIN", "/home/jovyan/.mlspace/envs/compression_horizon/bin/mls")
PYTHON = os.environ.get("WATCH_PYTHON", "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python")
LAUNCHER = os.path.join(PROJ, "scripts", "jobs", "run_jobs_compression_head.py")
STATE_DIR = os.path.join(PROJ, ".omc", "ablation_watch", "compression_head_eval")
LOG_DIR = os.path.join(STATE_DIR, "logs")
WATCH_LOG = os.path.join(STATE_DIR, "watch.log")
# Optimizer steps in one epoch, derived from the launcher so it tracks data-size/batch changes
# (for ETA display only): total_sequences * seq_len * epochs / global-batch-tokens.
CH_TRAIN_MAX_STEPS = J.LIMIT_DATASET_ITEMS * J.MAX_SEQ_LEN * J.NUM_TRAIN_EPOCHS // J.TARGET_GLOBAL_TOKENS

FINAL_FAILED = {"failed", "stopped", "deleted", "terminated", "error", "cancelled", "canceled", "killed", "aborted"}


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(WATCH_LOG, "a") as fh:
        fh.write(line + "\n")


def _env() -> dict:
    e = os.environ.copy()
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        e.pop(k, None)
    e.setdefault("PYTHONPATH", os.path.join(PROJ, "src"))
    return e


def _mls(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([MLS_BIN, "job", *args], capture_output=capture, text=True, env=_env(), cwd=PROJ)


def mls_list(limit: int = 200) -> list[dict]:
    cp = _mls(["list", "-O", "json", "-l", str(limit)])
    if cp.returncode != 0:
        log(f"  WARN: mls job list failed: {cp.stderr.strip()[-200:]}")
        return []
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        for key in ("jobs", "items", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return []
    return data if isinstance(data, list) else []


def status_for_desc(desc: str, jobs: list[dict]) -> tuple[str | None, str]:
    """Return (job_name, status) for the most relevant job matching ``desc`` (active preferred)."""
    matches = [j for j in jobs if str(j.get("job_desc", "")) == desc]
    if not matches:
        return None, "no-job"
    active = [j for j in matches if str(j.get("status", "")).lower() in ("running", "pending", "completing")]
    j = (active or matches)[0]
    return j.get("job_name"), str(j.get("status", "")).lower()


def save_logs(name: str, tail: int = 400) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    cp = _mls(["logs", name, "-t", str(tail)])
    path = os.path.join(LOG_DIR, f"{name}.log")
    with open(path, "w") as fh:
        fh.write(cp.stdout or "")
        if cp.stderr:
            fh.write("\n--- stderr ---\n" + cp.stderr)
    return path


def ch_tb_eta(out_dir_rel: str) -> str | None:
    """Best-effort 'step/loss ETA' from the compression-head TB log (logging_dir = <out_dir>/logs)."""
    events = sorted(glob.glob(os.path.join(PROJ, out_dir_rel, "logs", "events*")))
    if not events:
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(events[-1])
        ea.Reload()
        if "loss/total" not in ea.Tags().get("scalars", []):
            return None
        ev = ea.Scalars("loss/total")
    except Exception:
        return None
    if not ev:
        return None
    last = ev[-1]
    eta = "?"
    if len(ev) >= 2 and last.step < CH_TRAIN_MAX_STEPS:
        dstep, dt = last.step - ev[0].step, last.wall_time - ev[0].wall_time
        if dstep > 0 and dt > 0:
            rem = (CH_TRAIN_MAX_STEPS - last.step) / (dstep / dt)
            h, m = int(rem // 3600), int((rem % 3600) // 60)
            eta = f"{h}h{m:02d}m" if h else f"{m}m"
    return f"step {last.step}/{CH_TRAIN_MAX_STEPS} loss {last.value:.3f} ETA {eta}"


def classify(jobs: list[dict]) -> dict[str, dict]:
    """Map experiment -> {state, ch_out, label, job_name, status}."""
    active_descs = {
        str(j.get("job_desc", "")) for j in jobs if str(j.get("status", "")).lower() in ("running", "pending", "completing")
    }
    out: dict[str, dict] = {}
    for exp in J.EXPERIMENTS:
        _, ch_suffix, ch_out = J.render_ch_job(exp)
        _, eval_suffix, eval_out = J.render_eval_job(ch_out, ch_suffix)
        label = os.path.basename(ch_out)
        if os.path.exists(os.path.join(PROJ, eval_out)) or J.eval_job_desc(eval_suffix) in active_descs:
            state = "eval_submitted"
            name, status = None, "-"
        elif J.checkpoint_ready(os.path.join(PROJ, ch_out)):
            state = "ch_ready"
            name, status = None, "-"
        else:
            name, status = status_for_desc(J.ch_job_desc(ch_suffix), jobs)
            state = "ch_failed" if status in FINAL_FAILED else "ch_running"
        out[label] = {"state": state, "ch_out": ch_out, "job_name": name, "status": status}
    return out


def submit_ready_evals() -> None:
    """Reuse the idempotent launcher to submit evals for any ready, not-yet-submitted heads."""
    cp = subprocess.run([PYTHON, LAUNCHER, "--stage", "eval"], capture_output=True, text=True, env=_env(), cwd=PROJ)
    for line in (cp.stdout or "").splitlines():
        if "job_name" in line or "exists, skip" in line:
            log(f"  eval-submit: {line.strip()}")
    if cp.returncode != 0:
        log(f"  WARN: eval submit returned {cp.returncode}: {(cp.stderr or '').strip()[-200:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poll", type=int, default=180, help="Seconds between polls (default 180).")
    parser.add_argument("--plan", action="store_true", help="Print current states once and exit.")
    args = parser.parse_args()

    if args.plan:
        states = classify(mls_list())
        log("PLAN -- compression-head eval watcher (no waiting):")
        for label, info in states.items():
            log(f"  [{info['state']:14}] {label}  (job={info['job_name']} status={info['status']})")
        return 0

    log(f"=== compression-head eval watcher start (poll={args.poll}s, {len(J.EXPERIMENTS)} experiments) ===")
    while True:
        jobs = mls_list()
        states = classify(jobs)

        if any(v["state"] == "ch_ready" for v in states.values()):
            ready = [lbl for lbl, v in states.items() if v["state"] == "ch_ready"]
            log(f"checkpoint(s) ready -> submitting eval: {', '.join(ready)}")
            submit_ready_evals()
            jobs = mls_list()
            states = classify(jobs)

        for label, info in states.items():
            if info["state"] == "ch_running":
                eta = ch_tb_eta(info["ch_out"])
                log(f"  {label}: {info['status']}" + (f" | {eta}" if eta else ""))
            else:
                log(f"  {label}: {info['state']}")
            if info["state"] == "ch_failed" and info["job_name"]:
                p = save_logs(info["job_name"])
                log(f"    -> training failed; logs saved to {os.path.relpath(p, PROJ)} (no eval will be submitted)")

        done = all(v["state"] in ("eval_submitted", "ch_failed") for v in states.values())
        n_eval = sum(v["state"] == "eval_submitted" for v in states.values())
        n_fail = sum(v["state"] == "ch_failed" for v in states.values())
        log(f"--- {n_eval} eval-submitted, {n_fail} failed, {len(states) - n_eval - n_fail} pending ---")
        if done:
            log("All compression-head jobs resolved (eval submitted or failed). Watcher exiting.")
            return 0 if n_fail == 0 else 1
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
