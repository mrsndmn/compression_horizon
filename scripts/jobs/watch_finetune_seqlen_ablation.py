"""Watcher for the causal-LM finetune SEQUENCE-LENGTH ablation.

Polls the four ``run_jobs_finetune_seqlen_ablation`` variants. As each variant's finetuned checkpoint
(``…-ftw-seq{S}-lr{L}``) is saved, it submits that variant's progressive-cramming eval via the
launcher's idempotent ``--stage eval``. Exits once every variant's eval is submitted (output dir
exists or its eval job is queued/running), so running it as a background task notifies on completion.

Usage:
    python scripts/jobs/watch_finetune_seqlen_ablation.py --plan      # print states, don't wait
    python scripts/jobs/watch_finetune_seqlen_ablation.py --poll 300  # watch + submit evals
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jobs_finetune_seqlen_ablation as AB  # noqa: E402
import run_jobs_finetune_width as FW  # noqa: E402  (train job_desc)
import run_jobs_width_ablation_ft as EW  # noqa: E402  (eval render_job + job_desc)

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLS_BIN = os.environ.get("MLS_BIN", "/home/jovyan/.mlspace/envs/compression_horizon/bin/mls")
PYTHON = os.environ.get("WATCH_PYTHON", "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python")
LAUNCHER = os.path.join(PROJ, "scripts", "jobs", "run_jobs_finetune_seqlen_ablation.py")
STATE_DIR = os.path.join(PROJ, ".omc", "ablation_watch", "finetune_seqlen_ablation")
LOG_DIR = os.path.join(STATE_DIR, "logs")
WATCH_LOG = os.path.join(STATE_DIR, "watch.log")
DONE_JSON = os.path.join(STATE_DIR, "DONE.json")

ACTIVE = ("running", "pending", "completing")
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


def mls_list(limit: int = 200) -> list[dict]:
    cp = subprocess.run(
        [MLS_BIN, "job", "list", "-O", "json", "-l", str(limit)], capture_output=True, text=True, env=_env(), cwd=PROJ
    )
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
    matches = [j for j in jobs if str(j.get("job_desc", "")) == desc]
    if not matches:
        return None, "no-job"
    active = [j for j in matches if str(j.get("status", "")).lower() in ACTIVE]
    j = (active or matches)[0]
    return j.get("job_name"), str(j.get("status", "")).lower()


def model_saved(out_dir: str) -> bool:
    abs_dir = os.path.join(PROJ, out_dir)
    if not os.path.isfile(os.path.join(abs_dir, "config.json")):
        return False
    try:
        return any(f.endswith(".safetensors") for f in os.listdir(abs_dir))
    except OSError:
        return False


def eval_has_output(out_dir: str) -> bool:
    return os.path.isfile(os.path.join(PROJ, out_dir, "progressive_prefixes", "dataset_info.json"))


def build_targets() -> list[dict]:
    targets = []
    for v in AB.VARIANTS:
        ckpt = AB.variant_dir(v)
        _, eval_suffix, eval_out = EW.render_job(AB.eval_experiment(v))
        targets.append(
            {
                "label": os.path.basename(ckpt),
                "train_out": ckpt,
                "train_desc": FW.job_desc_for(os.path.basename(ckpt)),
                "eval_out": eval_out,
                "eval_desc": EW.job_desc_for(eval_suffix),
            }
        )
    return targets


def classify(target: dict, jobs: list[dict], active_descs: set[str]) -> tuple[str, str | None]:
    if eval_has_output(target["eval_out"]) or target["eval_desc"] in active_descs:
        return "eval_submitted", None
    if model_saved(target["train_out"]):
        return "train_ready", None
    name, status = status_for_desc(target["train_desc"], jobs)
    if status in FINAL_FAILED:
        return "train_failed", name
    return "train_running", name


def submit_ready_evals() -> None:
    cp = subprocess.run([PYTHON, LAUNCHER, "--stage", "eval"], capture_output=True, text=True, env=_env(), cwd=PROJ)
    for line in (cp.stdout or "").splitlines():
        if "job_name" in line or "exists, skip" in line or "Would launch" in line:
            log(f"  eval-submit: {line.strip()}")
    if cp.returncode != 0:
        log(f"  WARN: eval submit returned {cp.returncode}: {(cp.stderr or '').strip()[-200:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poll", type=int, default=300, help="Seconds between polls (default 300).")
    parser.add_argument("--plan", action="store_true", help="Print current states once and exit.")
    args = parser.parse_args()

    targets = build_targets()

    if args.plan:
        jobs = mls_list()
        active = {str(j.get("job_desc", "")) for j in jobs if str(j.get("status", "")).lower() in ACTIVE}
        log(f"PLAN -- finetune-seqlen-ablation watcher ({len(targets)} variants):")
        for t in targets:
            state, name = classify(t, jobs, active)
            log(f"  [{state:14}] {t['label']}  (job={name})")
        return 0

    log(f"=== finetune-seqlen-ablation watcher start (poll={args.poll}s, {len(targets)} variants) ===")
    seen_failed: set[str] = set()
    while True:
        jobs = mls_list()
        active = {str(j.get("job_desc", "")) for j in jobs if str(j.get("status", "")).lower() in ACTIVE}
        states = {t["label"]: classify(t, jobs, active) for t in targets}

        if any(s == "train_ready" for s, _ in states.values()):
            ready = [lbl for lbl, (s, _) in states.items() if s == "train_ready"]
            log(f"checkpoint(s) ready -> submitting eval: {', '.join(ready)}")
            submit_ready_evals()
            jobs = mls_list()
            active = {str(j.get("job_desc", "")) for j in jobs if str(j.get("status", "")).lower() in ACTIVE}
            states = {t["label"]: classify(t, jobs, active) for t in targets}

        for lbl, (state, name) in states.items():
            log(f"  {lbl}: {state}" + (f" (job={name})" if name else ""))
            if state == "train_failed" and name and name not in seen_failed:
                seen_failed.add(name)
                log(f"    -> finetune failed (job={name}); recreate it to have its eval auto-submitted.")

        n_sub = sum(1 for s, _ in states.values() if s == "eval_submitted")
        log(f"--- {n_sub}/{len(targets)} eval-submitted ---")
        if n_sub == len(targets):
            log("All finetune-seqlen-ablation evals submitted. Watcher exiting.")
            os.makedirs(STATE_DIR, exist_ok=True)
            json.dump(
                {"status": "complete", "finished_at": datetime.now().isoformat(), "variants": list(states.keys())},
                open(DONE_JSON, "w"),
                indent=2,
            )
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
