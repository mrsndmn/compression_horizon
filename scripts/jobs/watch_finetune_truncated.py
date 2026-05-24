"""Two-phase background watcher for the finetuned transformer-depth ablation.

Phase 1 -- finetuning (this script):
  Wait on each 8-GPU ``CH: ft-truncated`` job (one per firstlast checkpoint).
  On failure, save ``mls job logs`` and resubmit via
  ``run_jobs_finetune_truncated.submit_experiment(force=True)`` up to
  ``--max-retries``. A job is "done" when its ``…-ft`` checkpoint dir holds a
  saved model (config.json + a .safetensors file).

Phase 2 -- progressive re-eval + paper (delegated):
  Once every ``…-ft`` checkpoint exists, hand off to
  ``scripts/jobs/watch_ablation.py --launcher run_jobs_layer_ablation_ft``,
  which submits the baseline progressive runs on the finetuned checkpoints,
  waits for them, regenerates ``tab:layer_ablation`` (interleaved un-ft / ft
  rows), fills the ``layer-ablation-trend`` AUTOGEN sentence, and lints.

Usage:
    python scripts/jobs/watch_finetune_truncated.py --plan
    python scripts/jobs/watch_finetune_truncated.py --max-retries 2 --poll 180
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jobs_finetune_truncated as F  # noqa: E402

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLS_BIN = os.environ.get("MLS_BIN", "/home/jovyan/.mlspace/envs/compression_horizon/bin/mls")
PYTHON = os.environ.get("WATCH_PYTHON", "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python")
STATE_DIR = os.path.join(PROJ, ".omc", "ablation_watch", "finetune_truncated")
LOG_DIR = os.path.join(STATE_DIR, "logs")
WATCH_LOG = os.path.join(STATE_DIR, "watch.log")

_client = None
_extra = None


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
    return e


def _mls(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([MLS_BIN, "job", *args], capture_output=capture, text=True, env=_env(), cwd=PROJ)


def mls_list(limit: int = 100) -> list[dict]:
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


def mls_status(name: str) -> dict:
    cp = _mls(["status", name, "-O", "json"])
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        return {}


def mls_wait(name: str, poll: int) -> int:
    return _mls(["wait", name, "-i", str(poll)], capture=False).returncode


def save_logs(name: str, tail: int = 400) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    cp = _mls(["logs", name, "-t", str(tail)])
    path = os.path.join(LOG_DIR, f"{name}.log")
    with open(path, "w") as fh:
        fh.write(cp.stdout or "")
        if cp.stderr:
            fh.write("\n--- stderr ---\n" + cp.stderr)
    return path


def ft_done(out_dir: str) -> bool:
    """A finetune job is done when the checkpoint dir holds a saved model."""
    abs_dir = os.path.join(PROJ, out_dir)
    if not os.path.isfile(os.path.join(abs_dir, "config.json")):
        return False
    return any(f.endswith(".safetensors") for f in os.listdir(abs_dir)) if os.path.isdir(abs_dir) else False


def discover_job(job_desc: str) -> str | None:
    matches = [j for j in mls_list() if str(j.get("job_desc", "")) == job_desc]
    if not matches:
        return None
    active = [j for j in matches if str(j.get("status", "")).lower() in ("running", "pending", "completing")]
    return (active or matches)[0].get("job_name")


def _ensure_client():
    global _client, _extra
    if _client is None:
        _client, _extra = F.make_client()
    return _client, _extra


def _print_log_tail(path: str, n: int = 40) -> None:
    try:
        tail = open(path).read().splitlines()[-n:]
    except OSError:
        return
    print("    --- log tail ---", flush=True)
    for ln in tail:
        print("    " + ln, flush=True)
    print("    --- end log tail ---", flush=True)


def ensure_finetune(checkpoint: str, opts: dict, max_retries: int, poll: int) -> bool:
    model_short = os.path.basename(checkpoint.rstrip("/"))
    out_dir = F.finetuned_dir(checkpoint)
    job_desc = F.job_desc_for(model_short)

    if ft_done(out_dir):
        log(f"{model_short}: finetuned checkpoint already present, OK")
        return True

    name = discover_job(job_desc)
    attempt = 0
    while True:
        if name is None:
            log(f"{model_short}: no active job, submitting")
            client, extra = _ensure_client()
            result = F.submit_experiment(checkpoint, client, extra, opts, force=True)
            name = (result or {}).get("job_name")
            if name is None:
                log(f"{model_short}: FAILED to submit")
                return False

        log(f"{model_short}: waiting on {name} (attempt {attempt + 1}/{max_retries + 1})")
        rc = mls_wait(name, poll)
        status = str(mls_status(name).get("status", "")).lower()

        if rc == 0 and ft_done(out_dir):
            log(f"{model_short}: SUCCESS (status={status or 'ok'})")
            return True

        log(f"{model_short}: ended badly (wait_rc={rc}, status={status}, ckpt={ft_done(out_dir)})")
        log_path = save_logs(name)
        log(f"{model_short}: logs saved to {os.path.relpath(log_path, PROJ)}")
        _print_log_tail(log_path)

        attempt += 1
        if attempt > max_retries:
            log(f"{model_short}: giving up after {attempt} attempts -- inspect the log above")
            return False
        name = None


def run_phase2(poll: int) -> int:
    log("Phase 1 complete -> delegating Phase 2 to watch_ablation.py (run_jobs_layer_ablation_ft)")
    cp = subprocess.run(
        [
            PYTHON,
            os.path.join(PROJ, "scripts", "jobs", "watch_ablation.py"),
            "--launcher",
            "run_jobs_layer_ablation_ft",
            "--poll",
            str(poll),
        ],
        env=_env(),
        cwd=PROJ,
    )
    log(f"Phase 2 watcher exited with code {cp.returncode}")
    return cp.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--poll", type=int, default=180)
    parser.add_argument("--plan", action="store_true", help="Print discovered jobs/checkpoints, do not wait.")
    parser.add_argument("--skip-phase2", action="store_true", help="Only wait for finetuning; don't run re-eval.")
    args = parser.parse_args()

    opts = dict(F.DEFAULTS)

    if args.plan:
        log("PLAN for finetune-truncated (no waiting):")
        for ck in F.FINETUNE_CHECKPOINTS:
            ms = os.path.basename(ck.rstrip("/"))
            name = discover_job(F.job_desc_for(ms))
            st = str(mls_status(name).get("status", "?")) if name else "no-job"
            log(f"  {ms}: ckpt_done={ft_done(F.finetuned_dir(ck))} job={name} status={st}")
        return 0

    log(f"=== finetune-truncated watcher start (max_retries={args.max_retries}, poll={args.poll}s) ===")
    results = {
        os.path.basename(ck.rstrip("/")): ensure_finetune(ck, opts, args.max_retries, args.poll)
        for ck in F.FINETUNE_CHECKPOINTS
    }
    if not all(results.values()):
        failed = [k for k, v in results.items() if not v]
        log(f"NOT all finetuning jobs succeeded: {failed}. See {os.path.relpath(LOG_DIR, PROJ)}/. Phase 2 skipped.")
        return 1

    log("All finetuning checkpoints present.")
    if args.skip_phase2:
        return 0
    return run_phase2(args.poll)


if __name__ == "__main__":
    sys.exit(main())
