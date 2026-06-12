"""Background watcher for basin-area MLS jobs.

Waits for all 4 model jobs to finish, then regenerates the multi-model
figure and copies it to paper/figures/.

Usage:
    python scripts/jobs/watch_basin_area.py --poll 120
    python scripts/jobs/watch_basin_area.py --plan   # check status without waiting
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_jobs_basin_area import MODELS, job_desc_for  # noqa: E402

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLS_BIN = os.environ.get("MLS_BIN", "/home/jovyan/.mlspace/envs/compression_horizon/bin/mls")
PYTHON = os.environ.get("WATCH_PYTHON", "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python")

STATE_DIR = os.path.join(PROJ, ".omc", "basin_area_watch")
LOG_FILE = os.path.join(STATE_DIR, "watch.log")


# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as fh:
        fh.write(line + "\n")


def _env() -> dict:
    e = os.environ.copy()
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        e.pop(k, None)
    return e


def _mls(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([MLS_BIN, "job", *args], capture_output=capture, text=True, env=_env(), cwd=PROJ)


def mls_list(limit: int = 1000) -> list[dict]:
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
    log_dir = os.path.join(STATE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    cp = _mls(["logs", name, "-t", str(tail)])
    path = os.path.join(log_dir, f"{name}.log")
    with open(path, "w") as fh:
        fh.write(cp.stdout or "")
        if cp.stderr:
            fh.write("\n--- stderr ---\n" + cp.stderr)
    return path


def discover_job(desc: str) -> str | None:
    jobs = mls_list()
    matches = [j for j in jobs if str(j.get("job_desc", "")) == desc]
    if not matches:
        return None
    active = [j for j in matches if str(j.get("status", "")).lower() in ("running", "pending", "completing")]
    return (active or matches)[0].get("job_name")


def has_all_npz(exp_dir: str, n_samples: int = 10) -> bool:
    viz_dir = os.path.join(PROJ, exp_dir, "visualizations")
    return all(os.path.isfile(os.path.join(viz_dir, f"basin_area_sample_{i}.npz")) for i in range(n_samples))


# --------------------------------------------------------------------------- #
def wait_for_model(model: dict, poll: int) -> bool:
    name = model["name"]
    desc = job_desc_for(name)
    exp_dir = model["exp_dir"]

    job_name = discover_job(desc)
    if job_name is None:
        if has_all_npz(exp_dir):
            log(f"{name}: no job found but all NPZs exist, OK")
            return True
        log(f"{name}: no job found and NPZs missing — FAIL")
        return False

    log(f"{name}: waiting on {job_name}")
    rc = mls_wait(job_name, poll)
    status = str(mls_status(job_name).get("status", "")).lower()

    if rc == 0 and has_all_npz(exp_dir):
        log(f"{name}: SUCCESS (status={status})")
        return True

    log(f"{name}: job ended (wait_rc={rc}, status={status}, npzs={has_all_npz(exp_dir)})")
    log_path = save_logs(job_name)
    log(f"{name}: logs saved to {os.path.relpath(log_path, PROJ)}")
    return False


def regenerate_figure() -> bool:
    log("Regenerating multi-model basin area figure...")
    cmd = [
        PYTHON,
        "scripts/paper/plot_basin_area_vs_stage.py",
        "--plot_multi",
        "--multi_exp_dirs",
        *[m["exp_dir"] for m in MODELS],
        "--multi_model_names",
        *[m["name"] for m in MODELS],
        "--sample_ids",
        *[str(i) for i in range(10)],
        "--output",
        "paper/figures/basin_area_vs_stage_multi.png",
    ]
    env = _env()
    env["PYTHONPATH"] = "./src:." + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=PROJ)
    if cp.stdout:
        print(cp.stdout, flush=True)
    if cp.returncode != 0:
        log(f"  figure generation FAILED: {cp.stderr.strip()[-400:]}")
        return False
    log("  figure saved to paper/figures/basin_area_vs_stage_multi.{png,pdf}")
    return True


# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poll", type=int, default=120, help="Polling interval in seconds")
    parser.add_argument("--plan", action="store_true", help="Check status without waiting")
    args = parser.parse_args()

    os.makedirs(STATE_DIR, exist_ok=True)

    if args.plan:
        log("PLAN (status check, no waiting):")
        for model in MODELS:
            desc = job_desc_for(model["name"])
            job_name = discover_job(desc)
            st = str(mls_status(job_name).get("status", "?")) if job_name else "no-job"
            npz_ok = has_all_npz(model["exp_dir"])
            log(f"  {model['name']}: job={job_name} status={st} npzs={'OK' if npz_ok else 'MISSING'}")
        return 0

    log(f"=== basin_area watcher start (poll={args.poll}s) ===")

    results = {}
    for model in MODELS:
        results[model["name"]] = wait_for_model(model, args.poll)

    if not all(results.values()):
        failed = [k for k, v in results.items() if not v]
        log(f"NOT all jobs succeeded: {failed}. Figure not regenerated.")
        return 1

    log("All jobs finished successfully!")
    if not regenerate_figure():
        return 1

    log("=== basin_area watcher done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
