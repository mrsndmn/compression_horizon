"""Background watcher for the 50-sample trajectory-cluster recompute jobs.

Companion to ``scripts/jobs/run_jobs_trajectory_clusters_50samples.py``. The four jobs it owns are the
SmolLM2-135M / 360M (scale rows) and SmolLM2-1.7B lr 0.5 / lr 1.0 (LR-sweep rows) progressive-cramming
runs at ``--limit_dataset_items 50``. The 1.7B lr 0.1 and Llama-3.1-8B 50-sample runs already exist, so
they are not owned here (but the table generator still reads their caches).

Unlike the progressive-table ablations (``scripts/jobs/watch_ablation.py``, which regenerates a
``scripts/paper/tables/progressive.py`` table and parses compressed-token counts), this ablation's
artifacts are produced by a different pipeline, so the post-processing is bespoke:

  1. ``mls job wait`` on each owned job; on genuine failure save logs and resubmit (up to
     ``--max-retries``). Jobs are submitted by the launcher, not here -- a missing job is treated as
     pre-empted/stopped and resubmitted (also guarded by ``--max-retries``).
  2. Once every owned run has saved trajectories, run
     ``scripts/analyze_trajectory_clusters.py`` on each new run (writes the per-run
     ``artifacts/analysis/trajectory_clusters_135m/<run>/summary.json`` cache the table reads).
  3. Regenerate the trajectory-cluster tables + figure
     (``scripts/paper/tables/trajectory_clusters.py --save --figure``) and run ``paper/lint_paper.py``.

Usage:
    python scripts/jobs/watch_trajectory_clusters_50samples.py --plan
    python scripts/jobs/watch_trajectory_clusters_50samples.py --max-retries 1 --poll 120
    python scripts/jobs/watch_trajectory_clusters_50samples.py --no-artifacts   # wait only, don't regen
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jobs_trajectory_clusters_50samples as L  # noqa: E402

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLS_BIN = os.environ.get("MLS_BIN", "/home/jovyan/.mlspace/envs/compression_horizon/bin/mls")
PYTHON = os.environ.get("WATCH_PYTHON", "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python")

STATE_DIR = os.path.join(PROJ, ".omc", "ablation_watch", "trajectory_clusters_50samples")
LOG_DIR = os.path.join(STATE_DIR, "logs")
WATCH_LOG = os.path.join(STATE_DIR, "watch.log")
SUMMARY_JSON = os.path.join(STATE_DIR, "summary.json")

# The runs whose summary.json caches the table generator reads. The two pre-existing 50-sample runs
# are listed so step 2 backfills their cache if it is somehow missing; the four owned runs are always
# (re)analysed once their trajectories land.
ANALYSIS_RUNS = [
    "sl_4096_SmolLM2-135M_ds_pg19_1k_limit_50_lr_0.1",
    "sl_4096_SmolLM2-360M_ds_pg19_1k_limit_50_lr_0.1",
    "sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1",
    "sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.5",
    "sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_1.0",
    "sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1",
]
ANALYSIS_CACHE_DIR = "artifacts/analysis/trajectory_clusters_135m"

_client = None
_extra_options = None


# --------------------------------------------------------------------------- #
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


def has_output(out_dir: str) -> bool:
    return os.path.isfile(os.path.join(PROJ, out_dir, "progressive_prefixes", "dataset_info.json"))


def discover_job(predicate) -> str | None:
    matches = [j for j in mls_list() if predicate(str(j.get("job_desc", "")))]
    if not matches:
        return None
    active = [j for j in matches if str(j.get("status", "")).lower() in ("running", "pending", "completing")]
    return (active or matches)[0].get("job_name")


def _ensure_client():
    global _client, _extra_options
    if _client is None:
        _client, _extra_options = L.make_client()
    return _client, _extra_options


# --------------------------------------------------------------------------- #
def build_infos() -> list[dict]:
    infos = []
    for exp in L.EXPERIMENTS:
        _, exp_suffix, out_dir = L.render_job(exp)
        infos.append({"experiment": exp, "exp_suffix": exp_suffix, "out_dir": out_dir, "job_desc": L.job_desc_for(exp_suffix)})
    return infos


def resubmit(info: dict) -> str | None:
    client, extra = _ensure_client()
    result = L.submit_experiment(info["experiment"], client, extra, force=True)
    name = (result or {}).get("job_name")
    log(f"  resubmitted {info['exp_suffix']} -> {name}")
    return name


def _cleanup_partial(out_dir: str) -> None:
    abs_dir = os.path.join(PROJ, out_dir)
    if os.path.isdir(abs_dir) and not has_output(out_dir):
        log(f"  removing partial run dir before retry: {out_dir}")
        shutil.rmtree(abs_dir)


def _print_log_tail(path: str, n: int = 40) -> None:
    try:
        tail = open(path).read().splitlines()[-n:]
    except OSError:
        return
    print("    --- log tail ---", flush=True)
    for ln in tail:
        print("    " + ln, flush=True)
    print("    --- end log tail ---", flush=True)


def ensure_job_success(info: dict, max_retries: int, poll: int) -> bool:
    suffix, out_dir = info["exp_suffix"], info["out_dir"]
    if has_output(out_dir):
        log(f"{suffix}: already has output, OK")
        return True

    name = discover_job(lambda d: d == info["job_desc"])
    attempt = 0
    while True:
        if name is None:
            log(f"{suffix}: no active job found, resubmitting")
            name = resubmit(info)
            if name is None:
                log(f"{suffix}: FAILED to submit")
                return False

        log(f"{suffix}: waiting on {name} (attempt {attempt + 1}/{max_retries + 1})")
        rc = mls_wait(name, poll)
        status = str(mls_status(name).get("status", "")).lower()

        # Saved trajectories are the ground-truth success signal: if they exist, accept the run
        # regardless of the wait return code (avoids a duplicate resubmit on a flaky wait).
        if has_output(out_dir):
            log(f"{suffix}: SUCCESS (status={status or 'ok'}, wait_rc={rc})")
            return True

        log(f"{suffix}: job ended badly (wait_rc={rc}, status={status}, output={has_output(out_dir)})")
        log_path = save_logs(name)
        log(f"{suffix}: logs saved to {os.path.relpath(log_path, PROJ)}")
        _print_log_tail(log_path)

        attempt += 1
        if attempt > max_retries:
            log(f"{suffix}: giving up after {attempt} attempts -- inspect the log above")
            return False

        _cleanup_partial(out_dir)
        name = None


# --------------------------------------------------------------------------- #
def run_analysis() -> bool:
    """Run analyze_trajectory_clusters.py on every 50-sample run the table reads."""
    env = _env()
    env["PYTHONPATH"] = "./src" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    all_ok = True
    for run in ANALYSIS_RUNS:
        run_dir = os.path.join("artifacts", "experiments_progressive", run)
        if not has_output(run_dir):
            log(f"  analysis: SKIP {run} (no trajectories on disk)")
            all_ok = False
            continue
        log(f"  analysis: {run}")
        cp = subprocess.run(
            [PYTHON, "scripts/analyze_trajectory_clusters.py", "--run_dir", run_dir],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJ,
        )
        print(cp.stdout[-1200:], flush=True)
        cache = os.path.join(PROJ, ANALYSIS_CACHE_DIR, run, "summary.json")
        if cp.returncode != 0 or not os.path.isfile(cache):
            log(f"  analysis FAILED for {run}: {cp.stderr.strip()[-400:]}")
            all_ok = False
    return all_ok


def regenerate_tables() -> bool:
    log("regenerating trajectory-cluster tables + figure")
    env = _env()
    env["PYTHONPATH"] = "./src:." + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cp = subprocess.run(
        [PYTHON, "scripts/paper/tables/trajectory_clusters.py", "--save", "--figure"],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJ,
    )
    print(cp.stdout[-1500:], flush=True)
    if cp.returncode != 0:
        log(f"  table generation FAILED: {cp.stderr.strip()[-600:]}")
        return False
    log("  tables + figure written")
    return True


def run_lint() -> bool:
    cp = subprocess.run([PYTHON, "paper/lint_paper.py"], capture_output=True, text=True, env=_env(), cwd=PROJ)
    print(cp.stdout, flush=True)
    ok = cp.returncode == 0
    log(f"  paper lint: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(cp.stderr, flush=True)
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--poll", type=int, default=120)
    parser.add_argument("--plan", action="store_true", help="Discover jobs and print the plan, do not wait.")
    parser.add_argument("--skip-lint", action="store_true")
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Wait for all owned jobs to finish, then stop -- do NOT run analysis / regenerate tables.",
    )
    args = parser.parse_args()
    os.makedirs(STATE_DIR, exist_ok=True)

    infos = build_infos()

    if args.plan:
        log("PLAN for trajectory_clusters_50samples (no waiting):")
        for info in infos:
            name = discover_job(lambda d, _d=info["job_desc"]: d == _d)
            st = str(mls_status(name).get("status", "?")) if name else "no-job"
            log(f"  {info['exp_suffix']}: out={has_output(info['out_dir'])} job={name} status={st}")
        return 0

    log(f"=== trajectory_clusters_50samples watcher start (max_retries={args.max_retries}, poll={args.poll}s) ===")
    results = {info["exp_suffix"]: ensure_job_success(info, args.max_retries, args.poll) for info in infos}
    summary = {"jobs": results, "finished_at": datetime.now().isoformat()}

    if not all(results.values()):
        failed = [k for k, v in results.items() if not v]
        log(f"NOT all runs succeeded: {failed}. Artifacts not regenerated. See {os.path.relpath(LOG_DIR, PROJ)}/.")
        summary["status"] = "incomplete"
        json.dump(summary, open(SUMMARY_JSON, "w"), indent=2)
        return 1

    if args.no_artifacts:
        log("All owned runs have data -> --no-artifacts set, skipping analysis/tables.")
        summary["status"] = "runs_complete"
        json.dump(summary, open(SUMMARY_JSON, "w"), indent=2)
        log(f"=== watcher done (status=runs_complete). Summary: {os.path.relpath(SUMMARY_JSON, PROJ)} ===")
        return 0

    log("All owned runs have data -> running analysis + regenerating artifacts")
    analysis_ok = run_analysis()
    summary["analysis_ok"] = analysis_ok
    table_ok = regenerate_tables() if analysis_ok else False
    summary["tables_regenerated"] = table_ok
    if table_ok and not args.skip_lint:
        summary["lint_ok"] = run_lint()
    summary["status"] = "complete" if (analysis_ok and table_ok) else "artifacts_failed"
    json.dump(summary, open(SUMMARY_JSON, "w"), indent=2)
    log(f"=== watcher done (status={summary['status']}). Summary: {os.path.relpath(SUMMARY_JSON, PROJ)} ===")
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
