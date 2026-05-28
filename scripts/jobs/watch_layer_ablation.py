"""Background watcher for the transformer-depth ablation jobs.

Waits (via ``mls job wait``) for each baseline progressive run submitted by
``run_jobs_layer_ablation.py``, retrying any that fail, then regenerates the
results table and patches the appendix trend sentence.

Workflow
--------
1. For each of the four depth-ablated checkpoints (first N + last N layers),
   discover its cluster job by description and ``mls job wait`` on it.
2. If a job fails (non-zero wait exit, or no ``progressive_prefixes`` written),
   download its logs with ``mls job logs`` to ``.omc/layer_ablation/logs/``,
   print the tail for inspection, remove the partial output dir, and resubmit
   (up to ``--max-retries`` times). Autonomous code fixes are out of scope: the
   saved log is left for manual inspection if a job keeps failing.
3. Wait (without retrying) on the full-depth 24-layer reference run, which the
   table also needs.
4. Once all five runs have data: regenerate ``tab:layer_ablation``, run the
   paper lint, and fill the ``% AUTOGEN layer-ablation-trend`` block in
   ``paper/appendix.tex`` with the measured compressed-token trend.

Usage
-----
    # discovery + plan only, no blocking (sanity check):
    python scripts/jobs/watch_layer_ablation.py --plan

    # full run (long-blocking; launch in the background):
    python scripts/jobs/watch_layer_ablation.py --max-retries 2 --poll 60
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jobs_layer_ablation as launcher  # noqa: E402

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLS_BIN = os.environ.get("MLS_BIN", "/home/jovyan/.mlspace/envs/compression_horizon/bin/mls")
PYTHON = os.environ.get("WATCH_PYTHON", "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python")

STATE_DIR = os.path.join(PROJ, ".omc", "layer_ablation")
LOG_DIR = os.path.join(STATE_DIR, "logs")
WATCH_LOG = os.path.join(STATE_DIR, "watch.log")
SUMMARY_JSON = os.path.join(STATE_DIR, "summary.json")

APPENDIX = os.path.join(PROJ, "paper", "appendix.tex")
TABLE_TEX = os.path.join(PROJ, "paper", "tables", "layer_ablation.tex")
TABLE_NAME = "tab:layer_ablation"

# Statuses (lower-cased) that mean the job finished well / badly.
SUCCESS_STATUSES = {"succeeded", "completed"}

# Row labels of the generated table, in depth order, with the layer count each
# represents (used to build the trend sentence).
ROW_ORDER = [
    ("2 layers", 2),
    ("4 layers", 4),
    ("8 layers", 8),
    ("16 layers", 16),
    ("24 layers (full)", 24),
]

_client = None
_extra_options = None


# --------------------------------------------------------------------------- #
# small helpers
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
    return subprocess.run(
        [MLS_BIN, "job", *args],
        capture_output=capture,
        text=True,
        env=_env(),
        cwd=PROJ,
    )


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
    """Block until the job reaches a terminal state. Returns its exit code."""
    cp = _mls(["wait", name, "-i", str(poll)], capture=False)
    return cp.returncode


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
    """Return the best matching job_name: prefer active jobs, else the newest."""
    matches = [j for j in mls_list() if predicate(str(j.get("job_desc", "")))]
    if not matches:
        return None
    active = [j for j in matches if str(j.get("status", "")).lower() in ("running", "pending", "completing")]
    pick = (active or matches)[0]  # list is newest-first
    return pick.get("job_name")


def _ensure_client():
    global _client, _extra_options
    if _client is None:
        _client, _extra_options = launcher.make_client()
    return _client, _extra_options


# --------------------------------------------------------------------------- #
# per-job wait + retry
# --------------------------------------------------------------------------- #
def build_infos() -> list[dict]:
    infos = []
    for exp in launcher.LAYER_ABLATION_EXPERIMENTS:
        _, exp_suffix, out_dir = launcher.render_job(exp)
        infos.append(
            {
                "experiment": exp,
                "exp_suffix": exp_suffix,
                "out_dir": out_dir,
                "job_desc": launcher.job_desc_for(exp_suffix),
            }
        )
    return infos


def resubmit(info: dict) -> str | None:
    client, extra = _ensure_client()
    result = launcher.submit_experiment(info["experiment"], client, extra, force=True)
    name = (result or {}).get("job_name")
    log(f"  resubmitted {info['exp_suffix']} -> {name}")
    return name


def ensure_job_success(info: dict, max_retries: int, poll: int) -> bool:
    suffix, out_dir = info["exp_suffix"], info["out_dir"]
    if has_output(out_dir):
        log(f"{suffix}: already has output, OK")
        return True

    name = discover_job(lambda d: d == info["job_desc"])
    attempt = 0
    while True:
        if name is None:
            log(f"{suffix}: no active job found, submitting")
            name = resubmit(info)
            if name is None:
                log(f"{suffix}: FAILED to submit")
                return False

        log(f"{suffix}: waiting on {name} (attempt {attempt + 1}/{max_retries + 1})")
        rc = mls_wait(name, poll)
        status = str(mls_status(name).get("status", "")).lower()

        if rc == 0 and has_output(out_dir):
            log(f"{suffix}: SUCCESS (status={status or 'ok'})")
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
        name = None  # force a fresh submit on the next loop


def _cleanup_partial(out_dir: str) -> None:
    abs_dir = os.path.join(PROJ, out_dir)
    if os.path.isdir(abs_dir) and not has_output(out_dir):
        log(f"  removing partial run dir before retry: {out_dir}")
        shutil.rmtree(abs_dir)


def _print_log_tail(path: str, n: int = 40) -> None:
    try:
        with open(path) as fh:
            tail = fh.read().splitlines()[-n:]
    except OSError:
        return
    print("    --- log tail ---", flush=True)
    for ln in tail:
        print("    " + ln, flush=True)
    print("    --- end log tail ---", flush=True)


def ensure_reference(poll: int) -> bool:
    """Wait (no retry) on the full-depth reference run that the table needs."""
    out_dir = launcher.REFERENCE_OUT_DIR
    if has_output(out_dir):
        log("reference: already has output, OK")
        return True
    suffix = os.path.basename(out_dir)
    name = discover_job(lambda d: suffix in d and "firstlast" not in d)
    if name is None:
        log(f"reference: no job found for {suffix}; cannot complete the table until it has data")
        return False
    log(f"reference: waiting on {name}")
    rc = mls_wait(name, poll)
    ok = rc == 0 and has_output(out_dir)
    log(f"reference: {'OK' if ok else 'NOT READY'} (wait_rc={rc})")
    return ok


# --------------------------------------------------------------------------- #
# table + paper text
# --------------------------------------------------------------------------- #
def regenerate_table() -> bool:
    log("regenerating table tab:layer_ablation")
    env = _env()
    env["PYTHONPATH"] = "./src:." + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cp = subprocess.run(
        [PYTHON, "scripts/paper/tables/progressive.py", "--name", TABLE_NAME, "--save"],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJ,
    )
    print(cp.stdout[-1500:], flush=True)
    if cp.returncode != 0:
        log(f"  table generation FAILED: {cp.stderr.strip()[-400:]}")
        return False
    log(f"  table written to {os.path.relpath(TABLE_TEX, PROJ)}")
    return True


def run_lint() -> bool:
    cp = subprocess.run([PYTHON, "paper/lint_paper.py"], capture_output=True, text=True, env=_env(), cwd=PROJ)
    print(cp.stdout, flush=True)
    ok = cp.returncode == 0
    log(f"  paper lint: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(cp.stderr, flush=True)
    return ok


def _parse_compressed_tokens() -> dict[str, float]:
    """Map each row label to its Compressed Tokens mean from the rendered table."""
    if not os.path.isfile(TABLE_TEX):
        return {}
    vals: dict[str, float] = {}
    num_re = re.compile(r"[-+]?\d+(?:\.\d+)?")
    for raw in open(TABLE_TEX):
        if "&" not in raw:
            continue
        cols = raw.split("&")
        label = cols[0].strip()
        for known, _ in ROW_ORDER:
            if label == known and len(cols) > 1:
                m = num_re.search(cols[1])
                if m:
                    vals[known] = float(m.group())
    return vals


def _fmt(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 0.05 else f"{x:.1f}"


def update_paper_text() -> None:
    vals = _parse_compressed_tokens()
    missing = [lbl for lbl, _ in ROW_ORDER if lbl not in vals]
    if missing:
        log(f"  paper text: skipped trend sentence (missing parsed rows: {missing})")
        return

    ordered = [vals[lbl] for lbl, _ in ROW_ORDER]
    nums = [_fmt(v) for v in ordered]
    listing = ", ".join(nums[:-1]) + f", and {nums[-1]}"
    monotonic = all(b >= a for a, b in zip(ordered, ordered[1:]))
    trend = " increasing monotonically with depth" if monotonic else " varying non-monotonically with depth"
    sentence = (
        f"Concretely, the mean number of perfectly reconstructed (compressed) tokens is "
        f"{listing} for the 2-, 4-, 8-, 16-, and 24-layer variants, respectively,{trend}."
    )

    text = open(APPENDIX).read()
    pattern = re.compile(
        r"(% BEGIN AUTOGEN layer-ablation-trend[^\n]*\n)(.*?)(% END AUTOGEN layer-ablation-trend)",
        re.DOTALL,
    )
    if not pattern.search(text):
        log("  paper text: AUTOGEN markers not found in appendix.tex; leaving prose unchanged")
        return
    new_text = pattern.sub(lambda m: m.group(1) + sentence + "\n" + m.group(3), text)
    with open(APPENDIX, "w") as fh:
        fh.write(new_text)
    log(f"  paper text: filled trend sentence ({'monotonic' if monotonic else 'non-monotonic'})")
    log(f"    {sentence}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-retries", type=int, default=2, help="Resubmit a failed job up to this many times.")
    parser.add_argument("--poll", type=int, default=60, help="mls job wait poll interval (seconds).")
    parser.add_argument("--plan", action="store_true", help="Discover jobs and print the plan, do not wait.")
    parser.add_argument("--skip-paper", action="store_true", help="Regenerate the table but do not touch the paper.")
    args = parser.parse_args()

    os.makedirs(STATE_DIR, exist_ok=True)
    infos = build_infos()

    if args.plan:
        log("PLAN (no waiting):")
        for info in infos:
            name = discover_job(lambda d, _d=info["job_desc"]: d == _d)
            st = str(mls_status(name).get("status", "?")) if name else "no-job"
            log(f"  {info['exp_suffix']}: out={has_output(info['out_dir'])} job={name} status={st}")
        ref = launcher.REFERENCE_OUT_DIR
        ref_suffix = os.path.basename(ref)
        ref_name = discover_job(lambda d: ref_suffix in d and "firstlast" not in d)
        log(f"  reference {ref_suffix}: out={has_output(ref)} job={ref_name}")
        return 0

    log(f"=== layer-ablation watcher start (max_retries={args.max_retries}, poll={args.poll}s) ===")
    results = {info["exp_suffix"]: ensure_job_success(info, args.max_retries, args.poll) for info in infos}
    ref_ok = ensure_reference(args.poll)

    summary = {"jobs": results, "reference_ok": ref_ok, "finished_at": datetime.now().isoformat()}

    all_ok = all(results.values()) and ref_ok
    if not all_ok:
        failed = [k for k, v in results.items() if not v] + ([] if ref_ok else ["reference"])
        log(f"NOT all runs succeeded: {failed}. Table not regenerated. See {os.path.relpath(LOG_DIR, PROJ)}/.")
        summary["status"] = "incomplete"
        with open(SUMMARY_JSON, "w") as fh:
            json.dump(summary, fh, indent=2)
        return 1

    log("All runs have data -> regenerating table + paper text")
    table_ok = regenerate_table()
    summary["table_regenerated"] = table_ok
    if table_ok and not args.skip_paper:
        update_paper_text()
        summary["lint_ok"] = run_lint()
    summary["status"] = "complete" if table_ok else "table_failed"
    with open(SUMMARY_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)

    log(f"=== watcher done (status={summary['status']}). Summary: {os.path.relpath(SUMMARY_JSON, PROJ)} ===")
    return 0 if table_ok else 1


if __name__ == "__main__":
    sys.exit(main())
