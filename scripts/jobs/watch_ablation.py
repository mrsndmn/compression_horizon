"""Generic background watcher for an ablation's progressive-cramming jobs.

Parameterized by a launcher module (``--launcher``) that exposes:
  EXPERIMENTS, REFERENCE_OUT_DIR, TABLE_NAME, APPENDIX_MARKER, ROW_LABELS,
  render_job, job_desc_for, make_client, submit_experiment.

Workflow (same as the per-ablation watchers):
  1. ``mls job wait`` on each owned job; retry failures (``mls job logs`` saved +
     resubmit, up to ``--max-retries``).
  2. Wait (no retry) on the shared full-model reference run.
  3. On all-success: regenerate the table, run paper lint, and fill the
     ``% AUTOGEN <APPENDIX_MARKER>`` block in ``paper/appendix.tex`` with the
     measured per-variant compressed-token counts.

Usage:
    python scripts/jobs/watch_ablation.py --launcher run_jobs_init_ablation --plan
    python scripts/jobs/watch_ablation.py --launcher run_jobs_init_ablation --max-retries 2 --poll 120
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLS_BIN = os.environ.get("MLS_BIN", "/home/jovyan/.mlspace/envs/compression_horizon/bin/mls")
PYTHON = os.environ.get("WATCH_PYTHON", "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python")
APPENDIX = os.path.join(PROJ, "paper", "appendix.tex")

_client = None
_extra_options = None
L = None  # launcher module, set in main()
STATE_DIR = LOG_DIR = WATCH_LOG = SUMMARY_JSON = None  # set in main()


# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if WATCH_LOG:
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


def mls_list(limit: int = 1000) -> list[dict]:
    # The window must be large enough to still contain a long-running owned job after many
    # *other* users' jobs have been created since it launched. With a small window (e.g. 100) a
    # multi-hour progressive run scrolls off the recent list while still running, so discover_job
    # returns None and ensure_job_success wrongly resubmits it -> a duplicate sharing the out_dir.
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
    # Re-create the client on EVERY call so each MLS submission gets a fresh token.
    # The API access token expires after ~2h; caching it silently breaks long-running
    # watchers (submissions fail with error_code 20 "access_token expired"). The `mls`
    # CLI calls already re-auth per subprocess; only this in-process client needed the fix.
    global _client, _extra_options
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
            # Guard against resubmitting a job that actually finished while we were not looking
            # (e.g. it left the job-list window). A genuine output makes a resubmit a duplicate.
            if has_output(out_dir):
                log(f"{suffix}: no active job found but output exists, OK")
                return True
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
        name = None


def ensure_reference(poll: int) -> bool:
    out_dir = L.REFERENCE_OUT_DIR
    if has_output(out_dir):
        log("reference: already has output, OK")
        return True
    suffix = os.path.basename(out_dir)
    name = discover_job(lambda d: suffix in d and "-ablation" not in d)
    if name is None:
        log(f"reference: no job found for {suffix}; table cannot complete until it has data")
        return False
    log(f"reference: waiting on {name}")
    rc = mls_wait(name, poll)
    ok = rc == 0 and has_output(out_dir)
    log(f"reference: {'OK' if ok else 'NOT READY'} (wait_rc={rc})")
    return ok


# --------------------------------------------------------------------------- #
def regenerate_table() -> bool:
    log(f"regenerating table {L.TABLE_NAME}")
    env = _env()
    env["PYTHONPATH"] = "./src:." + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cp = subprocess.run(
        [PYTHON, "scripts/paper/tables/progressive.py", "--name", L.TABLE_NAME, "--save"],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJ,
    )
    print(cp.stdout[-1500:], flush=True)
    if cp.returncode != 0:
        log(f"  table generation FAILED: {cp.stderr.strip()[-400:]}")
        return False
    log("  table written")
    return True


def run_lint() -> bool:
    cp = subprocess.run([PYTHON, "paper/lint_paper.py"], capture_output=True, text=True, env=_env(), cwd=PROJ)
    print(cp.stdout, flush=True)
    ok = cp.returncode == 0
    log(f"  paper lint: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(cp.stderr, flush=True)
    return ok


def _table_tex_path() -> str:
    slug = L.TABLE_NAME[len("tab:") :] if L.TABLE_NAME.startswith("tab:") else L.TABLE_NAME
    return os.path.join(PROJ, "paper", "tables", slug + ".tex")


def _parse_compressed_tokens() -> dict[str, float]:
    path = _table_tex_path()
    if not os.path.isfile(path):
        return {}
    vals: dict[str, float] = {}
    num_re = re.compile(r"[-+]?\d+(?:\.\d+)?")
    for raw in open(path):
        if "&" not in raw:
            continue
        cols = raw.split("&")
        label = cols[0].strip()
        if label in L.ROW_LABELS and len(cols) > 1:
            m = num_re.search(cols[1])
            if m:
                vals[label] = float(m.group())
    return vals


def _fmt(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 0.05 else f"{x:.1f}"


def update_paper_text() -> None:
    vals = _parse_compressed_tokens()
    missing = [lbl for lbl in L.ROW_LABELS if lbl not in vals]
    if missing:
        log(f"  paper text: skipped trend sentence (missing parsed rows: {missing})")
        return

    listing = ", ".join(f"{_fmt(vals[lbl])} ({lbl.lower()})" for lbl in L.ROW_LABELS)
    sentence = f"Concretely, the mean number of perfectly reconstructed (compressed) tokens is {listing}."

    text = open(APPENDIX).read()
    marker = re.escape(L.APPENDIX_MARKER)
    pattern = re.compile(rf"(% BEGIN AUTOGEN {marker}[^\n]*\n)(.*?)(% END AUTOGEN {marker})", re.DOTALL)
    if not pattern.search(text):
        log(f"  paper text: AUTOGEN markers for {L.APPENDIX_MARKER!r} not found; leaving prose unchanged")
        return
    new_text = pattern.sub(lambda m: m.group(1) + sentence + "\n" + m.group(3), text)
    with open(APPENDIX, "w") as fh:
        fh.write(new_text)
    log("  paper text: filled trend sentence")
    log(f"    {sentence}")


# --------------------------------------------------------------------------- #
def main() -> int:
    global L, STATE_DIR, LOG_DIR, WATCH_LOG, SUMMARY_JSON
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launcher", required=True, help="Launcher module name, e.g. run_jobs_init_ablation")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--plan", action="store_true", help="Discover jobs and print the plan, do not wait.")
    parser.add_argument("--skip-paper", action="store_true")
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Wait for all eval jobs (+ reference) to finish, then stop -- do NOT regenerate the "
        "table or touch paper text. Use when the table/appendix is a deliberate later follow-up.",
    )
    args = parser.parse_args()

    L = importlib.import_module(args.launcher)
    slug = L.TABLE_NAME[len("tab:") :] if L.TABLE_NAME.startswith("tab:") else L.TABLE_NAME
    STATE_DIR = os.path.join(PROJ, ".omc", "ablation_watch", slug)
    LOG_DIR = os.path.join(STATE_DIR, "logs")
    WATCH_LOG = os.path.join(STATE_DIR, "watch.log")
    SUMMARY_JSON = os.path.join(STATE_DIR, "summary.json")
    os.makedirs(STATE_DIR, exist_ok=True)

    infos = build_infos()

    if args.plan:
        log(f"PLAN for {args.launcher} (no waiting):")
        for info in infos:
            name = discover_job(lambda d, _d=info["job_desc"]: d == _d)
            st = str(mls_status(name).get("status", "?")) if name else "no-job"
            log(f"  {info['exp_suffix']}: out={has_output(info['out_dir'])} job={name} status={st}")
        ref = L.REFERENCE_OUT_DIR
        ref_suffix = os.path.basename(ref)
        ref_name = discover_job(lambda d: ref_suffix in d and "-ablation" not in d)
        log(f"  reference {ref_suffix}: out={has_output(ref)} job={ref_name}")
        return 0

    log(f"=== {args.launcher} watcher start (max_retries={args.max_retries}, poll={args.poll}s) ===")
    results = {info["exp_suffix"]: ensure_job_success(info, args.max_retries, args.poll) for info in infos}
    ref_ok = ensure_reference(args.poll)
    summary = {"jobs": results, "reference_ok": ref_ok, "finished_at": datetime.now().isoformat()}

    if not (all(results.values()) and ref_ok):
        failed = [k for k, v in results.items() if not v] + ([] if ref_ok else ["reference"])
        log(f"NOT all runs succeeded: {failed}. Table not regenerated. See {os.path.relpath(LOG_DIR, PROJ)}/.")
        summary["status"] = "incomplete"
        json.dump(summary, open(SUMMARY_JSON, "w"), indent=2)
        return 1

    if args.no_table:
        log("All runs have data -> --no-table set, skipping table/paper (deferred follow-up)")
        summary["status"] = "eval_complete"
        json.dump(summary, open(SUMMARY_JSON, "w"), indent=2)
        log(f"=== watcher done (status=eval_complete). Summary: {os.path.relpath(SUMMARY_JSON, PROJ)} ===")
        return 0

    log("All runs have data -> regenerating table + paper text")
    table_ok = regenerate_table()
    summary["table_regenerated"] = table_ok
    if table_ok and not args.skip_paper:
        update_paper_text()
        summary["lint_ok"] = run_lint()
    summary["status"] = "complete" if table_ok else "table_failed"
    json.dump(summary, open(SUMMARY_JSON, "w"), indent=2)
    log(f"=== watcher done (status={summary['status']}). Summary: {os.path.relpath(SUMMARY_JSON, PROJ)} ===")
    return 0 if table_ok else 1


if __name__ == "__main__":
    sys.exit(main())
