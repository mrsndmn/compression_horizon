"""Watcher/finalizer for the pythia-1.4b CE-temperature sweep.

Polls the 9 ``CE_TEMPERATURE_EXPERIMENTS`` output dirs and, once every run's
``progressive_prefixes`` artifacts have landed and render cleanly (no nan),
renders ``tab:progressive_temperature`` to ``paper/tables/progressive_temperature.tex``,
registers it in ``paper/tables/tables.sh`` (idempotent), lint-checks, and commits.
The appendix already \\input's the table behind an ``\\IfFileExists`` guard, so the
table appears in the paper build automatically once the .tex is committed.

Self-correcting: a dir that exists but is only partially written renders with nan
(the paper lint would reject it), so the watcher just waits and retries until every
row is clean. Idempotent: if the .tex already exists and is clean it finalizes and
exits.

Usage:
    python scripts/jobs/watch_ce_temperature.py                 # poll + finalize
    python scripts/jobs/watch_ce_temperature.py --once          # single attempt
    python scripts/jobs/watch_ce_temperature.py --interval 600  # seconds between polls
    python scripts/jobs/watch_ce_temperature.py --no-commit     # render but do not git commit
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jobs_progressive as rjp  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLE_NAME = "tab:progressive_temperature"
TABLE_TEX = os.path.join(REPO, "paper", "tables", "progressive_temperature.tex")
TABLES_SH = os.path.join(REPO, "scripts", "paper", "tables", "tables.sh")


def out_dirs():
    dirs = []
    for experiment in rjp.CE_TEMPERATURE_EXPERIMENTS:
        _, _, out_dir_name = rjp.render_job(experiment)
        dirs.append(os.path.join(REPO, out_dir_name, "progressive_prefixes"))
    return dirs


def readiness():
    dirs = out_dirs()
    ready = [d for d in dirs if os.path.isdir(d)]
    return ready, dirs


def render_table() -> tuple[bool, str]:
    """Render the table .tex. Returns (ok, message). ok=False if a dir is missing or nan appears."""
    cmd = [
        sys.executable,
        os.path.join(REPO, "scripts", "paper", "tables", "progressive.py"),
        "--name",
        TABLE_NAME,
        "--save",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO, "src") + ":" + REPO + ":" + env.get("PYTHONPATH", "")
    cp = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    if cp.returncode != 0:
        tail = (cp.stderr.strip() or cp.stdout.strip()).splitlines()[-1:] or [""]
        return False, f"render failed: {tail[0]}"
    if not os.path.isfile(TABLE_TEX):
        return False, "render reported success but .tex missing"
    text = open(TABLE_TEX, encoding="utf-8").read()
    # A partially-written run yields nan cells; the paper lint forbids them, so wait.
    for lineno, line in enumerate(text.splitlines(), 1):
        if "&" in line and __import__("re").search(r"(?<![A-Za-z])nan(?![A-Za-z])", line, __import__("re").I):
            return False, f"table has nan on line {lineno} (a run is still partial)"
    return True, "rendered clean"


def register_in_tables_sh():
    """Ensure the table label is an ACTIVE entry in tables.sh (idempotent).

    Handles the commented ``# tab:progressive_temperature`` placeholder that is committed while the
    sweep runs: it uncomments that line rather than being fooled into a no-op by a substring match.
    """
    if not os.path.isfile(TABLES_SH):
        return
    lines = open(TABLES_SH, encoding="utf-8").read().splitlines()
    active = f"  {TABLE_NAME}"
    if any(ln.strip() == TABLE_NAME for ln in lines):  # already active, uncommented
        return
    out, replaced = [], False
    for ln in lines:
        if not replaced and ln.strip().startswith("#") and TABLE_NAME in ln:
            out.append(active)  # uncomment the placeholder
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        anchor = "tab:progressive_no_bos_token"
        tmp, inserted = [], False
        for ln in out:
            tmp.append(ln)
            if not inserted and anchor in ln and not ln.strip().startswith("#"):
                tmp.append(active)
                inserted = True
        out = tmp if inserted else out + [active]
    open(TABLES_SH, "w", encoding="utf-8").write("\n".join(out) + "\n")


def lint_ok() -> bool:
    cp = subprocess.run(
        [sys.executable, os.path.join(REPO, "paper", "lint_paper.py")], cwd=REPO, capture_output=True, text=True
    )
    if cp.returncode != 0:
        print(cp.stdout[-2000:])
        print(cp.stderr[-1000:])
    return cp.returncode == 0


def commit():
    subprocess.run(["git", "add", TABLE_TEX, TABLES_SH], cwd=REPO, check=False)
    msg = (
        "results: populate tab:progressive_temperature from pythia-1.4b runs\n\n"
        "Render the CE-temperature ablation table from the 9 completed pythia-1.4b\n"
        "progressive runs (T in {0.1,0.5,1.0,1.5,2.0} x {raw,t2}); register it in\n"
        "tables.sh. The appendix already \\input's it behind an \\IfFileExists guard.\n\n"
        "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
    )
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO, check=False)


def finalize(do_commit: bool) -> bool:
    ok, msg = render_table()
    print(f"  render: {msg}")
    if not ok:
        return False
    register_in_tables_sh()
    if not lint_ok():
        print("  lint failed after render -- not committing")
        return False
    print("  lint OK")
    if do_commit:
        commit()
        print("  committed tab:progressive_temperature")
    else:
        print("  --no-commit: left rendered .tex uncommitted")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=600, help="Seconds between polls (default 600).")
    parser.add_argument("--once", action="store_true", help="Single attempt, then exit.")
    parser.add_argument("--no-commit", action="store_true", help="Render + lint but do not git commit.")
    args = parser.parse_args()

    while True:
        ready, dirs = readiness()
        print(f"[watch] {len(ready)}/{len(dirs)} run dirs present", flush=True)
        if len(ready) == len(dirs):
            if finalize(do_commit=not args.no_commit):
                print("[watch] done.")
                return 0
        if args.once:
            print("[watch] --once: not all runs ready (or table not clean yet); exiting.")
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
