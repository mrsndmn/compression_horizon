"""Watcher/finalizer for tab:temperature_logit_stats (reconstruction logit geometry vs. CE temperature).

Polls the CE-temperature run dirs behind the table and, once every run's
``progressive_prefixes`` artifacts have landed, recomputes the GPU cache
(``artifacts/paper/temperature_logit_stats.json``), renders
``paper/tables/temperature_logit_stats.tex``, lint-checks, and commits.

Differs from ``watch_ce_temperature.py`` in one important way: this table is
CACHE-BACKED and the cache is **gitignored**, so the watcher must run the
``--compute`` pass itself. That pass needs a GPU + both base models
(pythia-1.4b, Llama-3.1-8B) and takes a few minutes, so it is only run when the
cache is actually stale -- see ``cache_state()``. The rendered ``.tex`` is the
only tracked artifact, and it is already registered in ``tables.sh``.

Staleness rule: recompute when the cache is missing, is missing a run that now
has artifacts, lacks the normalized ``top1_logp_norm`` field (pre-normalization
caches), or is older than the newest run artifact.

Self-correcting: a dir that exists but is only partially written yields a cache
entry over fewer samples or a nan cell (the paper lint rejects nan), so the
watcher waits and retries until every row is clean.

Usage:
    python scripts/jobs/watch_temperature_logit_stats.py                 # poll + finalize
    python scripts/jobs/watch_temperature_logit_stats.py --once          # single attempt
    python scripts/jobs/watch_temperature_logit_stats.py --interval 600  # seconds between polls
    python scripts/jobs/watch_temperature_logit_stats.py --recompute     # force the GPU pass
    python scripts/jobs/watch_temperature_logit_stats.py --no-commit     # render but do not git commit
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(REPO, "scripts", "paper", "tables", "temperature_logit_stats.py")
CACHE = os.path.join(REPO, "artifacts", "paper", "temperature_logit_stats.json")
TABLE_TEX = os.path.join(REPO, "paper", "tables", "temperature_logit_stats.tex")


def _load_generator():
    """Import the table generator so the run list stays single-sourced there."""
    spec = importlib.util.spec_from_file_location("temperature_logit_stats_gen", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN_MOD = _load_generator()


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO, "src") + ":" + REPO + ":" + env.get("PYTHONPATH", "")
    return env


def expected_runs():
    """[(label, abs_dir)] for every run the table wants, in table order."""
    return [(row[0], os.path.join(REPO, row[-1])) for row in GEN_MOD.run_rows()]


def readiness():
    runs = expected_runs()
    ready = [(lbl, d) for lbl, d in runs if os.path.isdir(d)]
    return ready, runs


def _newest_artifact_mtime(dirs) -> float:
    """Newest mtime across the run dirs (dir mtime is enough: files land inside it)."""
    mtimes = [os.path.getmtime(d) for d in dirs if os.path.isdir(d)]
    return max(mtimes) if mtimes else 0.0


def cache_state(ready) -> tuple[bool, str]:
    """(needs_compute, reason). Keeps the expensive GPU pass off the happy path."""
    if not os.path.isfile(CACHE):
        return True, "cache missing"
    try:
        cache = json.load(open(CACHE, encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return True, f"cache unreadable ({exc.__class__.__name__})"
    missing = [lbl for lbl, _ in ready if lbl not in cache]
    if missing:
        return True, f"cache missing {len(missing)} run(s), e.g. {missing[0]!r}"
    # Pre-normalization caches lack the normalized top-1 column.
    stale_field = [lbl for lbl, _ in ready if "top1_logp_norm" not in cache.get(lbl, {})]
    if stale_field:
        return True, f"{len(stale_field)} run(s) lack top1_logp_norm (pre-normalization cache)"
    newest = _newest_artifact_mtime([d for _, d in ready])
    if newest > os.path.getmtime(CACHE):
        return True, "cache older than newest run artifact"
    return False, "cache complete and current"


def run_compute() -> tuple[bool, str]:
    """GPU pass: recompute the cache. Needs CUDA + both base models."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "no CUDA device visible -- compute needs a GPU"
    except ImportError:
        return False, "torch not importable -- compute needs a GPU"
    cp = subprocess.run([sys.executable, GEN, "--compute"], cwd=REPO, capture_output=True, text=True, env=_env())
    if cp.returncode != 0:
        tail = (cp.stderr.strip() or cp.stdout.strip()).splitlines()[-1:] or [""]
        return False, f"compute failed: {tail[0]}"
    return True, "cache recomputed"


def render_table() -> tuple[bool, str]:
    """Render the .tex from the cache. Returns (ok, message)."""
    cp = subprocess.run([sys.executable, GEN, "--save"], cwd=REPO, capture_output=True, text=True, env=_env())
    if cp.returncode != 0:
        tail = (cp.stderr.strip() or cp.stdout.strip()).splitlines()[-1:] or [""]
        return False, f"render failed: {tail[0]}"
    if not os.path.isfile(TABLE_TEX):
        return False, "render reported success but .tex missing"
    text = open(TABLE_TEX, encoding="utf-8").read()
    # A partially-written run yields nan cells; the paper lint forbids them, so wait.
    for lineno, line in enumerate(text.splitlines(), 1):
        if "&" in line and re.search(r"(?<![A-Za-z])nan(?![A-Za-z])", line, re.I):
            return False, f"table has nan on line {lineno} (a run is still partial)"
    # Both top-1 forms must survive a regeneration.
    if r"Top-1 $\mathbf{z}/T$" not in text or r"Top-1 $\log p$" not in text:
        return False, "header lost the unnormalized/normalized Top-1 column pair"
    n_rows = sum(1 for ln in text.splitlines() if ln.strip().endswith(r"\\") and "$T" in ln)
    expected = len(expected_runs())
    if n_rows != expected:
        return False, f"rendered {n_rows} data rows, expected {expected}"
    return True, f"rendered clean ({n_rows} rows)"


def lint_ok() -> bool:
    cp = subprocess.run(
        [sys.executable, os.path.join(REPO, "paper", "lint_paper.py")], cwd=REPO, capture_output=True, text=True
    )
    if cp.returncode != 0:
        print(cp.stdout[-2000:])
        print(cp.stderr[-1000:])
    return cp.returncode == 0


def commit():
    """Commit the rendered .tex only -- the cache under artifacts/ is gitignored."""
    cp = subprocess.run(["git", "status", "--porcelain", TABLE_TEX], cwd=REPO, capture_output=True, text=True)
    if not cp.stdout.strip():
        print("  no change to tab:temperature_logit_stats -- nothing to commit")
        return
    subprocess.run(["git", "add", TABLE_TEX], cwd=REPO, check=False)
    msg = (
        "results: refresh tab:temperature_logit_stats from CE-temperature runs\n\n"
        "Recompute the reconstruction logit-geometry cache from the completed\n"
        "pythia-1.4b / Llama-3.1-8B temperature runs and re-render the table\n"
        "(margin z/T plus the unnormalized and normalized Top-1 columns).\n\n"
        "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
    )
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO, check=False)


def finalize(ready, do_commit: bool, force_compute: bool) -> bool:
    needs, reason = cache_state(ready)
    if force_compute:
        needs, reason = True, "--recompute requested"
    print(f"  cache: {reason}")
    if needs:
        ok, msg = run_compute()
        print(f"  compute: {msg}")
        if not ok:
            return False
    ok, msg = render_table()
    print(f"  render: {msg}")
    if not ok:
        return False
    if not lint_ok():
        print("  lint failed after render -- not committing")
        return False
    print("  lint OK")
    if do_commit:
        commit()
    else:
        print("  --no-commit: left rendered .tex uncommitted")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=600, help="Seconds between polls (default 600).")
    parser.add_argument("--once", action="store_true", help="Single attempt, then exit.")
    parser.add_argument("--recompute", action="store_true", help="Force the GPU compute pass.")
    parser.add_argument("--no-commit", action="store_true", help="Render + lint but do not git commit.")
    args = parser.parse_args()

    force_compute = args.recompute
    while True:
        ready, runs = readiness()
        print(f"[watch] {len(ready)}/{len(runs)} run dirs present", flush=True)
        if len(ready) == len(runs):
            if finalize(ready, do_commit=not args.no_commit, force_compute=force_compute):
                print("[watch] done.")
                return 0
            force_compute = False  # a forced pass is spent; later polls use the staleness rule
        if args.once:
            print("[watch] --once: not all runs ready (or table not clean yet); exiting.")
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
