"""Watch densify field-shard jobs and auto-merge each sample's shards into its dense NPZ.

``run_jobs_densify_fields.py`` fans accuracy-field computation across one-anchor-per-job
``a100.1gpu`` jobs, one ``field_shards/idx_XXXX/landscape_pca_pairs.npz`` per anchor. This
watcher polls each sample and, as soon as all its shards have landed (or no jobs remain),
merges them into ``<viz>/dense/landscape_pca_pairs.npz`` via ``merge_landscape_shards.py``
(``--no-base``: the shards themselves carry coords/mesh; the CPU seed NPZ has no accuracy
field). That dense NPZ is what ``animate_trajectory.py`` / ``visual_abstract.py`` consume.

Completion signal per sample:
  * READY   -- every expected shard NPZ (from the launcher manifest, else a glob) exists.
  * STALLED -- some shards exist but no ``s{N} idx_`` job is in progress anymore (the rest
               failed / were killed); merge what landed and warn about the gap.

Safe to re-run: a sample whose dense NPZ already exists is treated as done (unless --force).

Usage::

    PYTHONPATH=./src python scripts/jobs/watch_densify_merge.py --samples 1 2 3 4 5 6 7 8 9
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

DEFAULT_BASE = "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1"
MERGE_SCRIPT = "scripts/paper/merge_landscape_shards.py"


def _viz_dir(base_dir: str, sample: int) -> str:
    return os.path.join(base_dir, f"visualizations_s{sample}")


def _expected_shards(viz_dir: str) -> list:
    """Full set of shard NPZ paths for a sample: the launcher manifest if present, else a glob
    of already-created idx_* dirs (a lower bound while jobs are still being created)."""
    man = os.path.join(viz_dir, "field_shards", "manifest.json")
    if os.path.exists(man):
        try:
            m = json.load(open(man))
            shards = [s["npz"] for s in m.get("shards", [])]
            if shards:
                return shards
        except (json.JSONDecodeError, KeyError):
            pass
    return sorted(glob.glob(os.path.join(viz_dir, "field_shards", "idx_*", "landscape_pca_pairs.npz")))


def _in_progress_descs() -> list:
    from mls.manager.job.utils import get_in_progress_jobs

    return [j.get("job_desc", "") for j in get_in_progress_jobs()]


def _merge(viz_dir: str, merge_script: str, python: str) -> tuple:
    out = os.path.join(viz_dir, "dense", "landscape_pca_pairs.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    inputs = os.path.join(viz_dir, "field_shards", "idx_*", "landscape_pca_pairs.npz")
    cmd = [python, merge_script, "--no-base", "--inputs", inputs, "--output", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, out, (r.stdout + r.stderr).strip()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", dest="base_dir", type=str, default=DEFAULT_BASE)
    ap.add_argument("--samples", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--interval", type=float, default=45.0, help="Seconds between polls.")
    ap.add_argument("--max-wait", dest="max_wait", type=float, default=120.0, help="Give up after this many minutes.")
    ap.add_argument("--merge-script", dest="merge_script", type=str, default=MERGE_SCRIPT)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--force", action="store_true", help="Re-merge even if a sample's dense NPZ already exists.")
    args = ap.parse_args()

    samples = list(args.samples)
    done: set = set()
    deadline = time.time() + args.max_wait * 60.0
    print(f"[{_ts()}] watching samples {samples} under {args.base_dir} (interval={args.interval}s, max={args.max_wait}m)")

    while True:
        try:
            descs = _in_progress_descs()
        except Exception as e:  # transient API hiccup -- keep polling
            print(f"[{_ts()}] WARN: could not query jobs ({e}); retrying next poll")
            descs = None

        for s in samples:
            if s in done:
                continue
            vd = _viz_dir(args.base_dir, s)
            dense = os.path.join(vd, "dense", "landscape_pca_pairs.npz")
            if os.path.exists(dense) and not args.force:
                print(f"[{_ts()}] s{s}: dense NPZ already present -> done")
                done.add(s)
                continue

            expected = _expected_shards(vd)
            present = [p for p in expected if os.path.exists(p)]
            n_exp, n_pre = len(expected), len(present)
            running = None if descs is None else sum(1 for d in descs if f" s{s} idx_" in d)

            ready = n_exp > 0 and n_pre == n_exp
            stalled = n_pre > 0 and running == 0 and not ready

            if ready or stalled:
                tag = "READY" if ready else f"STALLED ({n_pre}/{n_exp}, no jobs left)"
                ok, out, log = _merge(vd, args.merge_script, args.python)
                last = log.splitlines()[-1] if log else ""
                if ok:
                    print(f"[{_ts()}] s{s}: {tag} -> merged. {last}")
                    done.add(s)
                else:
                    print(f"[{_ts()}] s{s}: {tag} -> MERGE FAILED (will retry): {last}")
            else:
                run_str = "?" if running is None else str(running)
                print(f"[{_ts()}] s{s}: {n_pre}/{n_exp} shards, {run_str} jobs running")

        if len(done) == len(samples):
            print(f"[{_ts()}] all {len(samples)} samples merged. Done.")
            break
        if time.time() > deadline:
            missing = [s for s in samples if s not in done]
            print(f"[{_ts()}] max-wait reached; unmerged samples: {missing}")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
