"""Build the step-based jump caches for the merged trajectory-shape tables.

Companion to the Euclidean-jump analysis (``analyze_trajectory_clusters.py``): that one measures each
jump between consecutive converged embeddings as a Euclidean *distance*; here we redefine the jump as
the **number of optimization steps** spent to absorb that token -- the marginal per-token step count,
a model-agnostic, effort-based unit -- and run the identical jump-shuffle dwelling test on this scalar
series. ``scripts/paper/tables/trajectory_clusters.py`` then renders the two metric families
side-by-side into ``tab:trajectory_cluster_structure{,_lr}``.

For each per-sample trajectory we order the converged stages by ``stage_index``, derive the marginal
step series (auto-detecting whether ``steps_taken`` is logged cumulatively or per-stage), and cache,
pooled over samples: mean ``Stages`` (= jumps + 1), the step gap-ratio (``max/median`` marginal step
count), the lag-1 autocorrelation of the step series with its jump-shuffle null (~0), and the
``Dwelling %`` (fraction of samples whose autocorrelation beats their own jump-shuffle 95th percentile).

The compute pass reads only the scalar ``steps_taken`` column (no embeddings, no GPU) and caches to
``artifacts/paper/trajectory_steps/<run>.json``. Run it once per run; the table render then reads the
caches::

    PYTHONPATH=./src:. python scripts/paper/tables/trajectory_steps.py --compute
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_from_disk

_EXP = "artifacts/experiments_progressive"
_CACHE_DIR = "artifacts/paper/trajectory_steps"

MIN_STAGES = 5  # matches scripts/analyze_trajectory_clusters.py
N_SHUFFLE = 200
SEED = 0
NULL_PCTILE = 95.0

# Model-scale trend (all lr=0.1), and the SmolLM2-1.7B learning-rate sweep -- the same runs as
# tab:trajectory_cluster_structure{,_lr}, so the two analyses are apples-to-apples.
SCALE_ROWS: List[Tuple[str, str]] = [
    ("SmolLM2-135M", "sl_4096_SmolLM2-135M_ds_pg19_1k_limit_50_lr_0.1"),
    ("SmolLM2-360M", "sl_4096_SmolLM2-360M_ds_pg19_1k_limit_50_lr_0.1"),
    ("SmolLM2-1.7B", "sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1"),
    ("Llama-3.1-8B", "sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1"),
]
LR_ROWS: List[Tuple[str, str]] = [
    ("SmolLM2-1.7B {\\small lr=0.1}", "sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1"),
    ("SmolLM2-1.7B {\\small lr=0.5}", "sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.5"),
    ("SmolLM2-1.7B {\\small lr=1.0}", "sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_1.0"),
]


def _autocorr(d: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D series (the dwelling signal)."""
    if d.size < 3:
        return float("nan")
    a, b = d[:-1], d[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _marginal_step_series(run_dir: str) -> Tuple[List[np.ndarray], List[int], bool]:
    """Per-sample marginal step series: the optimization steps spent on each jump.

    Reads only the scalar ``steps_taken`` column (memory-trivial -- never decodes the large
    per-row embedding columns). ``steps_taken`` is logged either cumulatively (non-decreasing
    within a sample) or per-stage; we auto-detect at the run level and difference the cumulative
    case. The jump from converged point k-1 to k is the effort of stage k, so both conventions
    yield n-1 values per n-stage trajectory (the cumulative diff and the per-stage tail both drop
    the initial point's formation cost). Returns ``(series, n_stages, cumulative)``.
    """
    ds = load_from_disk(run_dir).select_columns(["sample_id", "stage_index", "steps_taken"]).with_format("numpy")
    sid = np.asarray(ds["sample_id"]).reshape(-1)
    stg = np.asarray(ds["stage_index"]).reshape(-1)
    stp = np.asarray(ds["steps_taken"]).reshape(-1).astype(float)

    by_sid: dict[int, np.ndarray] = {}
    dec = tot = 0
    for s in np.unique(sid):
        m = sid == s
        v = stp[m][np.argsort(stg[m], kind="stable")]
        by_sid[int(s)] = v
        d = np.diff(v)
        dec += int((d < 0).sum())
        tot += d.size
    cumulative = (dec / tot if tot else 0.0) < 0.02  # cumulative => essentially never decreases

    series, n_stages = [], []
    for v in by_sid.values():
        if v.size < MIN_STAGES:
            continue
        d = (np.diff(v) if cumulative else v[1:]).astype(float)
        if d.size < 4:
            continue
        series.append(d)
        n_stages.append(int(v.size))
    return series, n_stages, cumulative


def compute_run(run_dir: str) -> dict:
    """Pooled step-jump dwelling statistics for one run."""
    series, n_stages, cumulative = _marginal_step_series(run_dir)
    rng = np.random.default_rng(SEED)
    gaps, r1s, shufs, dwell = [], [], [], []
    for d in series:
        med = float(np.median(d))
        gaps.append(d.max() / med if med > 0 else np.nan)
        r = _autocorr(d)
        r1s.append(r)
        sp = np.array([_autocorr(d[rng.permutation(d.size)]) for _ in range(N_SHUFFLE)])
        shufs.append(float(np.nanmean(sp)))
        dwell.append(bool(np.isfinite(r) and r > float(np.nanpercentile(sp, NULL_PCTILE))))
    n = len(r1s)
    return {
        "n_samples": n,
        "steps_convention": "cumulative" if cumulative else "per_stage",
        "mean_stages": float(np.mean(n_stages)) if n_stages else float("nan"),
        "gap_ratio_mean": float(np.nanmean(gaps)) if n else float("nan"),
        "autocorr_real_mean": float(np.nanmean(r1s)) if n else float("nan"),
        "autocorr_shuffle_mean": float(np.nanmean(shufs)) if n else float("nan"),
        "fraction_dwelling": float(np.mean(dwell)) if n else float("nan"),
    }


def cache_path(run_name: str) -> str:
    return os.path.join(_CACHE_DIR, f"{run_name}.json")


def compute_and_cache(run_name: str, force: bool = False) -> dict:
    path = cache_path(run_name)
    if os.path.exists(path) and not force:
        print(f"  cache exists, skipping compute: {path}")
        with open(path) as f:
            return json.load(f)
    run_dir = f"{_EXP}/{run_name}/progressive_prefixes"
    if not os.path.isdir(run_dir):
        raise SystemExit(f"missing run dir: {run_dir}")
    agg = compute_run(run_dir)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(agg, f, indent=2)
    print(
        f"  saved cache: {path}  (N={agg['n_samples']}, {agg['steps_convention']}, "
        f"r1={agg['autocorr_real_mean']:+.2f}, dwell={agg['fraction_dwelling'] * 100:.0f}%)"
    )
    return agg


def load_cache(run_name: str) -> Optional[dict]:
    path = cache_path(run_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compute", action="store_true", help="Read the run datasets and (re)write per-run caches.")
    ap.add_argument("--force", action="store_true", help="Recompute even if a cache exists (requires --compute).")
    args = ap.parse_args()
    if args.force and not args.compute:
        ap.error("--force requires --compute")

    runs: List[str] = []
    for _, run in SCALE_ROWS + LR_ROWS:
        if run not in runs:
            runs.append(run)
    for run in runs:
        agg = compute_and_cache(run, force=args.force) if args.compute else load_cache(run)
        if agg is None:
            raise SystemExit(f"No cache at {cache_path(run)}. Run once with --compute.")
        print(
            f"{run:55s} N={agg['n_samples']:3d} {agg['steps_convention']:9s} "
            f"stages={agg['mean_stages']:7.1f} gap={agg['gap_ratio_mean']:6.1f} "
            f"r1={agg['autocorr_real_mean']:+.2f} dwell={agg['fraction_dwelling'] * 100:.0f}%"
        )
    print("\nMerged tables (Euclidean + step) are rendered by scripts/paper/tables/trajectory_clusters.py --save")


if __name__ == "__main__":
    main()
