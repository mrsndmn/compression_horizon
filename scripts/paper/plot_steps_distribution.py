"""Distribution of per-token optimizer steps for one progressive-cramming trajectory,
tied to its dwell-and-leap basin structure (cache-only, no GPU).

Reads the same cached ``landscape_pca_pairs.npz`` as ``animate_trajectory.py`` for the
per-sample PCA ``coords`` (basin/jump geometry) and the run's ``progressive_prefixes/``
dataset for per-stage ``steps_taken``. Renders a talk-quality 3-panel figure:

  A. Histogram (log-spaced bins, log-y) of per-token optimizer steps.
  B. Steps vs prefix length, with basin-boundary (big-leap) tokens highlighted.
  C. Steps vs PCA jump magnitude -- the "harder token => bigger leap" basin link.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visual_abstract import _load_npz  # noqa: E402

DEFAULT_NPZ = "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/" "visualizations/landscape_pca_pairs.npz"


def _jumps(P: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(P, axis=0), axis=1)


def _bimodality(x: np.ndarray) -> float:
    n = x.size
    g = skew(x, bias=False)
    k = kurtosis(x, fisher=True, bias=False)
    denom = k + (3.0 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return float((g**2 + 1.0) / denom) if denom else float("nan")


def _acf1(d: np.ndarray) -> float:
    a, b = d[:-1], d[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _load_steps(npz, n_expected: int) -> np.ndarray:
    from datasets import Dataset

    dpath = str(npz["dataset_path"][0])
    sid = int(npz["sample_id"][0])
    ds = Dataset.load_from_disk(dpath)
    df = ds.select_columns(["sample_id", "stage_index", "steps_taken"]).to_pandas()
    df = df[df["sample_id"] == sid].sort_values("stage_index")
    steps = df["steps_taken"].to_numpy().astype(np.float64)
    if steps.shape[0] != n_expected:
        steps = (
            steps[:n_expected] if steps.shape[0] > n_expected else np.pad(steps, (0, n_expected - steps.shape[0]), mode="edge")
        )
    diffs = np.diff(steps)
    nondecreasing = float(np.mean(diffs >= -1e-9)) if diffs.size else 1.0
    med = float(np.median(np.clip(steps, 1.0, None)))
    if steps.shape[0] > 2 and nondecreasing > 0.97 and steps[-1] > 3.0 * med:
        steps = np.diff(steps, prepend=0.0)
    return np.clip(steps, 1.0, None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz_path", type=str, default=DEFAULT_NPZ)
    ap.add_argument("--output", type=str, default=None, help="Output PNG (default: alongside npz).")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--variance", type=float, default=0.99, help="PCA cumulative-variance rank for the basin/jump space.")
    args = ap.parse_args()

    npz = _load_npz(args.npz_path)
    coords = npz["coords"].astype(np.float64)
    evr = npz["explained_variance_ratio"].astype(np.float64)
    seq_len = npz["stage_seq_len"].astype(np.int64).reshape(-1) if "stage_seq_len" in npz else None
    n = coords.shape[0]

    cum = np.cumsum(evr) / evr.sum()
    k = int(np.searchsorted(cum, args.variance) + 1)
    k = max(2, min(k, n - 1))

    steps = _load_steps(npz, n)
    Pk = coords[:, :k]
    d = _jumps(Pk)  # jump that LANDS on stage j (j = 1..n-1)
    med_jump = float(np.median(d))

    # Basin (dwell-and-leap) statistics.
    real_acf = _acf1(d)
    rng = np.random.default_rng(0)
    shuf = np.array([_acf1(d[rng.permutation(d.size)]) for _ in range(500)])
    shuf_p95 = float(np.percentile(shuf, 95))
    n_components = len(set(DBSCAN(eps=2.0 * med_jump, min_samples=1).fit_predict(Pk)))
    gap_ratio = float(d.max() / med_jump)
    bimod = _bimodality(d)
    dwelling = real_acf > shuf_p95

    # Big-leap (basin-boundary) tokens via a robust MAD threshold on jump magnitude.
    mad = np.median(np.abs(d - np.median(d))) / 0.6745
    leap_thr = np.median(d) + 3.0 * mad
    leap_idx = np.where(d > leap_thr)[0] + 1  # stage index that the leap lands on

    # steps[j] aligned with jump d[j-1] (both index the converged stage j).
    rho, pval = spearmanr(steps[1:], d)

    x = seq_len if seq_len is not None else np.arange(1, n + 1)

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

    # --- Panel A: distribution of per-token optimizer steps ---
    axA = axes[0]
    bins = np.logspace(0, np.log10(max(steps.max(), 2.0)), 36)
    axA.hist(steps, bins=bins, color="#4477aa", alpha=0.85, edgecolor="white", linewidth=0.3)
    axA.set_xscale("log")
    axA.set_yscale("log")
    for q, c in [(50, "#cc3311"), (90, "#ee7733"), (99, "#aa3377")]:
        v = np.percentile(steps, q)
        axA.axvline(v, color=c, ls="--", lw=1.4, label=f"p{q} = {v:.0f}")
    axA.set_xlabel("optimizer steps to converge a token (log)")
    axA.set_ylabel("# tokens (log)")
    axA.set_title("A. Per-token step distribution")
    axA.legend(fontsize=10, loc="upper right")
    n_trivial = int((steps <= 1).sum())
    n_capped = int((steps >= steps.max()).sum())
    axA.text(
        0.03,
        0.04,
        f"n={n} tokens · total {steps.sum():.0f} steps\n"
        f"median {np.median(steps):.0f}, mean {steps.mean():.1f}, max {steps.max():.0f}\n"
        f"{n_trivial} tokens ≤1 step · {n_capped} hit the cap",
        transform=axA.transAxes,
        fontsize=9.5,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
    )

    # --- Panel B: steps along the trajectory, basin boundaries marked ---
    axB = axes[1]
    axB.plot(x, steps, color="0.6", lw=0.7, zorder=1)
    axB.scatter(x, steps, s=10, color="#4477aa", zorder=2, label="token")
    if leap_idx.size:
        axB.scatter(
            x[leap_idx],
            steps[leap_idx],
            s=44,
            facecolor="none",
            edgecolor="#cc3311",
            linewidths=1.3,
            zorder=3,
            label=f"big leap (>{leap_thr:.1f}); {leap_idx.size} basin edges",
        )
    axB.set_yscale("log")
    axB.set_xlabel("prefix length (tokens)")
    axB.set_ylabel("optimizer steps (log)")
    axB.set_title("B. Steps along the trajectory")
    axB.legend(fontsize=10, loc="upper left")

    # --- Panel C: steps vs PCA jump magnitude (the basin link) ---
    axC = axes[2]
    axC.scatter(d, steps[1:], s=12, color="#228833", alpha=0.55)
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlabel(f"PCA jump magnitude ‖P_k − P_(k−1)‖  (top-{k}d)")
    axC.set_ylabel("optimizer steps (log)")
    axC.set_title("C. Harder token ⇒ bigger leap")
    axC.text(
        0.03,
        0.95,
        f"Spearman ρ = {rho:+.2f}\n(p = {pval:.1e})",
        transform=axC.transAxes,
        fontsize=11,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
    )

    verdict = "GENUINE DWELLING BASINS" if dwelling else "within shuffle band"
    fig.suptitle(
        f"Llama-3.1-8B  ·  length-{int(x[-1])} sequence  ·  optimizer steps & basin structure\n"
        f"jump lag-1 autocorr {real_acf:+.2f} (shuffle p95 {shuf_p95:+.2f} → {verdict})  ·  "
        f"{n_components} connected components  ·  gap-ratio {gap_ratio:.0f}  ·  bimodality {bimod:.2f}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = args.output or os.path.join(os.path.dirname(args.npz_path), "optimizer_steps_distribution.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved: {out}")
    print(
        f"BASINS: lag-1 jump autocorr {real_acf:+.3f} vs shuffle p95 {shuf_p95:+.3f} -> "
        f"{'dwelling basins' if dwelling else 'no dwelling'}; "
        f"{n_components} components over {n} stages (~1 per {n / max(n_components, 1):.1f}); "
        f"gap_ratio {gap_ratio:.1f}, bimodality {bimod:.2f}, {leap_idx.size} big leaps."
    )
    print(f"STEPS: median {np.median(steps):.0f}, mean {steps.mean():.1f}, max {steps.max():.0f}, total {steps.sum():.0f}.")
    print(f"LINK: Spearman(steps, jump) = {rho:+.3f} (p={pval:.1e}).")


if __name__ == "__main__":
    main()
