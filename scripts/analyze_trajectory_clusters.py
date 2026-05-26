"""Are low-dimensional progressive-cramming trajectories disconnected clusters?

Validation probe for the hypothesis that the *low-dimensionality* of per-sample
progressive-cramming embedding trajectories is explained by **disconnected,
far-apart local-minima regions ("clusters")** rather than by a single smooth
low-dimensional curve. (Hypothesis: easy next-tokens nudge the converged
embedding a tiny amount while hard next-tokens force a big leap to a different
region -> the converged points should bunch into clusters separated by gaps.)

Unit of analysis (per-sample): the sequence of CONVERGED embeddings
``P_1 .. P_n`` (one per growing-prefix stage), read from a run's
``progressive_prefixes/`` dataset and ordered by ``stage_index``. This matches
the per-sample low-dim finding in ``analysis/trajectory.py::compute_pca_99``.

Three-part test, per sample (confirm OR refute), everything calibrated against a
smooth-curve null so that "smooth" can genuinely win:

  1. Jump-distance distribution (headline). ``d_k = ||P_k - P_{k-1}||``. Cluster
     signature = a bimodal / heavy-tailed jump distribution (many small hops +
     rare big leaps between far-apart clusters). Scale-free metrics:
     gap_ratio = max/median, CV = std/mean, Sarle's bimodality coefficient.
     Reported in full embedding dim; the verdict uses the top-k PCA space so it
     is apples-to-apples with the null.

  2. Connected-component clustering in PCA-reduced space (confirmation). Project
     each trajectory onto its top-k PCs (k = 99%-variance rank; the data is known
     to be low-dim, dodging the curse of dimensionality). Cluster with DBSCAN
     using ``eps = EPS_FACTOR * median jump`` and ``min_samples=1`` -- i.e.
     connected components of the graph that links points closer than a few median
     hops. A smooth, evenly spaced curve collapses to ONE component; only jumps
     materially larger than the local spacing split it.

  3. Smooth-curve null (calibration). Monte-Carlo null of genuinely SMOOTH
     low-dim curves (Gaussian-process samples, RBF kernel) of matched length and
     effective dimension, run through the SAME jump + clustering pipeline. A
     sample counts as "clustered" only when its gap_ratio AND its component count
     both beat the null's p95. A clean negative -- trajectories indistinguishable
     from smooth curves -- refutes the hypothesis and is reported as valid.

Outputs: per-sample + aggregate metrics (JSON), plots, and a written verdict.

Usage::

    PYTHONPATH=./src python scripts/analyze_trajectory_clusters.py \
        --run_dir artifacts/experiments_progressive/sl_4096_SmolLM2-135M_ds_pg19_1k_limit_50_lr_0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from scipy.stats import kurtosis, skew
from sklearn.cluster import DBSCAN

from compression_horizon.analysis.pca_reconstruction import fit_per_sample_pca
from compression_horizon.analysis.trajectory import compute_pca_99, compute_trajectory_length

# Decision thresholds (documented so the verdict is reproducible, not magic).
EPS_FACTOR = 2.0  # DBSCAN eps = EPS_FACTOR * median jump (links points within a few median hops)
NULL_PCTILE = 95.0  # real gap_ratio AND component count must beat this percentile of the null


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_trajectories(run_dir: Path) -> dict[int, np.ndarray]:
    """Load per-sample converged-embedding trajectories from a run directory.

    Returns ``{sample_id: array[n_stages, flat_dim]}`` ordered by stage_index.
    """
    ds_path = run_dir / "progressive_prefixes"
    if not ds_path.exists():
        raise FileNotFoundError(f"No progressive_prefixes/ under {run_dir}")
    # Select only the needed columns BEFORE decoding -- these datasets carry several
    # large per-row embedding columns (orig_embedding, initialization_embedding,
    # pca_coefficients_to_save) and a row-by-row `for row in ds` over all of them is
    # slow / OOM-prone. numpy formatting then returns whole columns as arrays.
    ds = load_from_disk(str(ds_path)).select_columns(["sample_id", "stage_index", "embedding"])
    ds = ds.with_format("numpy")
    sid_arr = np.asarray(ds["sample_id"]).reshape(-1)
    stage_arr = np.asarray(ds["stage_index"]).reshape(-1)
    emb_arr = np.asarray(ds["embedding"], dtype=np.float64).reshape(len(ds), -1)

    trajectories: dict[int, np.ndarray] = {}
    for sid in np.unique(sid_arr):
        mask = sid_arr == sid
        order = np.argsort(stage_arr[mask], kind="stable")
        trajectories[int(sid)] = emb_arr[mask][order]
    return trajectories


# --------------------------------------------------------------------------- #
# Jump-distance statistics (scale-free)
# --------------------------------------------------------------------------- #
def jump_distances(P: np.ndarray) -> np.ndarray:
    """Consecutive L2 jump distances d_k = ||P_k - P_{k-1}||."""
    return np.linalg.norm(np.diff(P, axis=0), axis=1)


def bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's finite-sample bimodality coefficient. >0.555 suggests bimodality."""
    n = x.size
    if n < 4:
        return float("nan")
    g = skew(x, bias=False)
    k = kurtosis(x, fisher=True, bias=False)  # excess kurtosis
    denom = k + (3.0 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    if denom == 0:
        return float("nan")
    return float((g**2 + 1.0) / denom)


def jump_stats(d: np.ndarray) -> dict:
    """Scale-free shape statistics of a jump-distance array."""
    med = float(np.median(d))
    mean = float(np.mean(d))
    return {
        "n_jumps": int(d.size),
        "mean": mean,
        "median": med,
        "max": float(np.max(d)),
        "gap_ratio": float(np.max(d) / med) if med > 0 else float("nan"),
        "cv": float(np.std(d) / mean) if mean > 0 else float("nan"),
        "bimodality": bimodality_coefficient(d),
    }


# --------------------------------------------------------------------------- #
# Connected-component clustering (smooth curve -> 1 component)
# --------------------------------------------------------------------------- #
def connected_components(coords: np.ndarray, eps_factor: float = EPS_FACTOR) -> np.ndarray:
    """DBSCAN labels with eps = eps_factor * median jump, min_samples=1.

    With min_samples=1 every point is a core point, so this returns the connected
    components of the graph linking points within ``eps`` of each other. A smooth,
    evenly spaced path -> a single component; far-apart clusters -> several.
    """
    n = coords.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    d = jump_distances(coords)
    med = float(np.median(d))
    if med <= 0:
        return np.zeros(n, dtype=int)
    eps = eps_factor * med
    return DBSCAN(eps=eps, min_samples=1).fit_predict(coords)


def cluster_metrics(coords: np.ndarray) -> dict:
    """Component count + inter/intra distance ratio in PCA space."""
    labels = connected_components(coords)
    cluster_ids = sorted(int(c) for c in set(labels))
    out = {
        "n_clusters": len(cluster_ids),
        "intra_mean": float("nan"),
        "inter_mean": float("nan"),
        "inter_intra_ratio": float("nan"),
        "labels": labels.tolist(),
    }
    intra, centroids = [], []
    for c in cluster_ids:
        pts = coords[labels == c]
        centroids.append(pts.mean(axis=0))
        if pts.shape[0] >= 2:
            diffs = pts[:, None, :] - pts[None, :, :]
            dmat = np.linalg.norm(diffs, axis=2)
            iu = np.triu_indices(pts.shape[0], k=1)
            intra.append(float(dmat[iu].mean()))
    out["intra_mean"] = float(np.mean(intra)) if intra else 0.0
    if len(centroids) >= 2:
        cen = np.stack(centroids, axis=0)
        diffs = cen[:, None, :] - cen[None, :, :]
        dmat = np.linalg.norm(diffs, axis=2)
        iu = np.triu_indices(len(centroids), k=1)
        out["inter_mean"] = float(dmat[iu].mean())
        if out["intra_mean"] and out["intra_mean"] > 0:
            out["inter_intra_ratio"] = out["inter_mean"] / out["intra_mean"]
    return out


# --------------------------------------------------------------------------- #
# Smooth-curve Monte-Carlo null (Gaussian process, RBF kernel)
# --------------------------------------------------------------------------- #
def smooth_curve_null(n_points: int, n_dim: int, n_samples: int, rng: np.random.Generator) -> dict:
    """Null distribution of jump-shape + component stats for SMOOTH low-dim curves.

    Each null curve is an (n_points x n_dim) Gaussian-process sample with an RBF
    kernel whose lengthscale (~n/4) guarantees smoothness -- no sudden leaps. The
    jump metrics are scale-free, so this isolates the *smoothness* of the path
    from its overall size; the curve is run through the identical jump+clustering
    pipeline used on the real data.
    """
    if n_points < 4:
        return {}
    t = np.arange(n_points, dtype=np.float64)
    lengthscale = max(n_points / 4.0, 2.0)
    sq = (t[:, None] - t[None, :]) ** 2
    cov = np.exp(-sq / (2.0 * lengthscale**2)) + 1e-8 * np.eye(n_points)
    chol = np.linalg.cholesky(cov)

    gaps = np.empty(n_samples)
    ncl = np.empty(n_samples)
    for i in range(n_samples):
        curve = chol @ rng.standard_normal((n_points, n_dim))  # smooth GP draw
        d = jump_distances(curve)
        med = np.median(d)
        gaps[i] = (d.max() / med) if med > 0 else np.nan
        ncl[i] = len(set(connected_components(curve)))
    return {
        "n_samples": int(n_samples),
        "gap_ratio_mean": float(np.nanmean(gaps)),
        "gap_ratio_std": float(np.nanstd(gaps)),
        "gap_ratio_p95": float(np.nanpercentile(gaps, NULL_PCTILE)),
        "n_clusters_mean": float(np.mean(ncl)),
        "n_clusters_p95": float(np.percentile(ncl, NULL_PCTILE)),
        "_gaps": gaps,  # dropped before JSON
    }


# --------------------------------------------------------------------------- #
# Random-walk nulls (separate "discrete basins" from "heavy-tailed wandering")
# --------------------------------------------------------------------------- #
def jump_autocorr(d: np.ndarray) -> float:
    """Lag-1 autocorrelation of the jump-magnitude series.

    Positive => small jumps follow small jumps and big follow big, i.e. the
    optimizer *dwells* then *leaps* (the basin signature). A jump-shuffle null
    has expected autocorr ~0, so a real value above the shuffle band means the
    bunching is in the *order* of the jumps, not just their size distribution.
    """
    if d.size < 3:
        return float("nan")
    a, b = d[:-1], d[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def shuffle_null(coords: np.ndarray, n_samples: int, rng: np.random.Generator) -> dict:
    """Jump-shuffle null: randomly reorder the real jumps, measure autocorrelation.

    Reordering the jump *vectors* and re-integrating leaves the multiset of jump
    *magnitudes* unchanged, so the only thing that changes is their ORDER. The
    decisive statistic is the lag-1 autocorrelation of the jump-magnitude series,
    which is exactly a random permutation of the real magnitudes (expected ~0). A
    real autocorr above this band means small/big jumps are bunched in order
    (dwelling), not just heavy-tailed. O(n) per draw.
    """
    mags = np.linalg.norm(np.diff(coords, axis=0), axis=1)  # real jump magnitudes
    if mags.size < 3:
        return {}
    ac = np.array([jump_autocorr(mags[rng.permutation(mags.size)]) for _ in range(n_samples)], dtype=float)
    return {
        "n_samples": int(n_samples),
        "autocorr_mean": float(np.nanmean(ac)),
        "autocorr_p95": float(np.nanpercentile(ac, NULL_PCTILE)),
    }


def iid_walk_null(n_points: int, n_dim: int, n_samples: int, rng: np.random.Generator) -> dict:
    """i.i.d. random-walk null: gap_ratio of memoryless isotropic Gaussian steps.

    The jumps of a walk ARE its steps, and gap_ratio (max/median) is scale-free,
    so this reduces to the gap_ratio of i.i.d. Gaussian step magnitudes -- no
    cumsum or scaling needed. A real gap_ratio above this null's p95 means the
    real jumps are heavier-tailed than diffusion. O(n) per draw.
    """
    if n_points < 4:
        return {}
    gaps = np.empty(n_samples)
    for i in range(n_samples):
        m = np.linalg.norm(rng.standard_normal((n_points - 1, n_dim)), axis=1)
        med = np.median(m)
        gaps[i] = (m.max() / med) if med > 0 else np.nan
    return {
        "n_samples": int(n_samples),
        "gap_ratio_mean": float(np.nanmean(gaps)),
        "gap_ratio_p95": float(np.nanpercentile(gaps, NULL_PCTILE)),
    }


# --------------------------------------------------------------------------- #
# PCA-reduced projection
# --------------------------------------------------------------------------- #
def pca_project(P: np.ndarray, k: int) -> np.ndarray:
    """Project trajectory onto its own top-k principal directions (centered)."""
    mean, components, _ = fit_per_sample_pca(torch.from_numpy(P))
    mean = mean.numpy()
    components = components.numpy()  # [r, flat_dim]
    k_eff = min(k, components.shape[0])
    return (P - mean) @ components[:k_eff].T  # [n_stages, k_eff]


# --------------------------------------------------------------------------- #
# Per-sample analysis + verdict
# --------------------------------------------------------------------------- #
def analyze_sample(P: np.ndarray, n_null: int, rng: np.random.Generator) -> dict:
    n_stages, flat_dim = P.shape

    pca99 = compute_pca_99(torch.from_numpy(P))
    k = int(pca99) if pca99 else 2
    k = max(2, min(k, n_stages - 1, flat_dim))
    coords = pca_project(P, k)

    stats_full = jump_stats(jump_distances(P))  # descriptive (full dim)
    stats_pca = jump_stats(jump_distances(coords))  # used for verdict (matches null space)
    clusters = cluster_metrics(coords)

    null = smooth_curve_null(n_stages, k, n_null, rng)
    null_gaps = null.pop("_gaps", np.array([]))
    gap_pctile = float((null_gaps < stats_pca["gap_ratio"]).mean() * 100.0) if null_gaps.size else float("nan")
    gap_above_null = bool(stats_pca["gap_ratio"] > null.get("gap_ratio_p95", np.inf)) if null else False
    clusters_above_null = (
        clusters["n_clusters"] >= 2 and bool(clusters["n_clusters"] > null.get("n_clusters_p95", np.inf)) if null else False
    )

    if gap_above_null and clusters_above_null:
        verdict = "clustered"
    elif (not gap_above_null) and (not clusters_above_null):
        verdict = "smooth"
    else:
        verdict = "ambiguous"

    # Random-walk nulls: separate genuine dwelling basins from heavy-tailed wandering.
    shuffle = shuffle_null(coords, n_null, rng)
    iid = iid_walk_null(n_stages, k, n_null, rng)
    autocorr_real = jump_autocorr(jump_distances(coords))
    beats_iid_gap = bool(stats_pca["gap_ratio"] > iid.get("gap_ratio_p95", np.inf)) if iid else False
    beats_shuffle_autocorr = (
        bool(np.isfinite(autocorr_real) and autocorr_real > shuffle.get("autocorr_p95", np.inf)) if shuffle else False
    )
    if beats_shuffle_autocorr:
        basin_strength = "dwelling_basins"  # bunching is in the order of jumps, not just their sizes
    elif beats_iid_gap:
        basin_strength = "heavy_tailed"  # jumps heavier-tailed than diffusion, but no temporal dwelling
    else:
        basin_strength = "diffuse_or_smooth"

    return {
        "n_stages": int(n_stages),
        "trajectory_length": compute_trajectory_length(torch.from_numpy(P)),
        "pca99": int(pca99) if pca99 else None,
        "pca_k_used": int(k),
        "jumps_full_dim": stats_full,
        "jumps_pca_space": stats_pca,
        "smooth_null": null,
        "shuffle_null": shuffle,
        "iid_walk_null": iid,
        "jump_autocorr": autocorr_real,
        "gap_ratio_percentile_vs_null": gap_pctile,
        "gap_above_null_p95": gap_above_null,
        "clusters_above_null_p95": bool(clusters_above_null),
        "beats_iid_gap_p95": beats_iid_gap,
        "beats_shuffle_autocorr_p95": beats_shuffle_autocorr,
        "basin_strength": basin_strength,
        "clustering": {kk: vv for kk, vv in clusters.items() if kk != "labels"},
        "verdict": verdict,
        "_coords": coords,
        "_labels": clusters["labels"],
        "_jumps": jump_distances(P),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _grid(n: int) -> tuple[int, int]:
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    return rows, cols


def plot_jump_histograms(results: dict, out: Path, max_samples: int) -> None:
    sids = list(results.keys())[:max_samples]
    rows, cols = _grid(len(sids))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, sid in zip(axes.flat, sids):
        r = results[sid]
        d = r["_jumps"]
        ax.hist(d, bins=min(20, max(5, d.size // 2)), color="#4477aa", alpha=0.85)
        ax.axvline(np.median(d), color="k", ls="--", lw=1, label="median")
        ax.axvline(np.max(d), color="#cc3311", ls=":", lw=1, label="max")
        ax.set_title(f"s{sid} | gap={r['jumps_full_dim']['gap_ratio']:.1f} | {r['verdict']}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(sids) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Per-sample jump-distance distributions (||P_k - P_{k-1}||)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "jump_histograms.png", dpi=130)
    plt.close(fig)


def plot_pca_scatter(results: dict, out: Path, max_samples: int) -> None:
    sids = list(results.keys())[:max_samples]
    rows, cols = _grid(len(sids))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows), squeeze=False)
    for ax, sid in zip(axes.flat, sids):
        r = results[sid]
        coords = r["_coords"]
        labels = np.asarray(r["_labels"])
        ax.plot(coords[:, 0], coords[:, 1], "-", color="0.7", lw=0.8, zorder=1)
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=28, zorder=2, vmin=0, vmax=9)
        ax.scatter(coords[0, 0], coords[0, 1], marker="*", s=120, edgecolor="k", facecolor="none", zorder=3)
        ax.set_title(f"s{sid} | k={r['pca_k_used']} | comp={r['clustering']['n_clusters']} | {r['verdict']}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(sids) :]:
        ax.axis("off")
    fig.suptitle("Per-sample trajectories in top-2 PCA dims (star=start, color=connected component)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "pca_scatter.png", dpi=130)
    plt.close(fig)


def plot_cluster_counts(results: dict, out: Path) -> None:
    counts = [r["clustering"]["n_clusters"] for r in results.values()]
    null_p95 = [r["smooth_null"].get("n_clusters_p95", np.nan) for r in results.values()]
    fig, ax = plt.subplots(figsize=(6, 4))
    maxc = max(counts) if counts else 1
    ax.hist(counts, bins=np.arange(-0.5, maxc + 1.5, 1), color="#228833", alpha=0.85, rwidth=0.9, label="real")
    mp = np.nanmean(null_p95)
    if np.isfinite(mp):
        ax.axvline(mp, color="#cc3311", ls="--", label=f"mean smooth-null p95 ({mp:.1f})")
    ax.set_xlabel("# connected components per trajectory")
    ax.set_ylabel("# samples")
    ax.set_title("Component-count distribution (real vs smooth-null p95)")
    ax.set_xticks(range(0, maxc + 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cluster_count_dist.png", dpi=130)
    plt.close(fig)


def plot_gap_vs_null(results: dict, out: Path) -> None:
    sids = list(results.keys())
    real = np.array([results[s]["jumps_pca_space"]["gap_ratio"] for s in sids])
    null_p95 = np.array([results[s]["smooth_null"].get("gap_ratio_p95", np.nan) for s in sids])
    null_mean = np.array([results[s]["smooth_null"].get("gap_ratio_mean", np.nan) for s in sids])
    order = np.argsort(real)
    x = np.arange(len(sids))
    fig, ax = plt.subplots(figsize=(max(7, len(sids) * 0.28), 4.5))
    iid_p95 = np.array([results[s]["iid_walk_null"].get("gap_ratio_p95", np.nan) for s in sids])
    ax.plot(x, real[order], "o-", color="#cc3311", label="real gap_ratio (PCA space)", ms=4)
    ax.plot(x, null_p95[order], "s--", color="#4477aa", label="smooth-null p95", ms=3)
    ax.plot(x, null_mean[order], "-", color="0.6", label="smooth-null mean")
    ax.plot(x, iid_p95[order], "^:", color="#ee7733", label="i.i.d. random-walk p95", ms=3)
    ax.set_xlabel("sample (sorted by real gap_ratio)")
    ax.set_ylabel("gap_ratio = max jump / median jump")
    ax.set_title("Real jump gap_ratio vs smooth + i.i.d.-walk nulls (above = heavier-tailed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gap_ratio_vs_null.png", dpi=130)
    plt.close(fig)


def plot_basin_structure(results: dict, out: Path) -> None:
    """Decisive test: real jump autocorrelation vs the jump-shuffle null band."""
    sids = list(results.keys())
    real = np.array([results[s]["jump_autocorr"] for s in sids])
    shuf_p95 = np.array([results[s]["shuffle_null"].get("autocorr_p95", np.nan) for s in sids])
    shuf_mean = np.array([results[s]["shuffle_null"].get("autocorr_mean", np.nan) for s in sids])
    color = {"dwelling_basins": "#cc3311", "heavy_tailed": "#ee7733", "diffuse_or_smooth": "0.6"}
    cols = [color.get(results[s]["basin_strength"], "0.6") for s in sids]
    order = np.argsort(real)
    x = np.arange(len(sids))
    fig, ax = plt.subplots(figsize=(max(7, len(sids) * 0.28), 4.5))
    ax.bar(x, real[order], color=[cols[i] for i in order], alpha=0.85, label="real jump autocorr")
    ax.plot(x, shuf_p95[order], "s--", color="#4477aa", ms=3, label="jump-shuffle p95")
    ax.plot(x, shuf_mean[order], "-", color="k", lw=1, label="jump-shuffle mean (~0)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("sample (sorted by real autocorr)")
    ax.set_ylabel("lag-1 autocorrelation of jump magnitudes")
    ax.set_title("Dwelling test: real jump autocorr vs jump-shuffle null (above blue = genuine dwelling basins)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "basin_structure.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Verdict writeup
# --------------------------------------------------------------------------- #
def _interpretation(agg: dict) -> str:
    frac_cl = agg["fraction_clustered"]
    frac_sm = agg["fraction_smooth"]
    frac_dwell = agg["fraction_dwelling_basins"]
    comp_per_stage = agg["comp_per_stage_mean"]
    inter_intra = agg["inter_intra_mean"]
    acf_r = agg["jump_autocorr_real_mean"]
    acf_s = agg["jump_autocorr_shuffle_mean"]
    many_basin = comp_per_stage >= 0.15
    shape = (
        f"a 'many-basin staircase' (~{comp_per_stage:.2f} basins per stage, i.e. one every "
        f"~{1.0/comp_per_stage:.1f} stages) rather than a handful of distant attractors"
        if many_basin and comp_per_stage > 0
        else "a small number of far-apart clusters (the strong form of the hypothesis)"
    )

    if frac_sm >= 0.5 and frac_cl < 0.5:
        return (
            "The data REFUTES the clustered-local-minima hypothesis: most trajectories are "
            "statistically indistinguishable from smooth low-dim curves (jump gap_ratio within the "
            "smooth-curve null, no excess components). Low-dimensionality looks like a smooth manifold."
        )

    not_smooth = (
        "Trajectories are decisively NOT smooth curves: jump gap_ratio sits at the "
        f"~{agg['gap_percentile_mean']:.0f}th percentile of the smooth-curve null and points split "
        f"into well-separated groups (mean inter/intra ratio {inter_intra:.1f}). "
    )
    if frac_dwell >= 0.5:
        return (
            not_smooth + "Crucially, this survives the JUMP-SHUFFLE null: jump-magnitude autocorrelation is "
            f"{acf_r:+.2f} for the real trajectories vs {acf_s:+.2f} for shuffles (~0 by construction), "
            "so small and large jumps are bunched in ORDER -- the optimizer genuinely DWELLS then "
            "LEAPS. That is real dwell-and-jump basin structure, not an artifact of the jump-size "
            f"distribution. The structure is {shape}. Low-dimensionality coexists with a discrete, "
            "punctuated walk through nearby-but-separated minima."
        )
    if frac_cl >= 0.5:
        return (
            not_smooth + "HOWEVER, this does NOT survive the jump-shuffle null: real jump-magnitude "
            f"autocorrelation ({acf_r:+.2f}) is within the shuffle band ({acf_s:+.2f}), meaning the "
            "gappiness is explained by a heavy-tailed distribution of jump SIZES rather than by "
            "temporal dwelling. The picture is closer to heavy-tailed wandering (variable step sizes "
            "in random-ish order) than to discrete, revisited basins. The 'clusters' are largely a "
            "consequence of jump-size heterogeneity, so the strong 'far-apart local minima' reading "
            "is not supported by this run."
        )
    return (
        "MIXED evidence: a non-trivial fraction show cluster signatures while others look smooth. "
        "Worth correlating with optimization difficulty (the deferred steps/surprisal link)."
    )


def aggregate(results: dict) -> dict:
    n = len(results)
    verdicts = [r["verdict"] for r in results.values()]
    n_cl = verdicts.count("clustered")
    n_sm = verdicts.count("smooth")
    n_amb = verdicts.count("ambiguous")
    gaps_real = np.array([r["jumps_pca_space"]["gap_ratio"] for r in results.values()])
    gaps_null = np.array([r["smooth_null"].get("gap_ratio_mean", np.nan) for r in results.values()])
    pctiles = np.array([r["gap_ratio_percentile_vs_null"] for r in results.values()])
    nstages = np.array([r["n_stages"] for r in results.values()], dtype=float)
    nclusters = np.array([r["clustering"]["n_clusters"] for r in results.values()], dtype=float)
    null_ncl = np.array([r["smooth_null"].get("n_clusters_mean", np.nan) for r in results.values()])
    bimod = np.array([r["jumps_full_dim"]["bimodality"] for r in results.values()])
    inter_intra = np.array([r["clustering"]["inter_intra_ratio"] for r in results.values()])
    pca99 = np.array([r["pca99"] for r in results.values() if r["pca99"] is not None], dtype=float)
    basin = [r["basin_strength"] for r in results.values()]
    acf_real = np.array([r["jump_autocorr"] for r in results.values()], dtype=float)
    acf_shuf = np.array([r["shuffle_null"].get("autocorr_mean", np.nan) for r in results.values()], dtype=float)
    return {
        "n_dwelling_basins": basin.count("dwelling_basins"),
        "n_heavy_tailed": basin.count("heavy_tailed"),
        "n_diffuse_or_smooth": basin.count("diffuse_or_smooth"),
        "fraction_dwelling_basins": basin.count("dwelling_basins") / n if n else 0.0,
        "fraction_heavy_tailed": basin.count("heavy_tailed") / n if n else 0.0,
        "jump_autocorr_real_mean": float(np.nanmean(acf_real)) if n else float("nan"),
        "jump_autocorr_shuffle_mean": float(np.nanmean(acf_shuf)) if n else float("nan"),
        "n_samples": n,
        "n_clustered": n_cl,
        "n_smooth": n_sm,
        "n_ambiguous": n_amb,
        "fraction_clustered": n_cl / n if n else 0.0,
        "fraction_smooth": n_sm / n if n else 0.0,
        "fraction_ambiguous": n_amb / n if n else 0.0,
        "gap_ratio_real_mean": float(np.nanmean(gaps_real)) if n else float("nan"),
        "gap_ratio_null_mean": float(np.nanmean(gaps_null)) if n else float("nan"),
        "gap_percentile_mean": float(np.nanmean(pctiles)) if n else float("nan"),
        "n_clusters_real_mean": float(np.mean(nclusters)) if n else float("nan"),
        "n_clusters_null_mean": float(np.nanmean(null_ncl)) if n else float("nan"),
        "n_clusters_max": int(np.max(nclusters)) if n else 0,
        "comp_per_stage_mean": float(np.mean(nclusters / nstages)) if n else float("nan"),
        "frac_bimodal_jumps": float(np.nanmean(bimod > 0.555)) if n else float("nan"),
        "inter_intra_mean": float(np.nanmean(inter_intra)) if n else float("nan"),
        "pca99_mean": float(np.mean(pca99)) if pca99.size else float("nan"),
    }


def write_verdict(run_name: str, results: dict, skipped: list[int], agg: dict, out: Path) -> None:
    n = len(results)
    frac_cl = agg["fraction_clustered"]
    frac_sm = agg["fraction_smooth"]
    frac_dwell = agg["fraction_dwelling_basins"]
    many_basin = agg["comp_per_stage_mean"] >= 0.15
    if frac_sm >= 0.5 and frac_cl < 0.5:
        headline = "REFUTED (trajectories are dominated by smooth low-dim curves, not clusters)"
    elif frac_dwell >= 0.5:
        shape = "a MANY-basin staircase" if many_basin else "a few far-apart clusters"
        headline = f"CONFIRMED -- genuine dwelling basins (beats the jump-shuffle null), shaped as {shape}"
    elif frac_cl >= 0.5:
        headline = (
            "MECHANISM PARTIAL -- trajectories are NOT smooth, but the gappiness is explained by "
            "heavy-tailed jump SIZES rather than temporal dwelling (does NOT beat the jump-shuffle null); "
            "more heavy-tailed wandering than discrete revisited basins"
        )
    elif frac_cl >= 0.2:
        headline = "PARTIALLY SUPPORTED (a substantial minority cluster; rest smooth/ambiguous)"
    else:
        headline = "INCONCLUSIVE (neither clustered nor smooth dominates)"

    lines = [
        f"# Verdict: are these low-dim trajectories disconnected clusters?  ({run_name})",
        "",
        f"**Run:** `{run_name}`  ",
        f"**Samples analyzed:** {n}  (skipped {len(skipped)} with too few stages)",
        "",
        f"## {headline}",
        "",
        "Per-sample classification ('clustered' requires BOTH a PCA-space jump gap_ratio "
        f"above the smooth-curve null's p{int(NULL_PCTILE)} AND more connected components "
        f"(DBSCAN eps={EPS_FACTOR}x median jump) than the null's p{int(NULL_PCTILE)}):",
        "",
        f"- clustered: {agg['n_clustered']} / {n}  ({frac_cl:.0%})",
        f"- smooth:    {agg['n_smooth']} / {n}  ({frac_sm:.0%})",
        f"- ambiguous: {agg['n_ambiguous']} / {n}  ({agg['fraction_ambiguous']:.0%})",
        "",
        "## Aggregate geometry",
        f"- mean gap_ratio (real, PCA space): {agg['gap_ratio_real_mean']:.2f}  "
        f"vs smooth-null mean {agg['gap_ratio_null_mean']:.2f}",
        f"- mean real gap_ratio percentile vs its own null: {agg['gap_percentile_mean']:.0f}",
        f"- mean # components/trajectory (real): {agg['n_clusters_real_mean']:.2f}  "
        f"vs smooth-null mean {agg['n_clusters_null_mean']:.2f}  (max real {agg['n_clusters_max']})",
        f"- mean components per stage: {agg['comp_per_stage_mean']:.2f}  "
        "(small constant => few clusters; grows with length => many-basin staircase)",
        f"- fraction of samples with bimodal jumps (Sarle > 0.555): {agg['frac_bimodal_jumps']:.0%}",
        f"- mean inter/intra-cluster distance ratio: {agg['inter_intra_mean']:.1f}  "
        "(higher => groups are far apart vs their internal spread)",
        f"- mean PCA-99 rank: {agg['pca99_mean']:.2f}  (the low-dim finding, for reference)",
        "",
        "## Basin structure (random-walk nulls)",
        "Separates genuine dwelling basins from mere heavy-tailed wandering:",
        "",
        f"- dwelling_basins:    {agg['n_dwelling_basins']} / {n}  ({agg['fraction_dwelling_basins']:.0%})  "
        "(jump autocorrelation beats the jump-shuffle null => small/big jumps are bunched in ORDER)",
        f"- heavy_tailed:       {agg['n_heavy_tailed']} / {n}  ({agg['fraction_heavy_tailed']:.0%})  "
        "(jumps heavier-tailed than an i.i.d. random walk, but no temporal dwelling)",
        f"- diffuse_or_smooth:  {agg['n_diffuse_or_smooth']} / {n}",
        f"- mean jump autocorrelation: real {agg['jump_autocorr_real_mean']:+.2f}  "
        f"vs jump-shuffle null {agg['jump_autocorr_shuffle_mean']:+.2f}  (shuffle ~0 by construction)",
        "",
        "## Interpretation",
        _interpretation(agg),
        "",
        "## Method",
        "- Unit: per-sample converged-embedding trajectory P_1..P_n across growing-prefix stages.",
        "- Jump test: d_k=||P_k-P_{k-1}||; gap_ratio/CV/bimodality (scale-free). Verdict uses top-k PCA space.",
        "- Clustering: connected components (DBSCAN eps=2x median jump, min_samples=1) in top-k PCA space.",
        "- Null: Monte-Carlo Gaussian-process smooth curves (RBF kernel, lengthscale~n/4) through the same pipeline.",
        "- 'clustered' counts only when real gap_ratio AND component count beat the smooth null; a clean 'smooth' is a valid refutation.",
        "",
        "## Caveats",
        "- Three nulls are used: smooth-curve (is it a curve?), i.i.d. random walk (heavier-tailed than "
        "diffusion?), and jump-shuffle (does structure survive fixing the jump-size distribution?). The "
        "jump-shuffle autocorrelation test is the decisive one for 'genuine dwelling basins'.",
        "- inter/intra ratio is partly circular with the DBSCAN eps rule; gap_ratio-vs-null and the "
        "jump-autocorrelation-vs-shuffle are the non-circular core evidence.",
        "- Difficulty->jump linkage (do big jumps coincide with high steps_taken / surprisal?) was "
        "deferred per the spec and is the recommended follow-up.",
        "",
        "See `summary.json` for per-sample metrics and the PNGs for visual evidence.",
    ]
    (out / "VERDICT.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="Run directory containing progressive_prefixes/")
    ap.add_argument(
        "--out_dir", default=None, help="Output dir (default: artifacts/analysis/trajectory_clusters_135m/<run_name>)"
    )
    ap.add_argument("--min_stages", type=int, default=5, help="Skip samples with fewer stages than this")
    ap.add_argument("--n_null", type=int, default=200, help="Monte-Carlo smooth-curve null samples")
    ap.add_argument("--max_plot_samples", type=int, default=12, help="Max samples per plot grid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_name = run_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts/analysis/trajectory_clusters_135m") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading trajectories from {run_dir} ...")
    trajectories = load_trajectories(run_dir)
    print(f"  {len(trajectories)} samples found.")

    results: dict[int, dict] = {}
    skipped: list[int] = []
    for sid in sorted(trajectories):
        P = trajectories[sid]
        if P.shape[0] < args.min_stages:
            skipped.append(sid)
            continue
        results[sid] = analyze_sample(P, args.n_null, rng)
        r = results[sid]
        sh = r["shuffle_null"].get("autocorr_p95", float("nan"))
        print(
            f"  s{sid}: n={r['n_stages']:3d} pca99={r['pca99']} "
            f"gap_ratio(pca)={r['jumps_pca_space']['gap_ratio']:.1f} "
            f"comp={r['clustering']['n_clusters']} "
            f"jump_acf={r['jump_autocorr']:+.2f} (shuffle p95 {sh:+.2f}) "
            f"-> {r['verdict']} / {r['basin_strength']}"
        )

    if not results:
        raise SystemExit(f"No samples with >= {args.min_stages} stages; nothing to analyze.")

    agg = aggregate(results)

    print("\nWriting plots ...")
    plot_jump_histograms(results, out_dir, args.max_plot_samples)
    plot_pca_scatter(results, out_dir, args.max_plot_samples)
    plot_cluster_counts(results, out_dir)
    plot_gap_vs_null(results, out_dir)
    plot_basin_structure(results, out_dir)

    clean = {str(sid): {k: v for k, v in r.items() if not k.startswith("_")} for sid, r in results.items()}
    summary = {
        "run": run_name,
        "run_dir": str(run_dir),
        "skipped_samples": skipped,
        "thresholds": {"eps_factor": EPS_FACTOR, "null_pctile": NULL_PCTILE},
        "aggregate": agg,
        "per_sample": clean,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    write_verdict(run_name, results, skipped, agg, out_dir)

    print(
        f"\nVerdict (vs smooth null): clustered={agg['fraction_clustered']:.0%} "
        f"smooth={agg['fraction_smooth']:.0%} ambiguous={agg['fraction_ambiguous']:.0%}"
    )
    print(
        f"Basin structure (random-walk nulls): dwelling_basins={agg['fraction_dwelling_basins']:.0%} "
        f"heavy_tailed={agg['fraction_heavy_tailed']:.0%} "
        f"(real jump-acf {agg['jump_autocorr_real_mean']:+.2f} vs shuffle {agg['jump_autocorr_shuffle_mean']:+.2f})"
    )
    print(f"Artifacts written to {out_dir}/")


if __name__ == "__main__":
    main()
