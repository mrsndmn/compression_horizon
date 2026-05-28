"""Test the 'low-dim PCA = a few big jumps' hypothesis on progressive-cramming trajectories.

Claim (synthesis of the trajectory findings): a per-sample trajectory's PCA-99% is low because its
variance is concentrated in a few large, surprise-driven jumps, while most (predictable) tokens are
tiny warm-started nudges. If true, the subspace spanned by the few LARGEST-magnitude jumps should
already coincide with the principal axes.

Per sample we measure:
  * pca99            -- minimum #PCs for 99% of trajectory variance (the paper's metric).
  * f@pca99          -- fraction of trajectory variance captured by the span of the pca99
                        LARGEST-magnitude jump vectors. PCA is optimal, so f@pca99 <= 0.99 always;
                        f@pca99 close to 0.99 means the big jumps' directions ARE the principal axes.
  * m99_jumps        -- smallest #largest-jumps whose span captures 99% of variance (>= pca99).
  * mag99            -- #largest jumps holding 99% of total squared jump magnitude (heavy-tailedness).

Memory-frugal: pulls scalar columns in bulk, then decodes ONE sample's embeddings at a time via
``select_columns(["embedding"])[indices]`` (never materialises the whole embedding column).

    PYTHONPATH=./src python scripts/analyze_pca_jump_concentration.py
"""

import argparse

import numpy as np
from datasets import load_from_disk
from scipy.stats import spearmanr

_EXP = "artifacts/experiments_progressive"
MIN_STAGES = 5
PROJ_CAP = 400  # cap on jump-subspace dimension when searching for 99%

RUNS = [
    ("135M", "sl_4096_SmolLM2-135M_ds_pg19_1k_limit_50_lr_0.1"),
    ("360M", "sl_4096_SmolLM2-360M_ds_pg19_1k_limit_50_lr_0.1"),
    ("1.7B", "sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1"),
    ("Llama-8B", "sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1"),
]


def pca99_fast(P: np.ndarray) -> int:
    """Min #PCs for 99% of variance via the n*n Gram-matrix eigenvalues (fast when n < d).

    Same definition as compute_pca_99: Gram eigenvalues equal the squared singular values, but the
    n*n eigendecomposition is far cheaper than an SVD of the n*d matrix for these tall-thin shapes.
    """
    Pc = P - P.mean(axis=0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(Pc @ Pc.T)[::-1], 0.0, None)
    tot = float(ev.sum())
    if tot <= 0:
        return 1
    return int(np.searchsorted(np.cumsum(ev) / tot, 0.99) + 1)


def sample_index_map(run_dir: str):
    ds = load_from_disk(run_dir).select_columns(["sample_id", "stage_index"]).with_format("numpy")
    sid = np.asarray(ds["sample_id"]).reshape(-1)
    stg = np.asarray(ds["stage_index"]).reshape(-1)
    out = {}
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        out[int(s)] = idx[np.argsort(stg[idx], kind="stable")]
    return out


def jump_subspace_capture(Pc: np.ndarray, jumps: np.ndarray, ks_needed: set, cap: int):
    """Cumulative fraction of position variance captured by the span of the top-magnitude jumps.

    Builds an orthonormal basis of jump vectors in descending-magnitude order (Gram-Schmidt) and
    accumulates the energy of the centered positions ``Pc`` projected onto it. Returns a dict
    {K: fraction} for the requested Ks, plus m99 (first K reaching 0.99).
    """
    mags = np.linalg.norm(jumps, axis=1)
    order = np.argsort(-mags)
    total = float((Pc**2).sum())
    fracs, basis, energy, m99 = {}, [], 0.0, None
    for m in range(min(cap, len(order))):
        v = jumps[order[m]].astype(np.float64).copy()
        for q in basis:
            v -= np.dot(v, q) * q
        nv = np.linalg.norm(v)
        if nv > 1e-9:
            q = v / nv
            basis.append(q)
            energy += float(((Pc @ q) ** 2).sum())
        frac = energy / total if total > 0 else float("nan")
        k = m + 1
        if k in ks_needed:
            fracs[k] = frac
        if m99 is None and frac >= 0.99:
            m99 = k
        if m99 is not None and all(k2 <= k for k2 in ks_needed):
            break
    return fracs, (m99 if m99 is not None else float("nan"))


def mag_concentration(jumps: np.ndarray, frac: float = 0.99) -> int:
    """#largest squared jump magnitudes needed to reach `frac` of the total (heavy-tailedness)."""
    a2 = np.sort(np.linalg.norm(jumps, axis=1) ** 2)[::-1]
    if a2.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(a2) / a2.sum(), frac) + 1)


def pca99_iid_null(mags: np.ndarray, dim: int, rng: np.random.Generator, n_draws: int = 1) -> float:
    """PCA-99% of an i.i.d. random-walk with the SAME jump magnitudes but RANDOM isotropic directions.

    Isolates the generic cumulative-sum (Karhunen-Loeve) contribution to low-dimensionality: if the
    real PCA-99% matches this null, the low-dim is just 'positions are running sums', not directional
    structure in the real jumps; if real << null, the real jumps are directionally correlated.
    """
    vals = []
    for _ in range(n_draws):
        dirs = rng.standard_normal((mags.size, dim))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        steps = dirs * mags[:, None]
        P = np.vstack([np.zeros((1, dim)), np.cumsum(steps, axis=0)])
        vals.append(pca99_fast(P))
    return float(np.mean(vals))


def analyze_run(label: str, run_dir: str, seed: int = 0) -> dict:
    idx_map = sample_index_map(run_dir)
    emb_ds = load_from_disk(run_dir).select_columns(["embedding"]).with_format("numpy")
    rng = np.random.default_rng(seed)
    rows = []
    for sid, idx in idx_map.items():
        if idx.size < MIN_STAGES:
            continue
        P = np.asarray(emb_ds[idx.tolist()]["embedding"], dtype=np.float64).reshape(idx.size, -1)
        pca99 = pca99_fast(P)
        Pc = P - P.mean(axis=0, keepdims=True)
        jumps = np.diff(P, axis=0)
        fracs, m99 = jump_subspace_capture(Pc, jumps, {pca99}, PROJ_CAP)
        rows.append(
            {
                "sid": sid,
                "n_jumps": int(jumps.shape[0]),
                "pca99": pca99,
                "pca99_iidnull": pca99_iid_null(np.linalg.norm(jumps, axis=1), P.shape[1], rng),
                "f_at_pca99": fracs.get(pca99, float("nan")),
                "m99_jumps": m99,
                "mag99": mag_concentration(jumps),
            }
        )
        del P, Pc, jumps
    pca = np.array([r["pca99"] for r in rows], float)
    null = np.array([r["pca99_iidnull"] for r in rows], float)
    f = np.array([r["f_at_pca99"] for r in rows], float)
    m99 = np.array([r["m99_jumps"] for r in rows], float)
    nj = np.array([r["n_jumps"] for r in rows], float)
    mag99 = np.array([r["mag99"] for r in rows], float)
    return {
        "label": label,
        "n_samples": len(rows),
        "mean_n_jumps": float(nj.mean()),
        "mean_pca99": float(pca.mean()),
        "mean_pca99_iidnull": float(null.mean()),
        "mean_f_at_pca99": float(np.nanmean(f)),
        "mean_m99_jumps": float(np.nanmean(m99)),
        "mean_mag99": float(mag99.mean()),
        "rho_pca99_m99": float(spearmanr(pca, m99, nan_policy="omit").statistic),
        "rho_pca99_njumps": float(spearmanr(pca, nj).statistic),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=None, help="subset of labels (default: all)")
    args = ap.parse_args()
    runs = [r for r in RUNS if args.runs is None or r[0] in args.runs]
    hdr = f"{'run':9s} {'N':>3s} {'jumps':>6s} {'pca99':>6s} {'iid_null':>8s} " f"{'f@pca99':>8s} {'m99_jmp':>8s} {'mag99':>6s}"
    print(hdr)
    for label, name in runs:
        s = analyze_run(label, f"{_EXP}/{name}/progressive_prefixes")
        print(
            f"{s['label']:9s} {s['n_samples']:3d} {s['mean_n_jumps']:6.0f} {s['mean_pca99']:6.1f} "
            f"{s['mean_pca99_iidnull']:8.1f} {s['mean_f_at_pca99']:8.3f} {s['mean_m99_jumps']:8.1f} "
            f"{s['mean_mag99']:6.1f}"
        )


if __name__ == "__main__":
    main()
