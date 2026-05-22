"""Non-linear dimensionality diagnostics for compression-embedding trajectories.

Complements the linear PCA analysis (``analysis/trajectory.py``,
``analysis/pca_reconstruction.py``) used in the paper. Two post-hoc additions
on a saved progressive trajectory ``{E^(1), ..., E^(n)}``:

  * :func:`estimate_twonn` -- Two-NN intrinsic-dimension estimator (Facco et
    al., 2017). Unlike PCA-99 % it does not assume the trajectory lies in a
    linear subspace, so it is a non-linear cross-check on the dimensionality
    claim.
  * :func:`project_2d` -- 2-D projection of the trajectory point cloud via
    PCA, t-SNE or UMAP, for visualization figures.

t-SNE / UMAP are visualization-only: they are not invertible and provide no
explained-variance curve, so they do not replace PCA for the quantitative
reconstruction analysis.
"""

from __future__ import annotations

import numpy as np


def _as_2d(points) -> np.ndarray:
    """Coerce array-like input to a 2-D ``[n, dim]`` float64 array."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"points must be [n, dim] (or 1-D), got shape {arr.shape}")
    return arr


def estimate_twonn(points, *, discard_fraction: float = 0.1) -> dict:
    """Two-NN intrinsic-dimension estimate (Facco et al., 2017).

    For every point the ratio ``mu = r2 / r1`` of its distances to the second
    and first nearest neighbours is Pareto-distributed with CDF
    ``F(mu) = 1 - mu^-d``, where ``d`` is the intrinsic dimension. ``d`` is
    recovered by an origin-anchored linear fit of ``-log(1 - F(mu))`` against
    ``log(mu)``; the noisy upper tail of ``mu`` (largest ``discard_fraction``)
    is dropped before the fit.

    ``points``: ``[n, dim]`` array-like. Returns a dict with the estimate and
    the point / used counts.
    """
    from sklearn.neighbors import NearestNeighbors

    x = _as_2d(points)
    n = x.shape[0]
    if n < 10:
        raise ValueError(f"Two-NN needs >= 10 points, got {n}")
    if not 0.0 <= discard_fraction < 1.0:
        raise ValueError(f"discard_fraction must be in [0, 1), got {discard_fraction}")

    distances, _ = NearestNeighbors(n_neighbors=3).fit(x).kneighbors(x)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    mu = np.divide(r2, r1, out=np.full_like(r1, np.inf), where=r1 > 0)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1.0)])
    m = mu.shape[0]
    if m < 3:
        raise ValueError("Two-NN: not enough non-degenerate points (coincident neighbours).")

    # Keep at most m-1 points so the empirical CDF never reaches 1.0.
    keep = max(2, min(m - 1, int(round(m * (1.0 - discard_fraction)))))
    log_mu = np.log(mu[:keep])
    f_emp = np.arange(1, keep + 1, dtype=np.float64) / m
    y = -np.log1p(-f_emp)
    intrinsic_dim = float(np.sum(log_mu * y) / np.sum(log_mu * log_mu))
    return {
        "intrinsic_dim": intrinsic_dim,
        "n_points": int(n),
        "n_used": int(keep),
        "discard_fraction": float(discard_fraction),
    }


def project_2d(points, method: str = "pca", *, seed: int = 42, **kwargs) -> np.ndarray:
    """Project a point cloud to 2-D. ``method`` in ``{"pca", "tsne", "umap"}``.

    Returns an ``[n, 2]`` float array. ``umap`` requires the optional
    ``umap-learn`` package. t-SNE / UMAP hyper-parameters (``perplexity``,
    ``n_neighbors``, ``min_dist``) may be overridden through ``kwargs``.
    """
    x = _as_2d(points)
    method = method.lower()
    n = x.shape[0]

    if method == "pca":
        from sklearn.decomposition import PCA

        return np.asarray(PCA(n_components=2, random_state=seed).fit_transform(x))

    if method == "tsne":
        from sklearn.manifold import TSNE

        default_perplexity = min(30.0, max(2.0, (n - 1) / 3.0))
        perplexity = float(kwargs.get("perplexity", default_perplexity))
        if perplexity >= n:
            raise ValueError(f"t-SNE perplexity ({perplexity}) must be < n_points ({n})")
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
        return np.asarray(tsne.fit_transform(x))

    if method == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("method='umap' requires the 'umap-learn' package (uv sync).") from exc

        n_neighbors = int(kwargs.get("n_neighbors", min(15, max(2, n - 1))))
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=n_neighbors,
            min_dist=float(kwargs.get("min_dist", 0.1)),
        )
        return np.asarray(reducer.fit_transform(x))

    raise ValueError(f"unknown method {method!r}; expected one of: pca, tsne, umap")


def plane_grid(coords_2d, *, resolution: int = 50, margin: float = 0.15):
    """Axis-aligned grid covering a 2-D point cloud (plus a relative margin).

    Used to sample the accuracy landscape behind a PCA-projected trajectory
    (paper Figure 3). Returns ``(grid_xy, extent)`` where ``grid_xy`` is
    ``[resolution**2, 2]`` (row-major, matching ``numpy.meshgrid(xs, ys)``
    ravelled) and ``extent`` is ``(xmin, xmax, ymin, ymax)``.
    """
    pts = _as_2d(coords_2d)
    if pts.shape[1] != 2:
        raise ValueError(f"coords_2d must be [n, 2], got shape {pts.shape}")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span_x = (xmax - xmin) or 1.0
    span_y = (ymax - ymin) or 1.0
    xmin, xmax = xmin - margin * span_x, xmax + margin * span_x
    ymin, ymax = ymin - margin * span_y, ymax + margin * span_y

    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return grid_xy, (float(xmin), float(xmax), float(ymin), float(ymax))


def reconstruct_from_plane(grid_xy, mean, basis_2) -> np.ndarray:
    """Lift 2-D PCA-plane coordinates back to the full embedding space.

    ``E = mean + g1 * PC1 + g2 * PC2``. This is the inverse of projecting an
    embedding onto the first two principal directions, and is what makes the
    PCA accuracy landscape well-defined (t-SNE / UMAP have no such inverse).

    ``grid_xy``: ``[g, 2]``; ``mean``: ``[dim]``; ``basis_2``: ``[2, dim]``
    (the two principal directions). Returns ``[g, dim]``.
    """
    grid = _as_2d(grid_xy)
    if grid.shape[1] != 2:
        raise ValueError(f"grid_xy must be [g, 2], got shape {grid.shape}")
    basis = np.asarray(basis_2, dtype=np.float64)
    mean_arr = np.asarray(mean, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != 2:
        raise ValueError(f"basis_2 must be [2, dim], got shape {basis.shape}")
    if mean_arr.ndim != 1 or basis.shape[1] != mean_arr.shape[0]:
        raise ValueError(f"basis_2 [2, dim] must match mean [dim]; got {basis.shape} and {mean_arr.shape}")
    return mean_arr[None, :] + grid @ basis
