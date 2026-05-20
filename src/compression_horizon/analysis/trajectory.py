"""Optimization-trajectory diagnostics (paper Section 5.1, Table 13).

Progressive cramming produces a sequence of converged embeddings
``{e^(k)}_{k=1..n}`` per sample (one per stage). Two paper-canonical metrics:

    L_traj = sum_{k=1..n-1} ||e^(k+1) - e^(k)||_2                        (eq. 3)
    PCA 99% = min { m : sum_{i=1..m} sigma_i^2 / sum sigma^2 >= 0.99 }

where ``sigma_i`` are the singular values of the mean-centered stage matrix.
PCA is run *per sample* (not pooled across samples), then the per-sample
component counts are averaged.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def compute_trajectory_length(stage_embeddings: torch.Tensor) -> float:
    """Eq. 3: sum of L2 distances between consecutive stage embeddings.

    ``stage_embeddings``: [n_stages, ...] tensor (any trailing shape). For a
    single-stage trajectory the length is 0.0.
    """
    if stage_embeddings.dim() < 2:
        raise ValueError(f"Expected stage_embeddings of shape [n_stages, ...], got {tuple(stage_embeddings.shape)}")
    n_stages = stage_embeddings.shape[0]
    if n_stages < 2:
        return 0.0
    flat = stage_embeddings.reshape(n_stages, -1).to(torch.float64)
    diffs = flat[1:] - flat[:-1]
    return float(torch.linalg.vector_norm(diffs, dim=1).sum().item())


def compute_pca_99(stage_embeddings: torch.Tensor, variance_threshold: float = 0.99) -> Optional[int]:
    """Minimum #PCA components reaching ``variance_threshold`` cumulative variance.

    Returns ``None`` if the trajectory has fewer than 2 stages (PCA undefined)
    or zero variance (all stages identical).
    """
    if stage_embeddings.dim() < 2:
        raise ValueError(f"Expected stage_embeddings of shape [n_stages, ...], got {tuple(stage_embeddings.shape)}")
    n_stages = stage_embeddings.shape[0]
    if n_stages < 2:
        return None
    flat = stage_embeddings.reshape(n_stages, -1).to(torch.float64)
    centered = flat - flat.mean(dim=0, keepdim=True)
    # Economy SVD: singular values squared = per-component variance (up to a factor of (n-1)).
    _, singular, _ = torch.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    total = float(variance.sum().item())
    if total == 0.0 or not math.isfinite(total):
        return None
    cumulative = torch.cumsum(variance, dim=0) / total
    # First index whose cumulative ratio reaches the threshold.
    above = (cumulative >= variance_threshold).nonzero(as_tuple=False)
    return int(above[0].item()) + 1 if above.numel() > 0 else int(variance.numel())


def summarize_trajectory(
    per_sample_lengths: list[float],
    per_sample_pca99: list[Optional[int]],
    per_sample_num_stages: list[int],
) -> dict:
    """Aggregate per-sample metrics into Table-13-style mean ± std.

    Samples whose ``pca_99`` is ``None`` (trajectory too short for PCA) are
    excluded from the PCA aggregate but kept in the trajectory-length aggregate
    (length 0.0 for a 1-stage trajectory is still a valid measurement).
    """
    if len(per_sample_lengths) != len(per_sample_pca99) or len(per_sample_lengths) != len(per_sample_num_stages):
        raise ValueError(
            f"Length mismatch: lengths={len(per_sample_lengths)}, "
            f"pca99={len(per_sample_pca99)}, num_stages={len(per_sample_num_stages)}"
        )
    n = len(per_sample_lengths)
    if n == 0:
        return {
            "trajectory_length": {"mean": 0.0, "std": 0.0},
            "pca_99": {"mean": 0.0, "std": 0.0},
            "num_stages": {"mean": 0.0, "std": 0.0},
            "num_samples": 0,
            "num_pca_excluded": 0,
        }

    lengths = torch.tensor(per_sample_lengths, dtype=torch.float64)
    stages = torch.tensor(per_sample_num_stages, dtype=torch.float64)
    pca_values = [v for v in per_sample_pca99 if v is not None]
    pca_excluded = n - len(pca_values)
    pca_tensor = torch.tensor(pca_values, dtype=torch.float64) if pca_values else torch.tensor([], dtype=torch.float64)

    def _mean_std(t: torch.Tensor) -> dict:
        if t.numel() == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=False).item()),
        }

    return {
        "trajectory_length": _mean_std(lengths),
        "pca_99": _mean_std(pca_tensor),
        "num_stages": _mean_std(stages),
        "num_samples": n,
        "num_pca_excluded": pca_excluded,
    }
