"""PCA reconstruction ablation (paper Section 5.3, Figure 5).

For each progressive sample we fit PCA on its own trajectory of stage
embeddings, then reconstruct the final converged embedding ``e*`` from the
top-``k`` principal components and measure teacher-forced reconstruction
accuracy on the original prefix. Repeating across a grid of ``k`` produces
an "accuracy vs #components" curve (paper Figure 5).

Paper's central observation (page 6): even though PCA 99 % of trajectory
*variance* requires only a few components (Table 13), reaching near-perfect
teacher-forced *accuracy* requires substantially more — i.e., the leftover
1 % of variance carries semantically important reconstruction information.
"""

from __future__ import annotations

import torch


def fit_per_sample_pca(
    stage_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit PCA on a single sample's trajectory.

    ``stage_embeddings``: [n_stages, ...] (trailing shape preserved through flatten).

    Returns ``(mean, components, singular_values)`` where:
        - ``mean`` has shape [flat_dim] (mean of flattened stage embeddings);
        - ``components`` has shape [r, flat_dim] with ``r = min(n_stages-1, flat_dim)``;
          each row is one principal direction (in descending variance order);
        - ``singular_values`` has shape [r] — singular values of the centered
          matrix; squared values divided by their sum give variance ratios.
    """
    if stage_embeddings.dim() < 2:
        raise ValueError(f"Expected stage_embeddings of shape [n_stages, ...], got {tuple(stage_embeddings.shape)}")
    n_stages = stage_embeddings.shape[0]
    if n_stages < 2:
        raise ValueError(f"Need >= 2 stages for PCA, got {n_stages}")
    flat = stage_embeddings.reshape(n_stages, -1).to(torch.float64)
    mean = flat.mean(dim=0)
    centered = flat - mean
    _, singular, vt = torch.linalg.svd(centered, full_matrices=False)
    return mean, vt, singular


def cumulative_variance_ratio(singular_values: torch.Tensor) -> torch.Tensor:
    """Per-component cumulative explained variance ratio (so element i = variance covered by top-(i+1))."""
    variance = singular_values.to(torch.float64) ** 2
    total = float(variance.sum().item())
    if total <= 0.0:
        return torch.zeros_like(variance)
    return torch.cumsum(variance, dim=0) / total


def project_top_k(
    target: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Project ``(target - mean)`` onto the top-k principal directions, then add ``mean`` back.

    ``target``: arbitrary trailing shape; flattened and unflattened around the projection.
    ``mean``: [flat_dim].
    ``components``: [r, flat_dim] (rows are principal directions).
    ``k``: number of components to keep (clamped to ``r``).

    Returns a tensor with the same shape as ``target`` (float64 — caller should cast).
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    target_shape = target.shape
    target_flat = target.reshape(-1).to(torch.float64)
    centered = target_flat - mean
    if k == 0:
        return mean.reshape(target_shape)
    k_eff = min(k, components.shape[0])
    v_k = components[:k_eff]  # [k_eff, flat_dim]
    coeffs = v_k @ centered  # [k_eff]
    reconstructed = mean + v_k.T @ coeffs
    return reconstructed.reshape(target_shape)


def summarize_pca_curve(per_sample_curves: list[dict]) -> dict:
    """Aggregate per-sample (k, accuracy, variance_ratio) points into mean ± std per k.

    Each per-sample entry must contain ``curve`` — a list of dicts with at
    least ``{"k": int, "accuracy": float}`` and optionally ``"variance_ratio"``.
    Different samples may have different k-grids (e.g., the per-sample ``max_k``
    rank limit varies with n_stages); we aggregate every k that appears in at
    least one sample's curve.
    """
    accuracy_by_k: dict[int, list[float]] = {}
    variance_by_k: dict[int, list[float]] = {}
    for sample in per_sample_curves:
        for point in sample.get("curve", []):
            k = int(point["k"])
            accuracy_by_k.setdefault(k, []).append(float(point["accuracy"]))
            if "variance_ratio" in point and point["variance_ratio"] is not None:
                variance_by_k.setdefault(k, []).append(float(point["variance_ratio"]))

    curve = []
    for k in sorted(accuracy_by_k.keys()):
        acc_values = torch.tensor(accuracy_by_k[k], dtype=torch.float64)
        entry = {
            "k": k,
            "mean": float(acc_values.mean().item()),
            "std": float(acc_values.std(unbiased=False).item()),
            "n_samples": int(acc_values.numel()),
        }
        if k in variance_by_k:
            var_values = torch.tensor(variance_by_k[k], dtype=torch.float64)
            entry["variance_ratio_mean"] = float(var_values.mean().item())
            entry["variance_ratio_std"] = float(var_values.std(unbiased=False).item())
        curve.append(entry)

    return {
        "curve": curve,
        "num_samples": len(per_sample_curves),
    }
