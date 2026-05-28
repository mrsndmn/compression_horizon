"""Unit tests for analysis/dimensionality.py (Two-NN + 2-D projections)."""

import numpy as np
import pytest

from compression_horizon.analysis import (
    estimate_twonn,
    plane_grid,
    project_2d,
    reconstruct_from_plane,
)


def test_twonn_recovers_linear_subspace_dimension():
    """A 5-D Gaussian cloud linearly embedded in 50-D should read as ~5-D."""
    rng = np.random.default_rng(0)
    intrinsic = 5
    n_points = 2000
    latent = rng.standard_normal((n_points, intrinsic))
    basis = rng.standard_normal((intrinsic, 50))
    points = latent @ basis

    estimate = estimate_twonn(points)

    assert estimate["n_points"] == n_points
    assert abs(estimate["intrinsic_dim"] - intrinsic) < 1.0


def test_twonn_orders_dimensions():
    """Two-NN must rank a 2-D cloud below a 10-D cloud."""
    rng = np.random.default_rng(1)
    low = estimate_twonn(rng.standard_normal((1500, 2)))["intrinsic_dim"]
    high = estimate_twonn(rng.standard_normal((1500, 10)))["intrinsic_dim"]
    assert low < high


def test_twonn_requires_enough_points():
    with pytest.raises(ValueError):
        estimate_twonn(np.zeros((4, 3)))


def test_twonn_rejects_bad_discard_fraction():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        estimate_twonn(rng.standard_normal((100, 4)), discard_fraction=1.0)


def test_project_2d_shapes():
    rng = np.random.default_rng(3)
    points = rng.standard_normal((120, 16))
    for method in ("pca", "tsne"):
        coords = project_2d(points, method=method)
        assert coords.shape == (120, 2)


def test_project_2d_is_deterministic_for_pca():
    rng = np.random.default_rng(4)
    points = rng.standard_normal((80, 12))
    first = project_2d(points, method="pca", seed=7)
    second = project_2d(points, method="pca", seed=7)
    np.testing.assert_allclose(first, second)


def test_project_2d_rejects_unknown_method():
    with pytest.raises(ValueError):
        project_2d(np.zeros((10, 3)), method="isomap")


def test_plane_grid_covers_point_cloud():
    rng = np.random.default_rng(5)
    coords = rng.standard_normal((50, 2))
    grid, (xmin, xmax, ymin, ymax) = plane_grid(coords, resolution=20, margin=0.1)
    assert grid.shape == (400, 2)
    assert xmin < coords[:, 0].min() and xmax > coords[:, 0].max()
    assert ymin < coords[:, 1].min() and ymax > coords[:, 1].max()


def test_plane_grid_rejects_bad_shape():
    with pytest.raises(ValueError):
        plane_grid(np.zeros((10, 3)))


def test_reconstruct_from_plane_is_projection_inverse():
    """Lifting plane coords through an orthonormal basis then projecting back is identity."""
    rng = np.random.default_rng(6)
    dim = 40
    raw = rng.standard_normal((2, dim))
    basis_2, _ = np.linalg.qr(raw.T)  # [dim, 2] orthonormal columns
    basis_2 = basis_2.T  # [2, dim]
    mean = rng.standard_normal(dim)
    coords = rng.standard_normal((30, 2))

    lifted = reconstruct_from_plane(coords, mean, basis_2)
    assert lifted.shape == (30, dim)
    projected_back = (lifted - mean) @ basis_2.T
    np.testing.assert_allclose(projected_back, coords, atol=1e-9)


def test_reconstruct_from_plane_rejects_mismatched_basis():
    with pytest.raises(ValueError):
        reconstruct_from_plane(np.zeros((5, 2)), mean=np.zeros(10), basis_2=np.zeros((2, 8)))
