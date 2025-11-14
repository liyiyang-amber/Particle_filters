"""Unit tests for kernel functions in Kernel Particle Filter."""

import numpy as np
import pytest
from models.kernel_particle_filter import (
    rbf_1d,
    scalar_kernel_full_matrix,
    matrix_kernel_and_divergence,
    gaspari_cohn,
    build_localization_matrix,
)


def test_rbf_1d_shape():
    """Test that 1D RBF kernel returns correct shapes."""
    d = np.array([0.0, 0.5, 1.0, 2.0])
    ell = 1.0

    K, dKdx = rbf_1d(d, ell)

    assert K.shape == d.shape
    assert dKdx.shape == d.shape


def test_rbf_1d_zero_distance():
    """Test that RBF kernel is 1 at zero distance."""
    d = np.array([0.0])
    ell = 1.0

    K, dKdx = rbf_1d(d, ell)

    assert np.isclose(K[0], 1.0)
    assert np.isclose(dKdx[0], 0.0)


def test_rbf_1d_positive_values():
    """Test that RBF kernel is always positive."""
    d = np.linspace(-5, 5, 100)
    ell = 1.0

    K, dKdx = rbf_1d(d, ell)

    assert np.all(K > 0)
    assert np.all(K <= 1.0)


def test_rbf_1d_decreasing_with_distance():
    """Test that RBF kernel decreases with distance."""
    d = np.array([0.0, 1.0, 2.0, 3.0])
    ell = 1.0

    K, dKdx = rbf_1d(d, ell)

    # Should be monotonically decreasing
    assert K[0] > K[1] > K[2] > K[3]


def test_rbf_1d_lengthscale_effect():
    """Test that larger lengthscale gives slower decay."""
    d = np.array([1.0])

    K_small, _ = rbf_1d(d, ell=0.5)
    K_large, _ = rbf_1d(d, ell=2.0)

    # Larger lengthscale should give larger kernel value at same distance
    assert K_large > K_small


def test_rbf_1d_derivative_sign():
    """Test that derivative has correct sign."""
    # For positive d, derivative should be negative
    d_pos = np.array([1.0])
    _, dKdx_pos = rbf_1d(d_pos, ell=1.0)
    assert dKdx_pos[0] < 0

    # For negative d, derivative should be positive
    d_neg = np.array([-1.0])
    _, dKdx_neg = rbf_1d(d_neg, ell=1.0)
    assert dKdx_neg[0] > 0


def test_scalar_kernel_full_matrix_shapes():
    """Test scalar kernel returns correct shapes."""
    n = 3
    Np = 10
    x = np.random.randn(n)
    ensemble = np.random.randn(Np, n)
    lengthscale = 1.0

    k, grad_k, divK = scalar_kernel_full_matrix(x, ensemble, lengthscale)

    assert k.shape == (Np,)
    assert grad_k.shape == (Np, n)
    assert divK.shape == (n,)


def test_scalar_kernel_identical_points():
    """Test scalar kernel when x equals ensemble member."""
    n = 3
    x = np.array([1.0, 2.0, 3.0])
    ensemble = np.array([x])  # Single particle identical to x
    lengthscale = 1.0

    k, grad_k, divK = scalar_kernel_full_matrix(x, ensemble, lengthscale)

    # Kernel should be 1 at zero distance
    assert np.isclose(k[0], 1.0)

    # Gradient should be zero
    np.testing.assert_allclose(grad_k[0], 0.0, atol=1e-10)


def test_scalar_kernel_positive_values():
    """Test that scalar kernel values are positive."""
    n = 3
    Np = 20
    x = np.random.randn(n)
    ensemble = np.random.randn(Np, n)
    lengthscale = 1.0

    k, grad_k, divK = scalar_kernel_full_matrix(x, ensemble, lengthscale)

    assert np.all(k > 0)
    assert np.all(k <= 1.0)


def test_matrix_kernel_and_divergence_shapes():
    """Test matrix kernel returns correct shapes."""
    n = 4
    Np = 15
    x = np.random.randn(n)
    ensemble = np.random.randn(Np, n)
    lengthscales = np.ones(n) * 1.0

    K_blocks, divK = matrix_kernel_and_divergence(x, ensemble, lengthscales)

    assert K_blocks.shape == (Np, n)
    assert divK.shape == (n,)


def test_matrix_kernel_identical_points():
    """Test matrix kernel when x equals ensemble member."""
    n = 3
    x = np.array([1.0, 2.0, 3.0])
    ensemble = np.array([x])
    lengthscales = np.ones(n) * 1.0

    K_blocks, divK = matrix_kernel_and_divergence(x, ensemble, lengthscales)

    # All diagonal entries should be 1
    np.testing.assert_allclose(K_blocks[0], 1.0)


def test_matrix_kernel_positive_values():
    """Test that matrix kernel values are positive."""
    n = 3
    Np = 20
    x = np.random.randn(n)
    ensemble = np.random.randn(Np, n)
    lengthscales = np.ones(n) * 1.0

    K_blocks, divK = matrix_kernel_and_divergence(x, ensemble, lengthscales)

    assert np.all(K_blocks > 0)
    assert np.all(K_blocks <= 1.0)


def test_matrix_kernel_different_lengthscales():
    """Test matrix kernel with different lengthscales per dimension."""
    n = 3
    x = np.array([0.0, 0.0, 0.0])
    ensemble = np.array([[1.0, 1.0, 1.0]])
    lengthscales = np.array([0.5, 1.0, 2.0])

    K_blocks, divK = matrix_kernel_and_divergence(x, ensemble, lengthscales)

    # Smaller lengthscale should give smaller kernel value
    assert K_blocks[0, 0] < K_blocks[0, 1] < K_blocks[0, 2]


def test_gaspari_cohn_shape():
    """Test Gaspari-Cohn function returns correct shape."""
    r = np.linspace(0, 3, 100)
    rho = gaspari_cohn(r)

    assert rho.shape == r.shape


def test_gaspari_cohn_compact_support():
    """Test that Gaspari-Cohn has compact support (zero for r > 2)."""
    r = np.array([0.0, 1.0, 2.0, 2.5, 3.0, 10.0])
    rho = gaspari_cohn(r)

    # Should be zero for r > 2
    assert np.isclose(rho[3], 0.0)
    assert np.isclose(rho[4], 0.0)
    assert np.isclose(rho[5], 0.0)


def test_gaspari_cohn_value_at_zero():
    """Test that Gaspari-Cohn is 1 at r=0."""
    r = np.array([0.0])
    rho = gaspari_cohn(r)

    assert np.isclose(rho[0], 1.0)


def test_gaspari_cohn_monotone_decreasing():
    """Test that Gaspari-Cohn is monotonically decreasing for r in [0, 2]."""
    r = np.linspace(0, 2, 100)
    rho = gaspari_cohn(r)

    # Should be decreasing
    assert np.all(np.diff(rho) <= 0)


def test_gaspari_cohn_in_range():
    """Test that Gaspari-Cohn values are in [0, 1]."""
    r = np.linspace(0, 5, 200)
    rho = gaspari_cohn(r)

    assert np.all(rho >= 0.0)
    assert np.all(rho <= 1.0)


def test_build_localization_matrix_shape():
    """Test localization matrix has correct shape."""
    n = 10
    radius = 5.0

    L = build_localization_matrix(n, radius)

    assert L.shape == (n, n)


def test_build_localization_matrix_infinite_radius():
    """Test that infinite radius gives all-ones matrix."""
    n = 5
    radius = np.inf

    L = build_localization_matrix(n, radius)

    np.testing.assert_array_equal(L, np.ones((n, n)))


def test_build_localization_matrix_symmetric():
    """Test that localization matrix is symmetric."""
    n = 10
    radius = 5.0

    L = build_localization_matrix(n, radius)

    np.testing.assert_allclose(L, L.T)


def test_build_localization_matrix_diagonal_ones():
    """Test that localization matrix has ones on diagonal."""
    n = 10
    radius = 1.0

    L = build_localization_matrix(n, radius)

    np.testing.assert_allclose(np.diag(L), 1.0)


def test_build_localization_matrix_compact_support():
    """Test that localization matrix has compact support."""
    n = 20
    radius = 2.0  # Will cutoff at distance 2*radius = 4 in normalized units

    L = build_localization_matrix(n, radius)

    # Elements far apart should be zero (beyond 2*radius in normalized distance)
    # Default metric is |i-j|, normalized by radius
    # So distance > 2*radius -> r > 2 -> rho = 0
    assert np.isclose(L[0, 10], 0.0)  # |0-10|/2 = 5 > 2


def test_build_localization_matrix_with_custom_metric():
    """Test localization matrix with custom distance metric."""
    n = 5
    radius = 1.0

    # Custom metric: all distances are 0.5
    metric = np.full((n, n), 0.5)
    np.fill_diagonal(metric, 0.0)

    L = build_localization_matrix(n, radius, metric=metric)

    # Off-diagonal should all have same value since all distances equal
    off_diag = L[0, 1]
    for i in range(n):
        for j in range(n):
            if i != j:
                assert np.isclose(L[i, j], off_diag)


def test_build_localization_matrix_values_in_range():
    """Test that localization matrix values are in [0, 1]."""
    n = 20
    radius = 5.0

    L = build_localization_matrix(n, radius)

    assert np.all(L >= 0.0)
    assert np.all(L <= 1.0)
