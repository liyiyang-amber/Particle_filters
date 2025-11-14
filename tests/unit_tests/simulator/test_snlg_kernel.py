import numpy as np
import pytest
from simulator.simulator_sensor_network_linear_gaussian import (
    make_grid_coords,
    se_kernel_cov,
)


def test_se_kernel_shape():
    """Test that SE kernel returns a square matrix of correct size."""
    coords = make_grid_coords(16)
    cov = se_kernel_cov(coords, alpha0=3.0, beta=20.0, alpha1=0.01)
    assert cov.shape == (16, 16)


def test_se_kernel_shape_large():
    """Test SE kernel with a larger grid."""
    coords = make_grid_coords(64)
    cov = se_kernel_cov(coords, alpha0=3.0, beta=20.0, alpha1=0.01)
    assert cov.shape == (64, 64)


def test_se_kernel_symmetric():
    """Test that the covariance matrix is symmetric."""
    coords = make_grid_coords(16)
    cov = se_kernel_cov(coords, alpha0=3.0, beta=20.0, alpha1=0.01)
    assert np.allclose(cov, cov.T)


def test_se_kernel_positive_definite():
    """Test that the covariance matrix is positive definite."""
    coords = make_grid_coords(16)
    cov = se_kernel_cov(coords, alpha0=3.0, beta=20.0, alpha1=0.01)
    
    # Check that all eigenvalues are positive
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0)


def test_se_kernel_diagonal_values():
    """Test that diagonal values equal alpha0 + alpha1."""
    coords = make_grid_coords(16)
    alpha0, alpha1 = 3.0, 0.01
    cov = se_kernel_cov(coords, alpha0=alpha0, beta=20.0, alpha1=alpha1)
    
    # Diagonal should be alpha0 + alpha1 (distance to self is 0)
    expected_diag = alpha0 + alpha1
    assert np.allclose(np.diag(cov), expected_diag)


def test_se_kernel_nugget_effect():
    """Test that alpha1 affects only diagonal."""
    coords = make_grid_coords(9)  # 3x3 grid
    
    # Compute with and without nugget
    cov_no_nugget = se_kernel_cov(coords, alpha0=3.0, beta=20.0, alpha1=0.0)
    cov_with_nugget = se_kernel_cov(coords, alpha0=3.0, beta=20.0, alpha1=0.5)
    
    # Off-diagonal should be the same
    mask = ~np.eye(9, dtype=bool)
    assert np.allclose(cov_no_nugget[mask], cov_with_nugget[mask])
    
    # Diagonal should differ by exactly alpha1
    assert np.allclose(np.diag(cov_with_nugget) - np.diag(cov_no_nugget), 0.5)


def test_se_kernel_decay_with_distance():
    """Test that covariance decays with distance."""
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [5.0, 0.0],
    ])
    
    cov = se_kernel_cov(coords, alpha0=3.0, beta=10.0, alpha1=0.01)
    
    # Point 0 to point 1 (distance 1)
    val_dist1 = cov[0, 1]
    
    # Point 0 to point 2 (distance 2)
    val_dist2 = cov[0, 2]
    
    # Point 0 to point 3 (distance 5)
    val_dist5 = cov[0, 3]
    
    # Covariance should decay with distance
    assert val_dist1 > val_dist2 > val_dist5
    assert val_dist5 > 0  # But still positive


def test_se_kernel_alpha0_scales_amplitude():
    """Test that alpha0 scales the covariance amplitude."""
    coords = make_grid_coords(9)
    
    cov1 = se_kernel_cov(coords, alpha0=1.0, beta=20.0, alpha1=0.0)
    cov2 = se_kernel_cov(coords, alpha0=2.0, beta=20.0, alpha1=0.0)
    
    # cov2 should be approximately 2 * cov1
    assert np.allclose(cov2, 2.0 * cov1)


def test_se_kernel_beta_affects_lengthscale():
    """Test that beta affects the lengthscale (correlation range)."""
    coords = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
    ])
    
    # Small beta = short lengthscale = fast decay
    cov_short = se_kernel_cov(coords, alpha0=3.0, beta=1.0, alpha1=0.0)
    
    # Large beta = long lengthscale = slow decay
    cov_long = se_kernel_cov(coords, alpha0=3.0, beta=100.0, alpha1=0.0)
    
    # With larger beta, correlation at distance 2 should be higher
    assert cov_long[0, 1] > cov_short[0, 1]


def test_se_kernel_identical_points():
    """Test covariance between identical points (should be alpha0 + alpha1)."""
    coords = np.array([[1.5, 2.5]])
    cov = se_kernel_cov(coords, alpha0=3.0, beta=20.0, alpha1=0.01)
    
    assert cov.shape == (1, 1)
    assert np.allclose(cov[0, 0], 3.01)


def test_se_kernel_formula_verification():
    """Test the SE kernel formula explicitly for a simple case."""
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    
    alpha0, beta, alpha1 = 2.0, 8.0, 0.1
    cov = se_kernel_cov(coords, alpha0=alpha0, beta=beta, alpha1=alpha1)
    
    # Distance squared between points is 1.0
    # Expected off-diagonal: alpha0 * exp(-1.0 / 8.0)
    expected_off_diag = alpha0 * np.exp(-1.0 / beta)
    assert np.allclose(cov[0, 1], expected_off_diag)
    assert np.allclose(cov[1, 0], expected_off_diag)
    
    # Diagonal should be alpha0 + alpha1
    assert np.allclose(cov[0, 0], alpha0 + alpha1)
    assert np.allclose(cov[1, 1], alpha0 + alpha1)


def test_se_kernel_zero_alpha1():
    """Test that zero nugget still produces a valid but possibly ill-conditioned matrix."""
    coords = make_grid_coords(4)
    cov = se_kernel_cov(coords, alpha0=3.0, beta=20.0, alpha1=0.0)
    
    # Should still be symmetric and have correct shape
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)
    
    # Eigenvalues should still be positive (though smallest may be very small)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= 0)


def test_se_kernel_large_beta_limit():
    """Test that very large beta leads to nearly constant covariance."""
    coords = make_grid_coords(9)
    cov = se_kernel_cov(coords, alpha0=3.0, beta=1e6, alpha1=0.01)
    
    # With very large beta, all off-diagonal entries should be close to alpha0
    mask = ~np.eye(9, dtype=bool)
    assert np.allclose(cov[mask], 3.0, atol=0.01)


def test_se_kernel_small_beta_limit():
    """Test that very small beta leads to nearly diagonal covariance."""
    coords = make_grid_coords(9)
    cov = se_kernel_cov(coords, alpha0=3.0, beta=1e-6, alpha1=0.01)
    
    # With very small beta, off-diagonal entries should be nearly zero
    mask = ~np.eye(9, dtype=bool)
    assert np.max(np.abs(cov[mask])) < 0.01
