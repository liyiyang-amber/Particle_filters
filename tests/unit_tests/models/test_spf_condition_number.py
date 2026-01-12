"""Unit tests for SPF condition number calculations."""

import numpy as np
import pytest
from models.Stochastic_particle_filter import kappa2_and_derivative


def test_kappa2_identity_matrix():
    """Test condition number of identity matrix."""
    n = 3
    M = np.eye(n)
    dM = np.zeros((n, n))
    
    kappa, dkappa = kappa2_and_derivative(M, dM)
    
    # Identity has condition number 1
    assert np.isclose(kappa, 1.0, rtol=1e-6)
    # Derivative should be 0 since dM = 0
    assert np.isclose(dkappa, 0.0, atol=1e-10)


def test_kappa2_diagonal_matrix():
    """Test condition number of diagonal matrix."""
    M = np.diag([1.0, 2.0, 4.0])
    dM = np.diag([0.1, 0.2, 0.3])
    
    kappa, dkappa = kappa2_and_derivative(M, dM)
    
    # For diagonal: kappa = max/min = 4/1 = 4
    assert np.isclose(kappa, 4.0, rtol=1e-6)
    
    # dkappa = (dlambda_max/lambda_min) - (lambda_max * dlambda_min / lambda_min^2)
    # dlambda_max = 0.3, dlambda_min = 0.1
    # dkappa = 0.3/1 - 4*0.1/1 = 0.3 - 0.4 = -0.1
    expected_dkappa = 0.3 / 1.0 - 4.0 * 0.1 / (1.0 ** 2)
    assert np.isclose(dkappa, expected_dkappa, rtol=1e-6)


def test_kappa2_symmetric_matrix():
    """Test with a generic symmetric matrix."""
    M = np.array([[2.0, 0.5], [0.5, 1.0]])
    dM = np.array([[0.1, 0.05], [0.05, 0.2]])
    
    kappa, dkappa = kappa2_and_derivative(M, dM)
    
    # Verify kappa is positive
    assert kappa > 0
    
    # For SPD matrix, kappa >= 1
    assert kappa >= 1.0 - 1e-6
    
    # Check that derivative is finite
    assert np.isfinite(dkappa)


def test_kappa2_well_conditioned():
    """Test a well-conditioned matrix (kappa close to 1)."""
    M = np.eye(4) * 1.5  # All eigenvalues = 1.5
    dM = np.eye(4) * 0.1
    
    kappa, dkappa = kappa2_and_derivative(M, dM)
    
    # Scaled identity has condition number 1
    assert np.isclose(kappa, 1.0, rtol=1e-6)


def test_kappa2_ill_conditioned():
    """Test an ill-conditioned matrix (large kappa)."""
    M = np.diag([100.0, 1.0])
    dM = np.diag([1.0, 1.0])
    
    kappa, dkappa = kappa2_and_derivative(M, dM)
    
    # kappa = 100/1 = 100
    assert np.isclose(kappa, 100.0, rtol=1e-6)
    
    # dkappa = 1/1 - 100*1/1 = 1 - 100 = -99
    expected_dkappa = 1.0 / 1.0 - 100.0 * 1.0 / (1.0 ** 2)
    assert np.isclose(dkappa, expected_dkappa, rtol=1e-6)


def test_kappa2_symmetrization():
    """Test that asymmetric matrices are symmetrized."""
    M = np.array([[2.0, 0.6], [0.5, 1.0]])  # Slightly asymmetric
    dM = np.array([[0.1, 0.06], [0.05, 0.2]])
    
    kappa, dkappa = kappa2_and_derivative(M, dM)
    
    # Should not raise error and return valid results
    assert kappa > 0
    assert np.isfinite(kappa)
    assert np.isfinite(dkappa)


def test_kappa2_zero_derivative():
    """Test when derivative matrix is zero."""
    M = np.diag([2.0, 1.0])
    dM = np.zeros((2, 2))
    
    kappa, dkappa = kappa2_and_derivative(M, dM)
    
    # kappa = 2/1 = 2
    assert np.isclose(kappa, 2.0, rtol=1e-6)
    # dkappa should be 0 since dM = 0
    assert np.isclose(dkappa, 0.0, atol=1e-10)


def test_kappa2_stability_with_regularization():
    """Test that regularization prevents numerical issues."""
    # Create a nearly singular matrix
    M = np.array([[1e-10, 0.0], [0.0, 1.0]])
    dM = np.array([[1e-11, 0.0], [0.0, 0.1]])
    
    kappa, dkappa = kappa2_and_derivative(M, dM)
    
    # Should return large but finite condition number
    assert kappa > 1e6
    assert np.isfinite(kappa)
    assert np.isfinite(dkappa)


def test_kappa2_random_spd_matrices():
    """Test with random SPD matrices."""
    rng = np.random.default_rng(42)
    
    for _ in range(10):
        n = rng.integers(2, 6)
        
        # Create random SPD matrix
        A = rng.standard_normal((n, n))
        M = A.T @ A + np.eye(n)  # Ensures SPD
        
        # Create random symmetric dM
        B = rng.standard_normal((n, n))
        dM = 0.5 * (B + B.T)
        
        kappa, dkappa = kappa2_and_derivative(M, dM)
        
        assert kappa >= 1.0 - 1e-6, "Condition number should be >= 1"
        assert np.isfinite(kappa)
        assert np.isfinite(dkappa)


def test_kappa2_increasing_condition():
    """Test matrices with increasing condition numbers."""
    for scale in [1.0, 10.0, 100.0, 1000.0]:
        M = np.diag([scale, 1.0])
        dM = np.eye(2)
        
        kappa, dkappa = kappa2_and_derivative(M, dM)
        
        # kappa should equal scale
        assert np.isclose(kappa, scale, rtol=1e-6)


def test_kappa2_3d_case():
    """Test 3D matrix case."""
    M = np.diag([5.0, 2.0, 1.0])
    dM = np.diag([0.5, 0.2, 0.1])
    
    kappa, dkappa = kappa2_and_derivative(M, dM)
    
    # kappa = 5/1 = 5
    assert np.isclose(kappa, 5.0, rtol=1e-6)
    
    # dkappa = 0.5/1 - 5*0.1/1 = 0.5 - 0.5 = 0
    expected_dkappa = 0.5 / 1.0 - 5.0 * 0.1 / (1.0 ** 2)
    assert np.isclose(dkappa, expected_dkappa, rtol=1e-6)


def test_kappa2_numerical_stability():
    """Test numerical stability with various matrix scales."""
    scales = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    
    for scale in scales:
        M = np.eye(3) * scale
        dM = np.eye(3) * (scale * 0.01)
        
        kappa, dkappa = kappa2_and_derivative(M, dM)
        
        # Scaled identity should have kappa close to 1
        assert np.isclose(kappa, 1.0, rtol=1e-3)
        assert np.isfinite(kappa)
        assert np.isfinite(dkappa)


def test_kappa2_positive_definite_check():
    """Verify that M with negative eigenvalues is handled."""
    # This shouldn't happen in practice, but test robustness
    M = np.diag([-1.0, 1.0])  # One negative eigenvalue
    dM = np.eye(2)
    
    kappa, dkappa = kappa2_and_derivative(M, dM)
    
    # Should still return finite values due to abs() in implementation
    assert np.isfinite(kappa)
    assert np.isfinite(dkappa)


def test_kappa2_derivative_sign_change():
    """Test cases where derivative changes sign."""
    # Case where dkappa should be negative
    M = np.diag([100.0, 1.0])
    dM = np.diag([0.1, 1.0])  # Smaller eigenvalue changes more
    
    kappa, dkappa = kappa2_and_derivative(M, dM)
    
    # dkappa = 0.1/1 - 100*1/1 = 0.1 - 100 = -99.9
    expected_dkappa = 0.1 / 1.0 - 100.0 * 1.0 / (1.0 ** 2)
    assert dkappa < 0, "Derivative should be negative"
    assert np.isclose(dkappa, expected_dkappa, rtol=1e-6)
    
    # Case where dkappa should be positive
    M2 = np.diag([100.0, 1.0])
    dM2 = np.diag([10.0, 0.1])  # Larger eigenvalue changes more
    
    kappa2, dkappa2 = kappa2_and_derivative(M2, dM2)
    
    # dkappa = 10/1 - 100*0.1/1 = 10 - 10 = 0
    expected_dkappa2 = 10.0 / 1.0 - 100.0 * 0.1 / (1.0 ** 2)
    assert np.isclose(dkappa2, expected_dkappa2, rtol=1e-6)


def test_kappa2_extreme_conditioning():
    """Test with extremely ill-conditioned matrix."""
    M = np.diag([1e6, 1.0])
    dM = np.diag([1e4, 1.0])
    
    kappa, dkappa = kappa2_and_derivative(M, dM)
    
    # kappa should be around 1e6
    assert kappa > 1e5
    assert np.isfinite(kappa)
    assert np.isfinite(dkappa)
