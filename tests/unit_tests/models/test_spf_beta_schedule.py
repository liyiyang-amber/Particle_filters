"""Unit tests for SPF beta schedule computation."""

import numpy as np
import pytest
from models.Stochastic_particle_filter import solve_beta_star_bisection, LinearGaussianBayes


@pytest.fixture
def simple_M_matrices():
    """Simple M0 and Mh matrices for beta schedule testing."""
    M0 = np.array([[2.0, 0.1], [0.1, 1.5]])
    Mh = np.array([[0.5, 0.05], [0.05, 0.3]])
    return M0, Mh


def test_beta_schedule_linear_equivalent():
    """Test that linear schedule (mu=0) produces linear beta."""
    M0 = np.eye(2)
    Mh = np.eye(2) * 0.5
    
    # With mu=0, ODE becomes beta'' = 0, so beta(λ) = λ
    lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=0.0, n_grid=101)
    
    # Check endpoints
    assert np.isclose(beta[0], 0.0, atol=1e-10)
    assert np.isclose(beta[-1], 1.0, atol=1e-10)
    
    # For mu=0, beta should be approximately linear
    beta_linear = lam
    np.testing.assert_allclose(beta, beta_linear, atol=1e-3)


def test_beta_schedule_endpoints(simple_M_matrices):
    """Test that beta schedule satisfies boundary conditions."""
    M0, Mh = simple_M_matrices
    
    lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=1e-2, n_grid=201)
    
    # Boundary conditions: beta(0) = 0, beta(1) = 1
    assert np.isclose(beta[0], 0.0, atol=1e-10)
    assert np.isclose(beta[-1], 1.0, atol=1e-10)


def test_beta_schedule_monotonicity(simple_M_matrices):
    """Test that beta is monotonically increasing."""
    M0, Mh = simple_M_matrices
    
    lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=1e-2, n_grid=201)
    
    # beta should be non-decreasing
    diffs = np.diff(beta)
    assert np.all(diffs >= -1e-10), "Beta should be monotonically increasing"


def test_beta_schedule_bounded(simple_M_matrices):
    """Test that beta stays in [0, 1]."""
    M0, Mh = simple_M_matrices
    
    lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=1e-2, n_grid=201)
    
    # beta should stay in [0, 1]
    assert np.all(beta >= -1e-10), "Beta should be >= 0"
    assert np.all(beta <= 1.0 + 1e-10), "Beta should be <= 1"


def test_beta_schedule_different_mu_values(simple_M_matrices):
    """Test beta schedules with different mu values."""
    M0, Mh = simple_M_matrices
    
    mu_values = [1e-4, 1e-3, 1e-2, 1e-1]
    
    for mu in mu_values:
        lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=mu, n_grid=201)
        
        # Check boundary conditions
        assert np.isclose(beta[0], 0.0, atol=1e-6)
        assert np.isclose(beta[-1], 1.0, atol=1e-6)
        
        # Check monotonicity
        assert np.all(np.diff(beta) >= -1e-6)


def test_beta_schedule_grid_sizes(simple_M_matrices):
    """Test that different grid sizes produce consistent results."""
    M0, Mh = simple_M_matrices
    mu = 1e-2
    
    grid_sizes = [101, 201, 301]
    beta_results = []
    
    for n_grid in grid_sizes:
        lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=mu, n_grid=n_grid)
        beta_results.append(beta)
        
        # Check boundary conditions
        assert np.isclose(beta[0], 0.0, atol=1e-6)
        assert np.isclose(beta[-1], 1.0, atol=1e-6)
    
    # Results should be qualitatively similar (check at midpoint)
    mid_values = [beta[len(beta)//2] for beta in beta_results]
    # All mid values should be reasonably close
    assert np.std(mid_values) < 0.1


def test_beta_schedule_derivative_consistency(simple_M_matrices):
    """Test that betadot is consistent with numerical derivative of beta."""
    M0, Mh = simple_M_matrices
    
    lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=1e-2, n_grid=201)
    
    # Compute numerical derivative
    dlam = lam[1] - lam[0]
    betadot_numerical = np.gradient(beta, dlam)
    
    # Should be reasonably close (not exact due to gradient approximation)
    np.testing.assert_allclose(betadot, betadot_numerical, rtol=0.1, atol=0.1)


def test_beta_schedule_with_diagonal_matrices():
    """Test with simple diagonal M matrices."""
    M0 = np.diag([2.0, 1.5, 1.0])
    Mh = np.diag([0.8, 0.5, 0.3])
    
    lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=5e-3, n_grid=151)
    
    assert np.isclose(beta[0], 0.0, atol=1e-6)
    assert np.isclose(beta[-1], 1.0, atol=1e-6)
    assert np.all(beta >= -1e-6)
    assert np.all(beta <= 1.0 + 1e-6)


def test_beta_schedule_with_lg_model():
    """Test beta schedule using LinearGaussianBayes model."""
    model = LinearGaussianBayes(
        m0=np.array([1.0, 2.0]),
        P0=np.array([[2.0, 0.5], [0.5, 1.0]]),
        H=np.array([[1.0, 0.5]]),
        R=np.array([[0.5]]),
        z=np.array([3.0])
    )
    
    lam, beta, betadot = solve_beta_star_bisection(model.M0, model.Mh, mu=1e-2, n_grid=201)
    
    assert np.isclose(beta[0], 0.0, atol=1e-6)
    assert np.isclose(beta[-1], 1.0, atol=1e-6)


def test_beta_schedule_symmetrization():
    """Test that asymmetric matrices are handled correctly."""
    # Slightly asymmetric matrices
    M0 = np.array([[2.0, 0.15], [0.1, 1.5]])
    Mh = np.array([[0.5, 0.06], [0.05, 0.3]])
    
    # Should not raise error
    lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=1e-2, n_grid=101)
    
    assert np.isclose(beta[0], 0.0, atol=1e-6)
    assert np.isclose(beta[-1], 1.0, atol=1e-6)


def test_beta_schedule_small_mu():
    """Test with very small mu (nearly linear)."""
    M0 = np.eye(2)
    Mh = np.eye(2) * 0.5
    
    lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=1e-6, n_grid=101)
    
    # With very small mu, should be close to linear
    np.testing.assert_allclose(beta, lam, atol=5e-2)


def test_beta_schedule_large_mu():
    """Test with larger mu value."""
    M0 = np.eye(2)
    Mh = np.eye(2) * 0.5
    
    lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=0.1, n_grid=201)
    
    # Should still satisfy boundary conditions
    assert np.isclose(beta[0], 0.0, atol=1e-6)
    assert np.isclose(beta[-1], 1.0, atol=1e-6)
    
    # For this simple case with uniform eigenvalues, deviation might be small
    # Just check that all values are finite and in range
    assert np.all(np.isfinite(beta))
    assert np.all(beta >= -1e-6)
    assert np.all(beta <= 1.0 + 1e-6)


def test_beta_schedule_initial_slope():
    """Test that initial slope beta'(0) is found correctly."""
    M0 = np.eye(2)
    Mh = np.eye(2) * 0.5
    
    lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=1e-2, n_grid=201)
    
    # Initial slope should be finite and reasonable
    assert np.isfinite(betadot[0])
    # For this simple case, should be close to 1
    assert 0.5 < betadot[0] < 2.0


def test_beta_schedule_reproducibility(simple_M_matrices):
    """Test that solving twice gives same result."""
    M0, Mh = simple_M_matrices
    
    lam1, beta1, betadot1 = solve_beta_star_bisection(M0, Mh, mu=1e-2, n_grid=201)
    lam2, beta2, betadot2 = solve_beta_star_bisection(M0, Mh, mu=1e-2, n_grid=201)
    
    np.testing.assert_array_equal(lam1, lam2)
    np.testing.assert_allclose(beta1, beta2, rtol=1e-10)
    np.testing.assert_allclose(betadot1, betadot2, rtol=1e-10)


def test_beta_schedule_bracket_expansion():
    """Test that bracket expansion works when needed."""
    # Use parameters that might need bracket expansion
    M0 = np.diag([10.0, 1.0])
    Mh = np.diag([5.0, 0.5])
    
    # Should not raise RuntimeError
    lam, beta, betadot = solve_beta_star_bisection(
        M0, Mh, mu=1e-2, n_grid=201, s_lo=-5.0, s_hi=5.0
    )
    
    assert np.isclose(beta[0], 0.0, atol=1e-6)
    assert np.isclose(beta[-1], 1.0, atol=1e-6)


def test_beta_schedule_different_dimensions():
    """Test beta schedule with different state dimensions."""
    for n in [1, 2, 3, 5]:
        M0 = np.eye(n) * 2.0
        Mh = np.eye(n) * 0.5
        
        lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=1e-2, n_grid=101)
        
        assert len(lam) == 101
        assert len(beta) == 101
        assert len(betadot) == 101
        assert np.isclose(beta[0], 0.0, atol=1e-6)
        assert np.isclose(beta[-1], 1.0, atol=1e-6)


def test_beta_schedule_values_in_range():
    """Test that all lambda, beta, betadot values are reasonable."""
    M0 = np.eye(3)
    Mh = np.eye(3) * 0.5
    
    lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=1e-2, n_grid=201)
    
    # Lambda should be in [0, 1]
    assert np.all(lam >= 0.0)
    assert np.all(lam <= 1.0)
    
    # Beta should be in [0, 1]
    assert np.all(beta >= -1e-6)
    assert np.all(beta <= 1.0 + 1e-6)
    
    # Betadot should be finite
    assert np.all(np.isfinite(betadot))


def test_beta_schedule_increasing_complexity():
    """Test with increasingly complex M matrices."""
    rng = np.random.default_rng(42)
    
    for _ in range(5):
        n = 2
        # Create random SPD matrices
        A = rng.standard_normal((n, n))
        M0 = A.T @ A + np.eye(n)
        B = rng.standard_normal((n, n))
        Mh = 0.5 * (B.T @ B + np.eye(n) * 0.1)
        
        lam, beta, betadot = solve_beta_star_bisection(M0, Mh, mu=1e-2, n_grid=151)
        
        assert np.isclose(beta[0], 0.0, atol=1e-5)
        assert np.isclose(beta[-1], 1.0, atol=1e-5)
        assert np.all(np.diff(beta) >= -1e-6)
