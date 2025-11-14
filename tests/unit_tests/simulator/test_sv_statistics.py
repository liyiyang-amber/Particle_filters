import numpy as np
import pytest
from simulator.simulator_sto_volatility_model import simulate_sv_1d


def _acf1(x):
    """Compute lag-1 autocorrelation."""
    x = x - x.mean()
    c0 = np.dot(x, x) / x.size
    c1 = np.dot(x[:-1], x[1:]) / (x.size - 1)
    return c1 / c0


def test_stationary_variance_matches_theory():
    """Test that empirical variance of X matches theoretical stationary variance."""
    n = 10000  # Large for law of large numbers
    alpha = 0.9
    sigma = 0.2
    
    res = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=1.0, seed=7)
    
    # Theoretical stationary variance: σ² / (1 - α²)
    theor_var = sigma**2 / (1 - alpha**2)
    emp_var = res.X.var(ddof=0)
    
    np.testing.assert_allclose(emp_var, theor_var, rtol=0.08, atol=0.0)


def test_acf1_matches_alpha():
    """Test that lag-1 autocorrelation of X matches alpha."""
    n = 20000
    alpha = 0.9
    sigma = 0.2
    
    res = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=1.0, seed=11)
    
    acf1 = _acf1(res.X)
    np.testing.assert_allclose(acf1, alpha, rtol=0.05, atol=0.02)


def test_alpha_zero_iid_latents():
    """Test that alpha=0 produces i.i.d. latent states."""
    n = 12000
    alpha = 0.0
    sigma = 0.3
    
    res = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=1.0, seed=9)
    
    # With alpha=0, X_t = sigma * V_t, so var(X_t) = sigma²
    emp_var = res.X.var(ddof=0)
    np.testing.assert_allclose(emp_var, sigma**2, rtol=0.05, atol=0.0)
    
    # ACF should be near zero
    acf1 = _acf1(res.X)
    assert abs(acf1) < 0.05


def test_initial_state_stationary_distribution():
    """Test that x0=None initializes from stationary distribution."""
    n = 10000
    alpha = 0.85
    sigma = 0.25
    
    # Run multiple simulations and check initial state distribution
    x0_samples = []
    for seed in range(100):
        res = simulate_sv_1d(n=10, alpha=alpha, sigma=sigma, beta=1.0, seed=seed, x0=None)
        x0_samples.append(res.X[0])
    
    x0_samples = np.array(x0_samples)
    
    # Theoretical variance of initial state
    theor_var = sigma**2 / (1 - alpha**2)
    emp_var = x0_samples.var(ddof=1)
    
    # Should be close (with some Monte Carlo error)
    np.testing.assert_allclose(emp_var, theor_var, rtol=0.2, atol=0.0)
