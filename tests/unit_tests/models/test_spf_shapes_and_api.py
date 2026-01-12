"""Unit tests for SPF main function: shapes, API, and basic functionality."""

import numpy as np
import pytest
from models.Stochastic_particle_filter import (
    run_generalized_spf,
    LinearGaussianBayes,
)


@pytest.fixture
def simple_lg_model():
    """Simple 2D linear-Gaussian model for testing."""
    m0 = np.array([1.0, 2.0])
    P0 = np.array([[2.0, 0.5], [0.5, 1.0]])
    H = np.array([[1.0, 0.5]])
    R = np.array([[0.5]])
    z = np.array([3.0])
    return LinearGaussianBayes(m0=m0, P0=P0, H=H, R=R, z=z)


@pytest.fixture
def simple_1d_model():
    """Simple 1D linear-Gaussian model."""
    m0 = np.array([5.0])
    P0 = np.array([[2.0]])
    H = np.array([[1.0]])
    R = np.array([[1.0]])
    z = np.array([7.0])
    return LinearGaussianBayes(m0=m0, P0=P0, H=H, R=R, z=z)


def test_spf_basic_run(simple_lg_model):
    """Test basic SPF execution with default parameters."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=1000,
        n_steps=100,
        beta_mode="linear",
        seed=42
    )
    
    # Check output shapes
    assert X.shape == (1000, 2)
    assert x_hat.shape == (2,)
    
    # Check info dictionary
    assert "lam" in info
    assert "beta" in info
    assert "betadot" in info
    assert len(info["lam"]) == 101  # n_steps + 1
    assert len(info["beta"]) == 101
    assert len(info["betadot"]) == 101


def test_spf_linear_mode(simple_lg_model):
    """Test SPF with linear beta mode."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=2000,
        n_steps=200,
        beta_mode="linear",
        seed=42
    )
    
    # In linear mode, beta should equal lambda
    np.testing.assert_allclose(info["beta"], info["lam"], atol=1e-10)
    
    # betadot should be 1 everywhere
    np.testing.assert_allclose(info["betadot"], np.ones_like(info["lam"]), atol=1e-10)


def test_spf_optimal_mode(simple_lg_model):
    """Test SPF with optimal beta mode."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=2000,
        n_steps=200,
        beta_mode="optimal",
        mu=1e-2,
        seed=42
    )
    
    # Beta should start at 0 and end at 1
    assert np.isclose(info["beta"][0], 0.0, atol=1e-6)
    assert np.isclose(info["beta"][-1], 1.0, atol=1e-6)
    
    # Beta should be monotonic
    assert np.all(np.diff(info["beta"]) >= -1e-6)


def test_spf_different_particle_counts(simple_lg_model):
    """Test SPF with different particle counts."""
    for N in [100, 500, 1000, 2000]:
        X, x_hat, info = run_generalized_spf(
            model=simple_lg_model,
            N=N,
            n_steps=50,
            beta_mode="linear",
            seed=42
        )
        
        assert X.shape == (N, 2)
        assert x_hat.shape == (2,)


def test_spf_different_step_counts(simple_lg_model):
    """Test SPF with different numbers of steps."""
    for n_steps in [50, 100, 200, 300]:
        X, x_hat, info = run_generalized_spf(
            model=simple_lg_model,
            N=1000,
            n_steps=n_steps,
            beta_mode="linear",
            seed=42
        )
        
        assert len(info["lam"]) == n_steps + 1
        assert X.shape == (1000, 2)


def test_spf_q_mode_scaled_identity(simple_lg_model):
    """Test SPF with scaled identity Q mode."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=1000,
        n_steps=100,
        beta_mode="linear",
        Q_mode="scaled_identity",
        q_scale=1e-2,
        seed=42
    )
    
    assert X.shape == (1000, 2)
    assert x_hat.shape == (2,)


def test_spf_q_mode_inv_m(simple_lg_model):
    """Test SPF with inv_M Q mode."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=1000,
        n_steps=100,
        beta_mode="linear",
        Q_mode="inv_M",
        seed=42
    )
    
    assert X.shape == (1000, 2)
    assert x_hat.shape == (2,)


def test_spf_different_mu_values(simple_lg_model):
    """Test SPF with different mu values in optimal mode."""
    for mu in [1e-4, 1e-3, 1e-2, 1e-1]:
        X, x_hat, info = run_generalized_spf(
            model=simple_lg_model,
            N=1000,
            n_steps=100,
            beta_mode="optimal",
            mu=mu,
            seed=42
        )
        
        assert X.shape == (1000, 2)
        # Beta endpoints should always be 0 and 1
        assert np.isclose(info["beta"][0], 0.0, atol=1e-5)
        assert np.isclose(info["beta"][-1], 1.0, atol=1e-5)


def test_spf_different_q_scales(simple_lg_model):
    """Test SPF with different q_scale values."""
    for q_scale in [1e-3, 1e-2, 1e-1]:
        X, x_hat, info = run_generalized_spf(
            model=simple_lg_model,
            N=1000,
            n_steps=100,
            beta_mode="linear",
            Q_mode="scaled_identity",
            q_scale=q_scale,
            seed=42
        )
        
        assert X.shape == (1000, 2)


def test_spf_reproducibility(simple_lg_model):
    """Test that same seed produces same results."""
    X1, x_hat1, info1 = run_generalized_spf(
        model=simple_lg_model,
        N=1000,
        n_steps=100,
        beta_mode="linear",
        seed=42
    )
    
    X2, x_hat2, info2 = run_generalized_spf(
        model=simple_lg_model,
        N=1000,
        n_steps=100,
        beta_mode="linear",
        seed=42
    )
    
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(x_hat1, x_hat2)


def test_spf_different_seeds(simple_lg_model):
    """Test that different seeds produce different results."""
    X1, x_hat1, info1 = run_generalized_spf(
        model=simple_lg_model,
        N=1000,
        n_steps=100,
        beta_mode="linear",
        seed=42
    )
    
    X2, x_hat2, info2 = run_generalized_spf(
        model=simple_lg_model,
        N=1000,
        n_steps=100,
        beta_mode="linear",
        seed=123
    )
    
    # Results should be different
    assert not np.array_equal(X1, X2)


def test_spf_1d_case(simple_1d_model):
    """Test SPF with 1D state."""
    X, x_hat, info = run_generalized_spf(
        model=simple_1d_model,
        N=1000,
        n_steps=100,
        beta_mode="linear",
        seed=42
    )
    
    assert X.shape == (1000, 1)
    assert x_hat.shape == (1,)


def test_spf_high_dimensional():
    """Test SPF with higher dimensional state."""
    n, d = 5, 3
    m0 = np.random.randn(n)
    P0 = np.eye(n) * 2.0
    H = np.random.randn(d, n)
    R = np.eye(d) * 0.5
    z = np.random.randn(d)
    
    model = LinearGaussianBayes(m0=m0, P0=P0, H=H, R=R, z=z)
    
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=1000,
        n_steps=100,
        beta_mode="linear",
        seed=42
    )
    
    assert X.shape == (1000, 5)
    assert x_hat.shape == (5,)


def test_spf_estimate_is_mean(simple_lg_model):
    """Test that x_hat is the mean of particles."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=1000,
        n_steps=100,
        beta_mode="linear",
        seed=42
    )
    
    # x_hat should be the mean of particles
    expected_mean = X.mean(axis=0)
    np.testing.assert_allclose(x_hat, expected_mean, rtol=1e-10)


def test_spf_particles_finite(simple_lg_model):
    """Test that all particles are finite."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=1000,
        n_steps=100,
        beta_mode="linear",
        seed=42
    )
    
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(x_hat))


def test_spf_invalid_beta_mode(simple_lg_model):
    """Test that invalid beta mode raises error."""
    with pytest.raises(ValueError, match="beta_mode must be"):
        run_generalized_spf(
            model=simple_lg_model,
            N=1000,
            n_steps=100,
            beta_mode="invalid",
            seed=42
        )


def test_spf_invalid_q_mode(simple_lg_model):
    """Test that invalid Q mode raises error."""
    with pytest.raises(ValueError, match="Q_mode must be"):
        run_generalized_spf(
            model=simple_lg_model,
            N=1000,
            n_steps=100,
            beta_mode="linear",
            Q_mode="invalid",
            seed=42
        )


def test_spf_small_particle_count(simple_lg_model):
    """Test SPF with very small particle count."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=10,
        n_steps=50,
        beta_mode="linear",
        seed=42
    )
    
    assert X.shape == (10, 2)
    assert x_hat.shape == (2,)


def test_spf_large_particle_count(simple_lg_model):
    """Test SPF with large particle count."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=5000,
        n_steps=100,
        beta_mode="linear",
        seed=42
    )
    
    assert X.shape == (5000, 2)
    assert x_hat.shape == (2,)


def test_spf_few_steps(simple_lg_model):
    """Test SPF with very few steps."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=1000,
        n_steps=10,
        beta_mode="linear",
        seed=42
    )
    
    assert len(info["lam"]) == 11
    assert X.shape == (1000, 2)


def test_spf_many_steps(simple_lg_model):
    """Test SPF with many steps."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=1000,
        n_steps=500,
        beta_mode="linear",
        seed=42
    )
    
    assert len(info["lam"]) == 501
    assert X.shape == (1000, 2)


def test_spf_combined_modes():
    """Test all combinations of beta_mode and Q_mode."""
    model = LinearGaussianBayes(
        m0=np.array([1.0, 2.0]),
        P0=np.eye(2),
        H=np.array([[1.0, 0.5]]),
        R=np.array([[0.5]]),
        z=np.array([3.0])
    )
    
    beta_modes = ["linear", "optimal"]
    q_modes = ["scaled_identity", "inv_M"]
    
    for beta_mode in beta_modes:
        for q_mode in q_modes:
            X, x_hat, info = run_generalized_spf(
                model=model,
                N=500,
                n_steps=50,
                beta_mode=beta_mode,
                Q_mode=q_mode,
                mu=1e-2,
                q_scale=1e-2,
                seed=42
            )
            
            assert X.shape == (500, 2)
            assert x_hat.shape == (2,)


def test_spf_particle_spread(simple_lg_model):
    """Test that particles have reasonable spread."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=2000,
        n_steps=200,
        beta_mode="linear",
        seed=42
    )
    
    # Particles should have non-zero variance
    particle_std = X.std(axis=0)
    assert np.all(particle_std > 0)


def test_spf_info_dict_structure(simple_lg_model):
    """Test that info dictionary has correct structure."""
    X, x_hat, info = run_generalized_spf(
        model=simple_lg_model,
        N=1000,
        n_steps=100,
        beta_mode="linear",
        seed=42
    )
    
    # Check keys
    assert set(info.keys()) == {"lam", "beta", "betadot"}
    
    # Check types
    assert isinstance(info["lam"], np.ndarray)
    assert isinstance(info["beta"], np.ndarray)
    assert isinstance(info["betadot"], np.ndarray)
    
    # Check all arrays have same length
    assert len(info["lam"]) == len(info["beta"]) == len(info["betadot"])
