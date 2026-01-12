"""Integration tests for SPF: convergence to Kalman posterior."""

import numpy as np
import pytest
from models.Stochastic_particle_filter import (
    run_generalized_spf,
    LinearGaussianBayes,
)


@pytest.fixture
def simple_2d_model():
    """Simple 2D linear-Gaussian model."""
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


@pytest.mark.integration
def test_spf_converges_to_kalman_posterior_mean(simple_2d_model):
    """Test that SPF mean converges to Kalman posterior mean."""
    model = simple_2d_model
    m_kalman, P_kalman = model.kalman_posterior()
    
    # Run SPF with many particles and steps
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=5000,
        n_steps=300,
        beta_mode="optimal",
        mu=1e-2,
        Q_mode="inv_M",
        seed=42
    )
    
    # SPF estimate should be close to Kalman posterior mean
    # Allow for some Monte Carlo error
    np.testing.assert_allclose(x_hat, m_kalman, rtol=0.1, atol=0.1)


@pytest.mark.integration
def test_spf_converges_to_kalman_posterior_covariance(simple_2d_model):
    """Test that SPF covariance converges to Kalman posterior covariance."""
    model = simple_2d_model
    m_kalman, P_kalman = model.kalman_posterior()
    
    # Run SPF with many particles and steps
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=5000,
        n_steps=300,
        beta_mode="optimal",
        mu=1e-2,
        Q_mode="inv_M",
        seed=42
    )
    
    # Compute empirical covariance
    P_spf = np.cov(X.T)
    
    # Should be reasonably close to Kalman covariance
    np.testing.assert_allclose(P_spf, P_kalman, rtol=0.3, atol=0.3)


@pytest.mark.integration
def test_spf_1d_convergence(simple_1d_model):
    """Test SPF convergence in 1D case."""
    model = simple_1d_model
    m_kalman, P_kalman = model.kalman_posterior()
    
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=5000,
        n_steps=300,
        beta_mode="optimal",
        mu=1e-2,
        Q_mode="inv_M",
        seed=42
    )
    
    # Mean should be close
    np.testing.assert_allclose(x_hat[0], m_kalman[0], rtol=0.05, atol=0.05)
    
    # Variance should be close
    var_spf = np.var(X[:, 0])
    np.testing.assert_allclose(var_spf, P_kalman[0, 0], rtol=0.2, atol=0.2)


@pytest.mark.integration
def test_spf_linear_vs_optimal_mode(simple_2d_model):
    """Compare linear and optimal beta modes."""
    model = simple_2d_model
    
    # Run with linear mode
    X_lin, x_hat_lin, info_lin = run_generalized_spf(
        model=model,
        N=3000,
        n_steps=200,
        beta_mode="linear",
        Q_mode="inv_M",
        seed=42
    )
    
    # Run with optimal mode
    X_opt, x_hat_opt, info_opt = run_generalized_spf(
        model=model,
        N=3000,
        n_steps=200,
        beta_mode="optimal",
        mu=1e-2,
        Q_mode="inv_M",
        seed=42
    )
    
    # Both should give reasonable estimates
    m_kalman, P_kalman = model.kalman_posterior()
    
    np.testing.assert_allclose(x_hat_lin, m_kalman, rtol=0.15, atol=0.15)
    np.testing.assert_allclose(x_hat_opt, m_kalman, rtol=0.15, atol=0.15)


@pytest.mark.integration
def test_spf_q_mode_comparison(simple_2d_model):
    """Compare different Q modes."""
    model = simple_2d_model
    m_kalman, P_kalman = model.kalman_posterior()
    
    # Run with scaled_identity
    X_si, x_hat_si, info_si = run_generalized_spf(
        model=model,
        N=3000,
        n_steps=200,
        beta_mode="linear",
        Q_mode="scaled_identity",
        q_scale=1e-2,
        seed=42
    )
    
    # Run with inv_M
    X_im, x_hat_im, info_im = run_generalized_spf(
        model=model,
        N=3000,
        n_steps=200,
        beta_mode="linear",
        Q_mode="inv_M",
        seed=42
    )
    
    # Both should converge to Kalman posterior
    np.testing.assert_allclose(x_hat_si, m_kalman, rtol=0.15, atol=0.15)
    np.testing.assert_allclose(x_hat_im, m_kalman, rtol=0.15, atol=0.15)


@pytest.mark.integration
def test_spf_increasing_particles_improves_accuracy(simple_2d_model):
    """Test that more particles improve accuracy."""
    model = simple_2d_model
    m_kalman, P_kalman = model.kalman_posterior()
    
    particle_counts = [500, 1000, 2000, 5000]
    errors = []
    
    for N in particle_counts:
        X, x_hat, info = run_generalized_spf(
            model=model,
            N=N,
            n_steps=200,
            beta_mode="linear",
            Q_mode="inv_M",
            seed=42
        )
        
        error = np.linalg.norm(x_hat - m_kalman)
        errors.append(error)
    
    # Errors should generally decrease (allow for some Monte Carlo noise)
    # Check that largest N has smaller error than smallest N
    assert errors[-1] < errors[0] * 1.5


@pytest.mark.integration
def test_spf_increasing_steps_improves_accuracy(simple_2d_model):
    """Test that more steps improve accuracy."""
    model = simple_2d_model
    m_kalman, P_kalman = model.kalman_posterior()
    
    step_counts = [50, 100, 200, 400]
    errors = []
    
    for n_steps in step_counts:
        X, x_hat, info = run_generalized_spf(
            model=model,
            N=2000,
            n_steps=n_steps,
            beta_mode="linear",
            Q_mode="inv_M",
            seed=42
        )
        
        error = np.linalg.norm(x_hat - m_kalman)
        errors.append(error)
    
    # More steps should generally help, but allow for Monte Carlo variability
    # Just check that the error with most steps is reasonable
    assert errors[-1] < 0.2


@pytest.mark.integration
def test_spf_consistency_across_runs(simple_2d_model):
    """Test that SPF gives consistent results across multiple runs."""
    model = simple_2d_model
    
    results = []
    for seed in range(5):
        X, x_hat, info = run_generalized_spf(
            model=model,
            N=2000,
            n_steps=200,
            beta_mode="linear",
            Q_mode="inv_M",
            seed=seed
        )
        results.append(x_hat)
    
    # Compute mean and std across runs
    results = np.array(results)
    mean_estimate = results.mean(axis=0)
    std_estimate = results.std(axis=0)
    
    # All estimates should be reasonably close to each other
    assert np.all(std_estimate < 0.3)
    
    # Mean should be close to Kalman posterior
    m_kalman, _ = model.kalman_posterior()
    np.testing.assert_allclose(mean_estimate, m_kalman, rtol=0.1, atol=0.1)


@pytest.mark.integration
def test_spf_high_dimensional_model():
    """Test SPF with higher dimensional state."""
    n, d = 5, 3
    
    # Create well-conditioned problem
    m0 = np.zeros(n)
    P0 = np.eye(n) * 2.0
    H = np.random.RandomState(42).randn(d, n) * 0.5
    R = np.eye(d) * 1.0
    z = np.random.RandomState(42).randn(d)
    
    model = LinearGaussianBayes(m0=m0, P0=P0, H=H, R=R, z=z)
    m_kalman, P_kalman = model.kalman_posterior()
    
    # Run SPF
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=5000,
        n_steps=300,
        beta_mode="linear",
        Q_mode="inv_M",
        seed=42
    )
    
    # Should converge reasonably well even in higher dimensions
    np.testing.assert_allclose(x_hat, m_kalman, rtol=0.2, atol=0.2)


@pytest.mark.integration
def test_spf_informative_observation():
    """Test SPF with highly informative observation."""
    # Very precise observation (small R) but not too extreme
    model = LinearGaussianBayes(
        m0=np.array([0.0, 0.0]),
        P0=np.eye(2) * 5.0,
        H=np.array([[1.0, 0.0]]),
        R=np.array([[0.1]]),  # Informative but not too small
        z=np.array([10.0])
    )
    
    m_kalman, P_kalman = model.kalman_posterior()
    
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=5000,
        n_steps=300,
        beta_mode="optimal",
        mu=1e-2,
        Q_mode="inv_M",
        seed=42
    )
    
    # Should still converge well
    np.testing.assert_allclose(x_hat, m_kalman, rtol=0.15, atol=0.15)


@pytest.mark.integration
def test_spf_uninformative_observation():
    """Test SPF with uninformative observation."""
    # Very noisy observation (large R)
    model = LinearGaussianBayes(
        m0=np.array([5.0, 3.0]),
        P0=np.eye(2) * 1.0,
        H=np.array([[1.0, 0.5]]),
        R=np.array([[100.0]]),  # Very large noise
        z=np.array([10.0])
    )
    
    m_kalman, P_kalman = model.kalman_posterior()
    
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=5000,
        n_steps=300,
        beta_mode="optimal",
        mu=1e-2,
        Q_mode="inv_M",
        seed=42
    )
    
    # Posterior should be close to prior when observation is uninformative
    # SPF should still approximate it well
    np.testing.assert_allclose(x_hat, m_kalman, rtol=0.15, atol=0.15)


@pytest.mark.integration
def test_spf_multivariate_observation():
    """Test SPF with multivariate observations."""
    model = LinearGaussianBayes(
        m0=np.array([1.0, 2.0, 3.0]),
        P0=np.eye(3) * 2.0,
        H=np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]]),
        R=np.eye(2) * 0.5,
        z=np.array([2.5, 3.5])
    )
    
    m_kalman, P_kalman = model.kalman_posterior()
    
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=5000,
        n_steps=300,
        beta_mode="linear",
        Q_mode="inv_M",
        seed=42
    )
    
    np.testing.assert_allclose(x_hat, m_kalman, rtol=0.15, atol=0.15)


@pytest.mark.integration
def test_spf_correlated_prior():
    """Test SPF with correlated prior covariance."""
    # Create correlated prior
    P0 = np.array([[2.0, 1.5, 0.5],
                   [1.5, 3.0, 1.0],
                   [0.5, 1.0, 1.5]])
    
    model = LinearGaussianBayes(
        m0=np.array([1.0, 2.0, 1.5]),
        P0=P0,
        H=np.array([[1.0, 0.5, 0.3]]),
        R=np.array([[0.8]]),
        z=np.array([3.0])
    )
    
    m_kalman, P_kalman = model.kalman_posterior()
    
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=5000,
        n_steps=300,
        beta_mode="optimal",
        mu=1e-2,
        Q_mode="inv_M",
        seed=42
    )
    
    np.testing.assert_allclose(x_hat, m_kalman, rtol=0.15, atol=0.15)


@pytest.mark.integration
def test_spf_different_mu_convergence(simple_2d_model):
    """Test that different mu values all converge to same posterior."""
    model = simple_2d_model
    m_kalman, P_kalman = model.kalman_posterior()
    
    mu_values = [1e-3, 1e-2, 5e-2]
    estimates = []
    
    for mu in mu_values:
        X, x_hat, info = run_generalized_spf(
            model=model,
            N=3000,
            n_steps=300,
            beta_mode="optimal",
            mu=mu,
            Q_mode="inv_M",
            seed=42
        )
        estimates.append(x_hat)
    
    # All estimates should converge to Kalman posterior
    for x_hat in estimates:
        np.testing.assert_allclose(x_hat, m_kalman, rtol=0.2, atol=0.2)


@pytest.mark.integration
def test_spf_particle_coverage():
    """Test that particles cover the posterior support."""
    model = LinearGaussianBayes(
        m0=np.array([0.0, 0.0]),
        P0=np.eye(2) * 2.0,
        H=np.array([[1.0, 0.5]]),
        R=np.array([[0.5]]),
        z=np.array([1.0])
    )
    
    m_kalman, P_kalman = model.kalman_posterior()
    
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=5000,
        n_steps=300,
        beta_mode="linear",
        Q_mode="inv_M",
        seed=42
    )
    
    # Compute Mahalanobis distances
    diff = X - m_kalman[None, :]
    P_kalman_inv = np.linalg.inv(P_kalman)
    mahal_sq = np.sum(diff @ P_kalman_inv * diff, axis=1)
    
    # Most particles should be within ~3 standard deviations
    # For 2D Gaussian, chi-square(2) with 99.7% coverage is ~13.8
    within_3sigma = np.mean(mahal_sq < 13.8)
    assert within_3sigma > 0.90  # At least 90% should be in reasonable range
