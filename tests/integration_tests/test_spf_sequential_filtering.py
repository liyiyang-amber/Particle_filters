"""Integration tests for SPF: sequential filtering scenario."""

import numpy as np
import pytest
from models.Stochastic_particle_filter import (
    run_generalized_spf,
    LinearGaussianBayes,
)


def generate_linear_gaussian_data(n_steps, A, H, Q, R, x0, seed=42):
    """Generate synthetic linear Gaussian state-space data."""
    rng = np.random.default_rng(seed)
    nx = len(x0)
    nz = H.shape[0]
    
    X = np.zeros((n_steps, nx))
    Z = np.zeros((n_steps, nz))
    
    X[0] = x0
    Z[0] = H @ X[0] + rng.multivariate_normal(np.zeros(nz), R)
    
    for t in range(1, n_steps):
        X[t] = A @ X[t-1] + rng.multivariate_normal(np.zeros(nx), Q)
        Z[t] = H @ X[t] + rng.multivariate_normal(np.zeros(nz), R)
    
    return X, Z


@pytest.fixture
def simple_ssm_params():
    """Simple state-space model parameters."""
    nx, nz = 2, 1
    A = np.array([[0.9, 0.2], [0.0, 0.8]])
    H = np.array([[1.0, 0.5]])
    Q = np.diag([0.1, 0.05])
    R = np.array([[0.2]])
    x0 = np.array([1.0, 1.0])
    return dict(A=A, H=H, Q=Q, R=R, x0=x0, nx=nx, nz=nz)


@pytest.fixture
def synthetic_data(simple_ssm_params):
    """Generate synthetic data."""
    return generate_linear_gaussian_data(
        n_steps=50,
        A=simple_ssm_params["A"],
        H=simple_ssm_params["H"],
        Q=simple_ssm_params["Q"],
        R=simple_ssm_params["R"],
        x0=simple_ssm_params["x0"],
        seed=42
    )


@pytest.mark.integration
def test_spf_sequential_filtering_basic(simple_ssm_params, synthetic_data):
    """Test SPF in sequential filtering scenario."""
    X_true, Z_obs = synthetic_data
    params = simple_ssm_params
    n_steps = len(X_true)
    
    # Initialize with true initial state
    m0 = params["x0"]
    P0 = np.eye(params["nx"]) * 0.5
    
    # Storage for filtered states
    filtered_means = []
    
    # Sequential filtering
    for t in range(n_steps):
        # Create model for this time step
        model = LinearGaussianBayes(
            m0=m0,
            P0=P0,
            H=params["H"],
            R=params["R"],
            z=Z_obs[t]
        )
        
        # Run SPF
        X_particles, x_hat, info = run_generalized_spf(
            model=model,
            N=2000,
            n_steps=100,
            beta_mode="linear",
            Q_mode="inv_M",
            seed=t
        )
        
        filtered_means.append(x_hat)
        
        # Predict for next step (if not last)
        if t < n_steps - 1:
            m0 = params["A"] @ x_hat
            # Simple covariance prediction (not exact but reasonable)
            P_filtered = np.cov(X_particles.T)
            P0 = params["A"] @ P_filtered @ params["A"].T + params["Q"]
    
    filtered_means = np.array(filtered_means)
    
    # Check that we got estimates for all time steps
    assert filtered_means.shape == (n_steps, params["nx"])
    
    # Estimates should be finite
    assert np.all(np.isfinite(filtered_means))


@pytest.mark.integration
def test_spf_tracking_performance(simple_ssm_params, synthetic_data):
    """Test that SPF tracks the true state reasonably well."""
    X_true, Z_obs = synthetic_data
    params = simple_ssm_params
    n_steps = len(X_true)
    
    m0 = params["x0"]
    P0 = np.eye(params["nx"]) * 0.5
    
    filtered_means = []
    errors = []
    
    for t in range(n_steps):
        model = LinearGaussianBayes(
            m0=m0,
            P0=P0,
            H=params["H"],
            R=params["R"],
            z=Z_obs[t]
        )
        
        X_particles, x_hat, info = run_generalized_spf(
            model=model,
            N=2000,
            n_steps=100,
            beta_mode="linear",
            Q_mode="inv_M",
            seed=t
        )
        
        filtered_means.append(x_hat)
        error = np.linalg.norm(x_hat - X_true[t])
        errors.append(error)
        
        # Predict for next step
        if t < n_steps - 1:
            m0 = params["A"] @ x_hat
            P_filtered = np.cov(X_particles.T)
            P0 = params["A"] @ P_filtered @ params["A"].T + params["Q"]
    
    # Average error should be reasonable
    mean_error = np.mean(errors)
    assert mean_error < 2.0  # Reasonable threshold for this problem


@pytest.mark.integration
def test_spf_single_step_filtering():
    """Test SPF for a single filtering step with known posterior."""
    # Simple 1D case where we can compute exact posterior
    m0 = 5.0
    P0 = 2.0
    H = 1.0
    R = 1.0
    z = 7.0
    
    model = LinearGaussianBayes(
        m0=np.array([m0]),
        P0=np.array([[P0]]),
        H=np.array([[H]]),
        R=np.array([[R]]),
        z=np.array([z])
    )
    
    # Analytic posterior
    m_post_exact, P_post_exact = model.kalman_posterior()
    
    # SPF estimate
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=10000,
        n_steps=300,
        beta_mode="optimal",
        mu=1e-2,
        Q_mode="inv_M",
        seed=42
    )
    
    # Mean should be very close with many particles
    np.testing.assert_allclose(x_hat[0], m_post_exact[0], rtol=0.05, atol=0.05)
    
    # Variance should also be close
    var_spf = np.var(X[:, 0])
    np.testing.assert_allclose(var_spf, P_post_exact[0, 0], rtol=0.15, atol=0.15)


@pytest.mark.integration
def test_spf_prediction_update_cycle():
    """Test SPF in a predict-update cycle."""
    # Start with initial state
    m0 = np.array([1.0, 2.0])
    P0 = np.eye(2) * 1.0
    
    # System matrices
    A = np.array([[0.9, 0.1], [0.0, 0.85]])
    Q = np.diag([0.1, 0.05])
    H = np.array([[1.0, 0.5]])
    R = np.array([[0.2]])
    
    # Observation
    z = np.array([2.5])
    
    # Update step (filtering)
    model_update = LinearGaussianBayes(m0=m0, P0=P0, H=H, R=R, z=z)
    X_updated, x_hat_updated, info = run_generalized_spf(
        model=model_update,
        N=3000,
        n_steps=150,
        beta_mode="linear",
        Q_mode="inv_M",
        seed=42
    )
    
    # Prediction step
    x_pred = A @ x_hat_updated
    P_updated = np.cov(X_updated.T)
    P_pred = A @ P_updated @ A.T + Q
    
    # Check that prediction is valid
    assert x_pred.shape == (2,)
    assert P_pred.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(P_pred) > 0)  # PSD check


@pytest.mark.integration
def test_spf_with_different_initial_conditions():
    """Test SPF sensitivity to initial conditions."""
    H = np.array([[1.0, 0.5]])
    R = np.array([[0.5]])
    z = np.array([3.0])
    
    # Try different priors
    initial_conditions = [
        (np.array([0.0, 0.0]), np.eye(2) * 5.0),
        (np.array([5.0, 5.0]), np.eye(2) * 2.0),
        (np.array([1.0, 2.0]), np.eye(2) * 1.0),
    ]
    
    estimates = []
    
    for m0, P0 in initial_conditions:
        model = LinearGaussianBayes(m0=m0, P0=P0, H=H, R=R, z=z)
        
        X, x_hat, info = run_generalized_spf(
            model=model,
            N=5000,
            n_steps=300,
            beta_mode="linear",
            Q_mode="inv_M",
            seed=42
        )
        
        estimates.append(x_hat)
        
        # Each should converge to their respective Kalman posterior
        m_kalman, _ = model.kalman_posterior()
        np.testing.assert_allclose(x_hat, m_kalman, rtol=0.15, atol=0.15)


@pytest.mark.integration
def test_spf_observability_full_observation():
    """Test SPF when all states are directly observed."""
    # Full observability: H = I
    model = LinearGaussianBayes(
        m0=np.array([1.0, 2.0]),
        P0=np.eye(2) * 2.0,
        H=np.eye(2),  # Observe both states
        R=np.eye(2) * 0.5,
        z=np.array([2.0, 3.0])
    )
    
    m_kalman, P_kalman = model.kalman_posterior()
    
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=5000,
        n_steps=200,
        beta_mode="linear",
        Q_mode="inv_M",
        seed=42
    )
    
    # Should converge well when fully observable
    np.testing.assert_allclose(x_hat, m_kalman, rtol=0.1, atol=0.1)


@pytest.mark.integration
def test_spf_observability_partial_observation():
    """Test SPF when only subset of states is observed."""
    # Partial observability: only observe first state
    model = LinearGaussianBayes(
        m0=np.array([1.0, 2.0, 3.0]),
        P0=np.eye(3) * 2.0,
        H=np.array([[1.0, 0.0, 0.0]]),  # Only observe first state
        R=np.array([[0.5]]),
        z=np.array([2.0])
    )
    
    m_kalman, P_kalman = model.kalman_posterior()
    
    X, x_hat, info = run_generalized_spf(
        model=model,
        N=5000,
        n_steps=200,
        beta_mode="linear",
        Q_mode="inv_M",
        seed=42
    )
    
    # Should still approximate posterior
    np.testing.assert_allclose(x_hat, m_kalman, rtol=0.15, atol=0.15)


@pytest.mark.integration
def test_spf_consistency_over_time_steps():
    """Test that SPF gives consistent results when run multiple times."""
    model = LinearGaussianBayes(
        m0=np.array([1.0, 2.0]),
        P0=np.eye(2) * 2.0,
        H=np.array([[1.0, 0.5]]),
        R=np.array([[0.5]]),
        z=np.array([3.0])
    )
    
    # Run multiple times with different seeds
    estimates = []
    for seed in range(10):
        X, x_hat, info = run_generalized_spf(
            model=model,
            N=2000,
            n_steps=150,
            beta_mode="linear",
            Q_mode="inv_M",
            seed=seed
        )
        estimates.append(x_hat)
    
    estimates = np.array(estimates)
    
    # Check consistency: std across runs should be small
    std_across_runs = estimates.std(axis=0)
    assert np.all(std_across_runs < 0.5)


@pytest.mark.integration
def test_spf_with_noisy_measurements_sequence():
    """Test SPF with sequence of increasingly noisy measurements."""
    m0 = np.array([0.0, 0.0])
    P0 = np.eye(2) * 3.0
    H = np.array([[1.0, 0.5]])
    
    # Sequence with increasing noise
    R_values = [0.1, 0.5, 1.0, 5.0]
    z_values = [2.0, 2.5, 3.0, 3.5]
    
    for R_val, z_val in zip(R_values, z_values):
        model = LinearGaussianBayes(
            m0=m0,
            P0=P0,
            H=H,
            R=np.array([[R_val]]),
            z=np.array([z_val])
        )
        
        X, x_hat, info = run_generalized_spf(
            model=model,
            N=3000,
            n_steps=150,
            beta_mode="linear",
            Q_mode="inv_M",
            seed=42
        )
        
        # Should produce valid estimate regardless of noise level
        assert np.all(np.isfinite(x_hat))
        
        # Update prior for next iteration
        m0 = x_hat
        P0 = np.cov(X.T)


@pytest.mark.integration
def test_spf_stability_long_sequence():
    """Test SPF stability over a longer sequence."""
    params = {
        "A": np.array([[0.95, 0.05], [0.0, 0.9]]),
        "H": np.array([[1.0, 0.3]]),
        "Q": np.diag([0.05, 0.03]),
        "R": np.array([[0.15]]),
        "x0": np.array([1.0, 1.0])
    }
    
    X_true, Z_obs = generate_linear_gaussian_data(
        n_steps=100,
        A=params["A"],
        H=params["H"],
        Q=params["Q"],
        R=params["R"],
        x0=params["x0"],
        seed=42
    )
    
    m0 = params["x0"]
    P0 = np.eye(2) * 0.5
    
    all_estimates = []
    
    for t in range(100):
        model = LinearGaussianBayes(
            m0=m0,
            P0=P0,
            H=params["H"],
            R=params["R"],
            z=Z_obs[t]
        )
        
        X_particles, x_hat, info = run_generalized_spf(
            model=model,
            N=1500,
            n_steps=100,
            beta_mode="linear",
            Q_mode="inv_M",
            seed=t
        )
        
        all_estimates.append(x_hat)
        
        # Predict
        m0 = params["A"] @ x_hat
        P_filtered = np.cov(X_particles.T)
        P0 = params["A"] @ P_filtered @ params["A"].T + params["Q"]
    
    all_estimates = np.array(all_estimates)
    
    # Check that estimates remain finite throughout
    assert np.all(np.isfinite(all_estimates))
    
    # Check that estimates don't diverge unreasonably
    assert np.all(np.abs(all_estimates) < 50.0)
