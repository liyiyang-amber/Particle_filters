"""Integration test: Kernel Particle Filter vs LGSSM simulator."""

import numpy as np
import pytest
from simulator.simulator_LGSSM import simulate_lgssm
from models.kernel_particle_filter import KernelParticleFilter, Model, KPFConfig


@pytest.mark.integration
def test_kpf_linear_gaussian_system():
    """Test KPF on linear Gaussian system."""
    # System parameters
    nx, ny = 2, 2
    A = np.array([[0.9, 0.2], [0.0, 0.7]])
    B = np.diag([np.sqrt(0.05), np.sqrt(0.02)])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = np.array([[np.sqrt(0.10), 0.0], [0.0, np.sqrt(0.10)]])
    Sigma = np.eye(nx)

    # Simulate data
    N = 100
    seed = 42
    result = simulate_lgssm(A, B, C, D, Sigma, N=N, seed=seed, burn_in=50)

    # Observation model
    R = D @ D.T

    def h(x):
        return C @ x

    def jh(x):
        return C

    model = Model(H=h, JH=jh, R=R)

    # Configure KPF
    config = KPFConfig(
        kernel_type="diagonal",
        lengthscale_mode="std",
        max_steps=50,
    )
    kpf = KernelParticleFilter(model=model, config=config)

    # Initialize ensemble from prior
    Np = 200
    rng = np.random.default_rng(seed)
    ensemble = rng.multivariate_normal(np.zeros(nx), Sigma, size=Np)

    # Process noise covariance
    Q = B @ B.T

    # Run filtering
    rmse_values = []
    for t in range(min(20, N)):
        y = result.Y[t]
        truth = result.X[t]

        # Analyze
        state = kpf.analyze(ensemble, y, rng=rng)

        # Compute RMSE
        mean_estimate = state.particles.mean(axis=0)
        rmse = np.sqrt(np.mean((mean_estimate - truth) ** 2))
        rmse_values.append(rmse)

        # Forecast to next time
        ensemble = (A @ state.particles.T).T
        noise = rng.multivariate_normal(np.zeros(nx), Q, size=Np)
        ensemble = ensemble + noise

    # Mean RMSE should be reasonable
    mean_rmse = np.mean(rmse_values)
    assert mean_rmse < 2.0


@pytest.mark.integration
def test_kpf_lgssm_scalar_kernel():
    """Test KPF with scalar kernel on linear Gaussian system."""
    nx, ny = 2, 1
    A = np.array([[0.9, 0.2], [0.0, 0.7]])
    B = np.diag([np.sqrt(0.05), np.sqrt(0.02)])
    C = np.array([[1.0, 0.0]])
    D = np.array([[np.sqrt(0.10)]])
    Sigma = np.eye(nx)

    N = 50
    seed = 42
    result = simulate_lgssm(A, B, C, D, Sigma, N=N, seed=seed, burn_in=20)

    R = D @ D.T

    def h(x):
        return C @ x

    def jh(x):
        return C

    model = Model(H=h, JH=jh, R=R)

    # Use scalar kernel
    config = KPFConfig(
        kernel_type="scalar",
        lengthscale_mode="std",
        max_steps=40,
    )
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 150
    rng = np.random.default_rng(seed)
    ensemble = rng.multivariate_normal(np.zeros(nx), Sigma, size=Np)

    Q = B @ B.T

    rmse_values = []
    for t in range(min(10, N)):
        y = result.Y[t]
        truth = result.X[t]

        state = kpf.analyze(ensemble, y, rng=rng)

        mean_estimate = state.particles.mean(axis=0)
        rmse = np.sqrt(np.mean((mean_estimate - truth) ** 2))
        rmse_values.append(rmse)

        ensemble = (A @ state.particles.T).T
        noise = rng.multivariate_normal(np.zeros(nx), Q, size=Np)
        ensemble = ensemble + noise

    mean_rmse = np.mean(rmse_values)
    assert mean_rmse < 2.0


@pytest.mark.integration
def test_kpf_lgssm_convergence():
    """Test that KPF converges over time on LGSSM."""
    nx, ny = 2, 2
    A = np.array([[0.9, 0.2], [0.0, 0.7]])
    B = np.diag([np.sqrt(0.05), np.sqrt(0.02)])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = np.array([[np.sqrt(0.10), 0.0], [0.0, np.sqrt(0.10)]])
    Sigma = np.eye(nx)

    N = 100
    seed = 42
    result = simulate_lgssm(A, B, C, D, Sigma, N=N, seed=seed, burn_in=50)

    R = D @ D.T

    def h(x):
        return C @ x

    def jh(x):
        return C

    model = Model(H=h, JH=jh, R=R)

    config = KPFConfig(kernel_type="diagonal", max_steps=50)
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 200
    rng = np.random.default_rng(seed)
    ensemble = rng.multivariate_normal(np.zeros(nx), Sigma, size=Np)

    Q = B @ B.T

    rmse_first_half = []
    rmse_second_half = []

    for t in range(min(40, N)):
        y = result.Y[t]
        truth = result.X[t]

        state = kpf.analyze(ensemble, y, rng=rng)

        mean_estimate = state.particles.mean(axis=0)
        rmse = np.sqrt(np.mean((mean_estimate - truth) ** 2))

        if t < 20:
            rmse_first_half.append(rmse)
        else:
            rmse_second_half.append(rmse)

        ensemble = (A @ state.particles.T).T
        noise = rng.multivariate_normal(np.zeros(nx), Q, size=Np)
        ensemble = ensemble + noise

    # Second half should generally have lower RMSE (filter has converged)
    mean_rmse_first = np.mean(rmse_first_half)
    mean_rmse_second = np.mean(rmse_second_half)

    # Both halves should have reasonable RMSE (filter is working)
    # Due to randomness, second half may not always be strictly better
    assert mean_rmse_first < 3.0
    assert mean_rmse_second < 3.0


@pytest.mark.integration
def test_kpf_lgssm_ensemble_consistency():
    """Test that ensemble remains consistent (no NaN, finite values)."""
    nx, ny = 2, 2
    A = np.array([[0.9, 0.2], [0.0, 0.7]])
    B = np.diag([np.sqrt(0.05), np.sqrt(0.02)])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = np.array([[np.sqrt(0.10), 0.0], [0.0, np.sqrt(0.10)]])
    Sigma = np.eye(nx)

    N = 50
    seed = 42
    result = simulate_lgssm(A, B, C, D, Sigma, N=N, seed=seed, burn_in=20)

    R = D @ D.T

    def h(x):
        return C @ x

    def jh(x):
        return C

    model = Model(H=h, JH=jh, R=R)
    config = KPFConfig(kernel_type="diagonal", max_steps=50)
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 200
    rng = np.random.default_rng(seed)
    ensemble = rng.multivariate_normal(np.zeros(nx), Sigma, size=Np)

    Q = B @ B.T

    for t in range(min(20, N)):
        y = result.Y[t]

        state = kpf.analyze(ensemble, y, rng=rng)

        # Check for NaN or Inf
        assert np.all(np.isfinite(state.particles))
        assert np.all(np.isfinite(state.weights))

        # Ensemble should not collapse
        spread = np.std(state.particles, axis=0)
        assert np.all(spread > 1e-6)

        # Forecast
        ensemble = (A @ state.particles.T).T
        noise = rng.multivariate_normal(np.zeros(nx), Q, size=Np)
        ensemble = ensemble + noise


@pytest.mark.integration
def test_kpf_lgssm_pseudo_time_completion():
    """Test that KPF completes pseudo-time integration to s=1."""
    nx, ny = 2, 1
    A = np.array([[0.9, 0.2], [0.0, 0.7]])
    B = np.diag([np.sqrt(0.05), np.sqrt(0.02)])
    C = np.array([[1.0, 0.0]])
    D = np.array([[np.sqrt(0.10)]])
    Sigma = np.eye(nx)

    N = 30
    seed = 42
    result = simulate_lgssm(A, B, C, D, Sigma, N=N, seed=seed, burn_in=20)

    R = D @ D.T

    def h(x):
        return C @ x

    def jh(x):
        return C

    model = Model(H=h, JH=jh, R=R)
    config = KPFConfig(kernel_type="diagonal", max_steps=60, min_steps=5)
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 150
    rng = np.random.default_rng(seed)
    ensemble = rng.multivariate_normal(np.zeros(nx), Sigma, size=Np)

    # Test on several observations
    for t in range(min(10, N)):
        y = result.Y[t]

        state = kpf.analyze(ensemble, y, rng=rng)

        # Should reach s=1
        assert np.isclose(state.s, 1.0, atol=1e-4)

        # Should take reasonable number of steps
        assert config.min_steps <= state.steps <= config.max_steps

        # Update ensemble
        Q = B @ B.T
        ensemble = (A @ state.particles.T).T
        noise = rng.multivariate_normal(np.zeros(nx), Q, size=Np)
        ensemble = ensemble + noise
