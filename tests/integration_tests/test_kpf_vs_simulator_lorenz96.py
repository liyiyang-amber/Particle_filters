"""Integration test: Kernel Particle Filter vs Lorenz 96 simulator."""

import numpy as np
import pytest
from simulator.simulator_Lorenz_96 import simulate_lorenz96
from models.kernel_particle_filter import KernelParticleFilter, Model, KPFConfig


@pytest.mark.integration
def test_kpf_lorenz96_basic_filtering():
    """Test KPF on Lorenz 96 system with basic configuration."""
    # Simulate Lorenz 96 data
    nx = 40
    total_steps = 100
    Np = 200
    obs_interval = 10
    obs_fraction = 4
    obs_error_std = 1.0
    seed = 42

    result = simulate_lorenz96(
        nx=nx,
        total_steps=total_steps,
        Np=Np,
        obs_interval=obs_interval,
        obs_fraction=obs_fraction,
        obs_error_std=obs_error_std,
        seed=seed,
    )

    # Extract observation model
    H_idx = result.H_idx
    R = result.R

    def h(x):
        return x[H_idx]

    def jh(x):
        ny = len(H_idx)
        J = np.zeros((ny, nx))
        J[np.arange(ny), H_idx] = 1.0
        return J

    model = Model(H=h, JH=jh, R=R)

    # Configure KPF
    config = KPFConfig(
        kernel_type="diagonal",
        lengthscale_mode="std",
        max_steps=50,
        min_steps=5,
    )
    kpf = KernelParticleFilter(model=model, config=config)

    # Run filtering on first few observations
    n_obs = 5
    ensemble = result.ensemble_traj[:, 0, :]  # Initial ensemble

    rmse_values = []
    for i in range(n_obs):
        obs_idx = i * (obs_interval // obs_interval)
        y = result.observations[obs_idx]

        # Analyze
        state = kpf.analyze(ensemble, y)

        # Compute RMSE
        truth = result.truth_traj[result.obs_times[obs_idx]]
        mean_estimate = state.particles.mean(axis=0)
        rmse = np.sqrt(np.mean((mean_estimate - truth) ** 2))
        rmse_values.append(rmse)

        # Update ensemble for next step
        ensemble = state.particles

    # RMSE should be reasonable 
    mean_rmse = np.mean(rmse_values)
    assert mean_rmse < 5.0  # Reasonable bound for Lorenz 96


@pytest.mark.integration
def test_kpf_lorenz96_diagonal_vs_scalar_kernel():
    """Compare diagonal and scalar kernels on Lorenz 96."""
    # Small system for faster test
    nx = 20
    total_steps = 50
    Np = 100
    obs_interval = 10
    seed = 42

    result = simulate_lorenz96(
        nx=nx,
        total_steps=total_steps,
        Np=Np,
        obs_interval=obs_interval,
        seed=seed,
    )

    H_idx = result.H_idx
    R = result.R

    def h(x):
        return x[H_idx]

    def jh(x):
        ny = len(H_idx)
        J = np.zeros((ny, nx))
        J[np.arange(ny), H_idx] = 1.0
        return J

    model = Model(H=h, JH=jh, R=R)

    # Test with diagonal kernel
    config_diag = KPFConfig(kernel_type="diagonal", max_steps=30)
    kpf_diag = KernelParticleFilter(model=model, config=config_diag)

    ensemble_diag = result.ensemble_traj[:, 0, :].copy()
    y = result.observations[0]
    state_diag = kpf_diag.analyze(ensemble_diag, y)

    # Test with scalar kernel
    config_scalar = KPFConfig(kernel_type="scalar", max_steps=30)
    kpf_scalar = KernelParticleFilter(model=model, config=config_scalar)

    ensemble_scalar = result.ensemble_traj[:, 0, :].copy()
    state_scalar = kpf_scalar.analyze(ensemble_scalar, y)

    # Both should complete successfully and reach s=1
    assert np.isclose(state_diag.s, 1.0, atol=1e-3)
    assert np.isclose(state_scalar.s, 1.0, atol=1e-3)

    # Both should give reasonable estimates
    truth = result.truth_traj[0]
    rmse_diag = np.sqrt(np.mean((state_diag.particles.mean(axis=0) - truth) ** 2))
    rmse_scalar = np.sqrt(np.mean((state_scalar.particles.mean(axis=0) - truth) ** 2))

    assert rmse_diag < 10.0
    assert rmse_scalar < 10.0


@pytest.mark.integration
def test_kpf_lorenz96_with_localization():
    """Test KPF on Lorenz 96 with localization."""
    nx = 40
    total_steps = 50
    Np = 150
    obs_interval = 10
    seed = 42

    result = simulate_lorenz96(
        nx=nx,
        total_steps=total_steps,
        Np=Np,
        obs_interval=obs_interval,
        seed=seed,
    )

    H_idx = result.H_idx
    R = result.R

    def h(x):
        return x[H_idx]

    def jh(x):
        ny = len(H_idx)
        J = np.zeros((ny, nx))
        J[np.arange(ny), H_idx] = 1.0
        return J

    model = Model(H=h, JH=jh, R=R)

    # Configure with localization
    config = KPFConfig(
        kernel_type="diagonal",
        localization_radius=10.0,
        max_steps=40,
    )
    kpf = KernelParticleFilter(model=model, config=config)

    ensemble = result.ensemble_traj[:, 0, :]
    y = result.observations[0]

    state = kpf.analyze(ensemble, y)

    # Should complete successfully
    assert np.isclose(state.s, 1.0, atol=1e-3)

    # Should give reasonable estimate
    truth = result.truth_traj[0]
    rmse = np.sqrt(np.mean((state.particles.mean(axis=0) - truth) ** 2))
    assert rmse < 10.0


@pytest.mark.integration
def test_kpf_lorenz96_convergence_over_time():
    """Test that KPF improves estimates over multiple assimilation cycles."""
    nx = 30
    total_steps = 100
    Np = 150
    obs_interval = 10
    seed = 42

    result = simulate_lorenz96(
        nx=nx,
        total_steps=total_steps,
        Np=Np,
        obs_interval=obs_interval,
        seed=seed,
    )

    H_idx = result.H_idx
    R = result.R

    def h(x):
        return x[H_idx]

    def jh(x):
        ny = len(H_idx)
        J = np.zeros((ny, nx))
        J[np.arange(ny), H_idx] = 1.0
        return J

    model = Model(H=h, JH=jh, R=R)
    config = KPFConfig(kernel_type="diagonal", max_steps=40)
    kpf = KernelParticleFilter(model=model, config=config)

    # Simple forecast model: just add noise
    def forecast_ensemble(ensemble, rng):
        Q = np.eye(nx) * 0.1
        noise = rng.multivariate_normal(np.zeros(nx), Q, size=ensemble.shape[0])
        return ensemble + noise

    ensemble = result.ensemble_traj[:, 0, :]
    rng = np.random.default_rng(seed)

    rmse_prior = []
    rmse_post = []

    for i in range(min(5, len(result.obs_times))):
        obs_idx = result.obs_times[i]
        y = result.observations[i]
        truth = result.truth_traj[obs_idx]

        # Prior RMSE
        prior_mean = ensemble.mean(axis=0)
        rmse_prior.append(np.sqrt(np.mean((prior_mean - truth) ** 2)))

        # Analyze
        state = kpf.analyze(ensemble, y)

        # Posterior RMSE
        post_mean = state.particles.mean(axis=0)
        rmse_post.append(np.sqrt(np.mean((post_mean - truth) ** 2)))

        # Forecast to next time
        ensemble = forecast_ensemble(state.particles, rng)

    # Posterior should generally be better than prior
    mean_rmse_prior = np.mean(rmse_prior)
    mean_rmse_post = np.mean(rmse_post)

    assert mean_rmse_post < mean_rmse_prior


@pytest.mark.integration
def test_kpf_lorenz96_ensemble_spread():
    """Test that ensemble spread is reasonable after KPF analysis."""
    nx = 30
    total_steps = 50
    Np = 200
    obs_interval = 10
    seed = 42

    result = simulate_lorenz96(
        nx=nx,
        total_steps=total_steps,
        Np=Np,
        obs_interval=obs_interval,
        seed=seed,
    )

    H_idx = result.H_idx
    R = result.R

    def h(x):
        return x[H_idx]

    def jh(x):
        ny = len(H_idx)
        J = np.zeros((ny, nx))
        J[np.arange(ny), H_idx] = 1.0
        return J

    model = Model(H=h, JH=jh, R=R)
    config = KPFConfig(kernel_type="diagonal", max_steps=40)
    kpf = KernelParticleFilter(model=model, config=config)

    ensemble = result.ensemble_traj[:, 0, :]
    y = result.observations[0]

    # Prior spread
    prior_spread = np.std(ensemble, axis=0).mean()

    state = kpf.analyze(ensemble, y)

    # Posterior spread
    post_spread = np.std(state.particles, axis=0).mean()

    # Spread should still be positive 
    assert post_spread > 0.1

    # Posterior spread typically smaller than prior 
    assert post_spread < prior_spread * 1.5


@pytest.mark.integration
def test_kpf_lorenz96_different_lengthscale_modes():
    """Test KPF with different lengthscale modes."""
    nx = 20
    total_steps = 30
    Np = 100
    seed = 42

    result = simulate_lorenz96(
        nx=nx,
        total_steps=total_steps,
        Np=Np,
        seed=seed,
    )

    H_idx = result.H_idx
    R = result.R

    def h(x):
        return x[H_idx]

    def jh(x):
        ny = len(H_idx)
        J = np.zeros((ny, nx))
        J[np.arange(ny), H_idx] = 1.0
        return J

    model = Model(H=h, JH=jh, R=R)
    ensemble = result.ensemble_traj[:, 0, :]
    y = result.observations[0]

    # Test std mode
    config_std = KPFConfig(lengthscale_mode="std", max_steps=30)
    kpf_std = KernelParticleFilter(model=model, config=config_std)
    state_std = kpf_std.analyze(ensemble.copy(), y)

    # Test fixed mode
    config_fixed = KPFConfig(lengthscale_mode="fixed", fixed_lengthscale=2.0, max_steps=30)
    kpf_fixed = KernelParticleFilter(model=model, config=config_fixed)
    state_fixed = kpf_fixed.analyze(ensemble.copy(), y)

    # Both should complete successfully
    assert np.isclose(state_std.s, 1.0, atol=1e-3)
    assert np.isclose(state_fixed.s, 1.0, atol=1e-3)

    # Both should give reasonable estimates
    truth = result.truth_traj[0]
    rmse_std = np.sqrt(np.mean((state_std.particles.mean(axis=0) - truth) ** 2))
    rmse_fixed = np.sqrt(np.mean((state_fixed.particles.mean(axis=0) - truth) ** 2))

    assert rmse_std < 10.0
    assert rmse_fixed < 10.0
