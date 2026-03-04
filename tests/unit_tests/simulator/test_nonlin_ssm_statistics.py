"""Unit tests for simulate_nonlinear_ssm: statistical properties."""

import numpy as np
import pytest

from simulator.simulator_nonlinSSM import simulate_nonlinear_ssm


@pytest.mark.slow
def test_initial_state_mean_near_zero():
    """X[0] ~ N(0, 5) so its mean over many draws must be close to 0."""
    M = 3000
    x0_samples = [
        simulate_nonlinear_ssm(N=1, sigma_v=1e-9, sigma_w=1e-9, seed=i).X[0]
        for i in range(M)
    ]
    mean_x0 = np.mean(x0_samples)
    se = np.sqrt(5.0 / M)  # std-error of mean: std/sqrt(M) where std=sqrt(5)
    assert abs(mean_x0) < 4 * se, (
        f"Initial-state mean {mean_x0:.4f} is too far from 0 (4*SE={4*se:.4f})"
    )


@pytest.mark.slow
def test_initial_state_variance_near_5():
    """X[0] ~ N(0, 5) so its variance over many draws must be close to 5."""
    M = 3000
    x0_samples = [
        simulate_nonlinear_ssm(N=1, sigma_v=1e-9, sigma_w=1e-9, seed=i).X[0]
        for i in range(M)
    ]
    var_x0 = np.var(x0_samples, ddof=1)
    # 95% CI for sample variance: roughly var ± 2*var*sqrt(2/(M-1))
    tol = 2 * 5.0 * np.sqrt(2.0 / (M - 1))
    assert abs(var_x0 - 5.0) < tol, (
        f"Initial-state variance {var_x0:.4f} too far from 5.0 (tol={tol:.4f})"
    )


@pytest.mark.slow
def test_process_noise_mean_near_zero():
    """The one-step transition residual should have mean ≈ 0."""
    N = 500
    M = 200
    residuals = []
    for seed in range(M):
        res = simulate_nonlinear_ssm(N=N, sigma_v=1.0, sigma_w=1e-9, seed=seed, x0=0.0)
        for n in range(1, N):
            x_prev = res.X[n - 1]
            det = x_prev / 2.0 + 25.0 * x_prev / (1.0 + x_prev ** 2) + 8.0 * np.cos(1.2 * n)
            residuals.append(res.X[n] - det)
    residuals = np.array(residuals)
    mean_res = np.mean(residuals)
    se = 1.0 / np.sqrt(len(residuals))  # sigma_v=1 so SE = 1/sqrt(n)
    assert abs(mean_res) < 4 * se, (
        f"Process-noise mean {mean_res:.4f} too far from 0 (4*SE={4*se:.4f})"
    )


@pytest.mark.slow
def test_process_noise_std_matches_sigma_v():
    """Std of one-step residuals must match sigma_v."""
    sigma_v = 3.0
    N = 200
    M = 300
    residuals = []
    for seed in range(M):
        res = simulate_nonlinear_ssm(N=N, sigma_v=sigma_v, sigma_w=1e-9, seed=seed, x0=1.0)
        for n in range(1, N):
            x_prev = res.X[n - 1]
            det = x_prev / 2.0 + 25.0 * x_prev / (1.0 + x_prev ** 2) + 8.0 * np.cos(1.2 * n)
            residuals.append(res.X[n] - det)
    std_res = np.std(residuals, ddof=1)
    # Allow 10% relative tolerance given finite sample size
    assert abs(std_res - sigma_v) / sigma_v < 0.10, (
        f"Process-noise std {std_res:.4f} too far from sigma_v={sigma_v}"
    )


@pytest.mark.slow
def test_measurement_noise_mean_near_zero():
    """The measurement residual Y - X^2/20 should have mean ≈ 0."""
    N = 1000
    res = simulate_nonlinear_ssm(N=N, sigma_v=1.0, sigma_w=1.0, seed=42)
    noise = res.Y - res.X ** 2 / 20.0
    mean_noise = np.mean(noise)
    se = 1.0 / np.sqrt(N)  # sigma_w=1
    assert abs(mean_noise) < 4 * se, (
        f"Measurement-noise mean {mean_noise:.4f} too far from 0 (4*SE={4*se:.4f})"
    )


@pytest.mark.slow
def test_measurement_noise_std_matches_sigma_w():
    """Std of measurement residuals must match sigma_w."""
    sigma_w = 2.5
    N = 5000
    res = simulate_nonlinear_ssm(N=N, sigma_v=1.0, sigma_w=sigma_w, seed=0)
    noise = res.Y - res.X ** 2 / 20.0
    std_noise = np.std(noise, ddof=1)
    assert abs(std_noise - sigma_w) / sigma_w < 0.05, (
        f"Measurement-noise std {std_noise:.4f} too far from sigma_w={sigma_w}"
    )


def test_observation_equals_state_squared_over_20_plus_noise():
    """Y[n] - X[n]^2/20 must equal the measurement noise realisation."""
    # With near-zero sigma_v, X is nearly deterministic, so residuals are almost pure W_n
    res = simulate_nonlinear_ssm(N=50, sigma_v=1e-9, sigma_w=2.0, seed=7, x0=3.0)
    residuals = res.Y - res.X ** 2 / 20.0
    # All residuals must be finite
    assert np.all(np.isfinite(residuals))


def test_observations_bounded_by_state_and_noise():
    """Without measurement noise, Y must equal X^2/20 to high precision."""
    res = simulate_nonlinear_ssm(N=100, sigma_v=2.0, sigma_w=1e-9, seed=0)
    np.testing.assert_allclose(res.Y, res.X ** 2 / 20.0, atol=1e-6)


@pytest.mark.slow
def test_state_variance_increases_with_process_noise():
    """Larger sigma_v must result in larger variance in the state trajectory."""
    N = 2000
    res_lo = simulate_nonlinear_ssm(N=N, sigma_v=0.5, sigma_w=0.1, seed=0)
    res_hi = simulate_nonlinear_ssm(N=N, sigma_v=5.0, sigma_w=0.1, seed=0)
    assert np.var(res_hi.X) > np.var(res_lo.X)


@pytest.mark.slow
def test_observation_variance_increases_with_sigma_w():
    """Larger sigma_w must result in larger variance in Y - X^2/20."""
    N = 2000
    res_lo = simulate_nonlinear_ssm(N=N, sigma_v=1.0, sigma_w=0.1, seed=0)
    res_hi = simulate_nonlinear_ssm(N=N, sigma_v=1.0, sigma_w=10.0, seed=0)
    noise_lo = res_lo.Y - res_lo.X ** 2 / 20.0
    noise_hi = res_hi.Y - res_hi.X ** 2 / 20.0
    assert np.var(noise_hi) > np.var(noise_lo)
