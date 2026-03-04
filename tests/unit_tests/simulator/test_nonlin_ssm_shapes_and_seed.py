"""Unit tests for simulate_nonlinear_ssm: shapes, dtypes, reproducibility, and I/O."""

import os
import tempfile

import numpy as np
import pytest

from simulator.simulator_nonlinSSM import (
    NonlinearSSMSimulationResult,
    simulate_nonlinear_ssm,
)


@pytest.fixture
def default_params():
    return dict(N=50, sigma_v=10.0, sigma_w=1.0, seed=42)

# Shape and dtype tests
def test_output_shapes(default_params):
    """X and Y must both have shape (N,)."""
    res = simulate_nonlinear_ssm(**default_params)
    N = default_params["N"]
    assert res.X.shape == (N,), f"Expected X.shape=({N},), got {res.X.shape}"
    assert res.Y.shape == (N,), f"Expected Y.shape=({N},), got {res.Y.shape}"


def test_output_dtypes(default_params):
    """X and Y must be float64 arrays."""
    res = simulate_nonlinear_ssm(**default_params)
    assert res.X.dtype == np.float64
    assert res.Y.dtype == np.float64


def test_result_is_dataclass(default_params):
    """Return type must be NonlinearSSMSimulationResult."""
    res = simulate_nonlinear_ssm(**default_params)
    assert isinstance(res, NonlinearSSMSimulationResult)


def test_stored_noise_params(default_params):
    """sigma_v and sigma_w must be stored correctly on the result object."""
    res = simulate_nonlinear_ssm(**default_params)
    assert res.sigma_v == default_params["sigma_v"]
    assert res.sigma_w == default_params["sigma_w"]


def test_various_lengths():
    """Output shape is always (N,) for different N values."""
    for N in [1, 10, 100, 500]:
        res = simulate_nonlinear_ssm(N=N, sigma_v=1.0, sigma_w=1.0, seed=0)
        assert res.X.shape == (N,)
        assert res.Y.shape == (N,)


# Reproducibility and randomness
def test_seed_reproducibility(default_params):
    """Same seed must produce bit-for-bit identical results."""
    r1 = simulate_nonlinear_ssm(**default_params)
    r2 = simulate_nonlinear_ssm(**default_params)
    np.testing.assert_array_equal(r1.X, r2.X)
    np.testing.assert_array_equal(r1.Y, r2.Y)


def test_different_seeds_produce_different_outputs():
    """Different seeds must yield different trajectories."""
    r1 = simulate_nonlinear_ssm(N=50, sigma_v=10.0, sigma_w=1.0, seed=1)
    r2 = simulate_nonlinear_ssm(N=50, sigma_v=10.0, sigma_w=1.0, seed=2)
    assert not np.allclose(r1.X, r2.X)
    assert not np.allclose(r1.Y, r2.Y)


def test_no_seed_runs_without_error():
    """Omitting the seed should still produce valid output."""
    res = simulate_nonlinear_ssm(N=20, sigma_v=1.0, sigma_w=1.0)
    assert res.X.shape == (20,)
    assert np.all(np.isfinite(res.X))


# Input validation
def test_raises_on_non_positive_N():
    with pytest.raises(ValueError, match="N must be positive"):
        simulate_nonlinear_ssm(N=0, sigma_v=1.0, sigma_w=1.0)
    with pytest.raises(ValueError, match="N must be positive"):
        simulate_nonlinear_ssm(N=-10, sigma_v=1.0, sigma_w=1.0)


def test_raises_on_non_positive_sigma_v():
    with pytest.raises(ValueError):
        simulate_nonlinear_ssm(N=10, sigma_v=0.0, sigma_w=1.0)
    with pytest.raises(ValueError):
        simulate_nonlinear_ssm(N=10, sigma_v=-1.0, sigma_w=1.0)


def test_raises_on_non_positive_sigma_w():
    with pytest.raises(ValueError):
        simulate_nonlinear_ssm(N=10, sigma_v=1.0, sigma_w=0.0)
    with pytest.raises(ValueError):
        simulate_nonlinear_ssm(N=10, sigma_v=1.0, sigma_w=-1.0)


def test_raises_on_infinite_sigma():
    with pytest.raises(ValueError):
        simulate_nonlinear_ssm(N=10, sigma_v=np.inf, sigma_w=1.0)
    with pytest.raises(ValueError):
        simulate_nonlinear_ssm(N=10, sigma_v=1.0, sigma_w=np.inf)


# Burn-in tests
def test_burnin_does_not_change_output_shape():
    """burn_in must not alter output shape."""
    r0 = simulate_nonlinear_ssm(N=30, sigma_v=1.0, sigma_w=1.0, seed=7, burn_in=0)
    r1 = simulate_nonlinear_ssm(N=30, sigma_v=1.0, sigma_w=1.0, seed=7, burn_in=20)
    assert r0.X.shape == r1.X.shape
    assert r0.Y.shape == r1.Y.shape


def test_burnin_changes_trajectory():
    """Applying burn-in must produce a different trajectory than no burn-in."""
    r0 = simulate_nonlinear_ssm(N=30, sigma_v=1.0, sigma_w=1.0, seed=7, burn_in=0)
    r1 = simulate_nonlinear_ssm(N=30, sigma_v=1.0, sigma_w=1.0, seed=7, burn_in=20)
    # After burn-in the initial RNG state differs so trajectories must differ
    assert not np.allclose(r0.X, r1.X)


# x0 override
def test_custom_x0_is_used_as_first_sample():
    """When x0 is provided the very first state should be close to x0
    (x0 is the starting point *before* the first transition, so X[0] ≈ x0
    with near-zero process noise)."""
    res = simulate_nonlinear_ssm(N=5, sigma_v=1e-9, sigma_w=1e-9, seed=0, x0=7.0)
    # With near-zero noise X[0] must be very close to the provided x0
    assert res.X[0] == pytest.approx(7.0, abs=1e-6)


def test_custom_x0_different_from_default():
    """Providing x0 must produce a different trajectory than the default."""
    r_default = simulate_nonlinear_ssm(N=50, sigma_v=1.0, sigma_w=1.0, seed=99)
    r_x0 = simulate_nonlinear_ssm(N=50, sigma_v=1.0, sigma_w=1.0, seed=99, x0=100.0)
    assert not np.allclose(r_default.X, r_x0.X)


# Numerical sanity
def test_all_finite(default_params):
    """All states and observations must be finite for typical parameters."""
    res = simulate_nonlinear_ssm(**default_params)
    assert np.all(np.isfinite(res.X)), "X contains non-finite values"
    assert np.all(np.isfinite(res.Y)), "Y contains non-finite values"


def test_observations_non_negative_on_average():
    """Since Y = X^2/20 + W, Y should be non-negative on average with small noise."""
    res = simulate_nonlinear_ssm(N=500, sigma_v=1.0, sigma_w=0.001, seed=0)
    # Y ≈ X^2/20 >= 0, so mean should be >> 0 when noise is small
    assert np.mean(res.Y) > 0.0


def test_observation_noise_scales_with_sigma_w():
    """Higher sigma_w should produce larger variance in Y - X^2/20."""
    res_low = simulate_nonlinear_ssm(N=2000, sigma_v=1.0, sigma_w=0.01, seed=0, x0=5.0)
    res_high = simulate_nonlinear_ssm(N=2000, sigma_v=1.0, sigma_w=5.0, seed=0, x0=5.0)
    noise_low = res_low.Y - res_low.X ** 2 / 20.0
    noise_high = res_high.Y - res_high.X ** 2 / 20.0
    assert np.std(noise_high) > np.std(noise_low)


def test_zero_noise_deterministic_trajectory():
    """With near-zero noise the trajectory must be essentially deterministic."""
    r1 = simulate_nonlinear_ssm(N=20, sigma_v=1e-9, sigma_w=1e-9, seed=0, x0=1.0)
    r2 = simulate_nonlinear_ssm(N=20, sigma_v=1e-9, sigma_w=1e-9, seed=99, x0=1.0)
    # With negligible noise the trajectories from different seeds should be nearly identical
    np.testing.assert_allclose(r1.X, r2.X, atol=1e-5)
    np.testing.assert_allclose(r1.Y, r2.Y, atol=1e-5)


def test_zero_noise_observation_equals_state_squared_over_20():
    """With near-zero sigma_w every observation must be ≈ X_n^2/20."""
    res = simulate_nonlinear_ssm(N=30, sigma_v=1.0, sigma_w=1e-9, seed=5, x0=3.0)
    np.testing.assert_allclose(res.Y, res.X ** 2 / 20.0, atol=1e-6)


def test_state_transition_formula_zero_process_noise():
    """With near-zero sigma_v each state must satisfy the deterministic recurrence."""
    res = simulate_nonlinear_ssm(N=10, sigma_v=1e-9, sigma_w=1e-9, seed=0, x0=2.0)
    for n in range(1, 10):
        x_prev = res.X[n - 1]
        # n is 1-indexed time step (0-indexed loop index n corresponds to time step n)
        expected = x_prev / 2.0 + 25.0 * x_prev / (1.0 + x_prev ** 2) + 8.0 * np.cos(1.2 * n)
        assert res.X[n] == pytest.approx(expected, abs=1e-4), (
            f"State transition failed at step {n}: expected {expected}, got {res.X[n]}"
        )


# I/O round-trip
def test_to_file_roundtrip(default_params):
    """Save to .npz and reload; arrays must match."""
    res = simulate_nonlinear_ssm(**default_params)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_sim")
        res.to_file(path)
        loaded = np.load(path + ".npz")
        np.testing.assert_array_equal(loaded["X"], res.X)
        np.testing.assert_array_equal(loaded["Y"], res.Y)
        assert float(loaded["sigma_v"]) == res.sigma_v
        assert float(loaded["sigma_w"]) == res.sigma_w


def test_to_file_raises_if_exists(default_params):
    """to_file must raise FileExistsError when overwrite=False and file exists."""
    res = simulate_nonlinear_ssm(**default_params)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_sim")
        res.to_file(path)
        with pytest.raises(FileExistsError):
            res.to_file(path, overwrite=False)


def test_to_file_overwrite_succeeds(default_params):
    """to_file must succeed silently when overwrite=True."""
    res = simulate_nonlinear_ssm(**default_params)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_sim")
        res.to_file(path)
        res.to_file(path, overwrite=True)  # should not raise
