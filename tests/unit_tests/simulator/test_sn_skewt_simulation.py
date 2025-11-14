import numpy as np
import pytest
from simulator.simulator_sensor_network_skewt_dynamic import (
    GridConfig,
    DynConfig,
    MeasConfig,
    SimConfig,
    simulate_trial,
    simulate_many,
)


# ===== Tests for simulate_trial =====

def test_simulate_trial_output_keys():
    """Test that simulate_trial returns all expected keys."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10, save_lambda=True)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    expected_keys = {"X", "Z", "Lambda", "Sigma", "L", "R", "gamma", "meta"}
    assert set(result.keys()) == expected_keys


def test_simulate_trial_output_keys_no_lambda():
    """Test that Lambda is not saved when save_lambda=False."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10, save_lambda=False)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert "Lambda" not in result
    assert "X" in result
    assert "Z" in result


def test_simulate_trial_shapes():
    """Test that output arrays have correct shapes."""
    d = 16
    T = 20
    
    grid_cfg = GridConfig(d=d)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=T, save_lambda=True)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert result["X"].shape == (T, d)
    assert result["Z"].shape == (T, d)
    assert result["Lambda"].shape == (T, d)
    assert result["Sigma"].shape == (d, d)
    assert result["L"].shape == (d, d)
    assert result["R"].shape == (d, 2)
    assert result["gamma"].shape == (d,)


def test_simulate_trial_dtypes():
    """Test that output arrays have correct data types."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10, save_lambda=True)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert result["X"].dtype == np.float64
    assert result["Z"].dtype == np.int64 or result["Z"].dtype == int
    assert result["Lambda"].dtype == np.float64


def test_simulate_trial_seed_reproducibility():
    """Test that same seed produces identical results."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=20, save_lambda=True)
    
    result1 = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    # Reset seed
    dyn_cfg2 = DynConfig(seed=42)
    result2 = simulate_trial(grid_cfg, dyn_cfg2, meas_cfg, sim_cfg)
    
    np.testing.assert_array_equal(result1["X"], result2["X"])
    np.testing.assert_array_equal(result1["Z"], result2["Z"])
    np.testing.assert_array_equal(result1["Lambda"], result2["Lambda"])


def test_simulate_trial_different_seeds():
    """Test that different seeds produce different results."""
    grid_cfg = GridConfig(d=16)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=20, save_lambda=True)
    
    dyn_cfg1 = DynConfig(seed=1)
    result1 = simulate_trial(grid_cfg, dyn_cfg1, meas_cfg, sim_cfg)
    
    dyn_cfg2 = DynConfig(seed=2)
    result2 = simulate_trial(grid_cfg, dyn_cfg2, meas_cfg, sim_cfg)
    
    assert not np.allclose(result1["X"], result2["X"])
    assert not np.array_equal(result1["Z"], result2["Z"])


def test_simulate_trial_counts_non_negative():
    """Test that all count measurements are non-negative."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=30)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert np.all(result["Z"] >= 0)


def test_simulate_trial_lambda_positive():
    """Test that all Poisson rates are positive."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=30, save_lambda=True)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert np.all(result["Lambda"] > 0)


def test_simulate_trial_no_nan_or_inf():
    """Test that simulation produces no NaN or Inf values."""
    grid_cfg = GridConfig(d=25)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=50, save_lambda=True)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert not np.any(np.isnan(result["X"]))
    assert not np.any(np.isinf(result["X"]))
    assert not np.any(np.isnan(result["Lambda"]))
    assert not np.any(np.isinf(result["Lambda"]))


def test_simulate_trial_state_clipping():
    """Test that state clipping prevents extreme values."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(clip_x=(-5.0, 5.0), seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=100)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    # States used for measurement should be within clip range
    # internal X may exceed, but the clipped version is used for lambda
    assert not np.any(np.isinf(result["Lambda"]))


def test_simulate_trial_minimal_config():
    """Test simulation with minimal valid configuration."""
    grid_cfg = GridConfig(d=1)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=1)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert result["X"].shape == (1, 1)
    assert result["Z"].shape == (1, 1)


def test_simulate_trial_large_grid():
    """Test simulation with large grid."""
    grid_cfg = GridConfig(d=400)  # 20x20
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert result["X"].shape == (10, 400)
    assert result["Z"].shape == (10, 400)


def test_simulate_trial_heavy_tails():
    """Test simulation with heavy-tailed dynamics (low nu)."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(nu=3.0, seed=42)  # Low nu => heavy tails
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=50)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    # Should still produce valid output
    assert result["X"].shape == (50, 16)
    assert not np.any(np.isnan(result["X"]))


def test_simulate_trial_skewness():
    """Test simulation with skewed dynamics."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(gamma_scale=0.5, seed=42)  # Larger skew
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=100)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    # With skew, mean should deviate from zero over time
    mean_state = np.mean(result["X"])
    
    # Should have some non-zero mean due to skew
    assert abs(mean_state) > 0.01


def test_simulate_trial_gamma_vector_custom():
    """Test simulation with custom gamma vector."""
    d = 16
    grid_cfg = GridConfig(d=d)
    
    # Custom gamma vector pointing in specific direction
    gamma_custom = np.zeros(d)
    gamma_custom[0] = 0.3  # Skew only first component
    
    dyn_cfg = DynConfig(gamma_vec=gamma_custom, seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=100)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    # First component should have larger mean than others
    mean_first = np.mean(result["X"][:, 0])
    mean_others = np.mean(result["X"][:, 1:])
    
    assert mean_first > mean_others


def test_simulate_trial_meta_storage():
    """Test that metadata is correctly stored."""
    grid_cfg = GridConfig(d=16, alpha0=2.5)
    dyn_cfg = DynConfig(alpha=0.85, nu=6.0, seed=42)
    meas_cfg = MeasConfig(m1=1.5, m2=0.4)
    sim_cfg = SimConfig(T=20)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    meta = result["meta"]
    assert meta["grid_cfg"]["d"] == 16
    assert meta["grid_cfg"]["alpha0"] == 2.5
    assert meta["dyn_cfg"]["alpha"] == 0.85
    assert meta["dyn_cfg"]["nu"] == 6.0
    assert meta["meas_cfg"]["m1"] == 1.5


# ===== Tests for simulate_many =====

def test_simulate_many_output_keys():
    """Test that simulate_many returns all expected keys."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10, n_trials=3, save_lambda=True)
    
    result = simulate_many(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    expected_keys = {"X", "Z", "Lambda", "Sigma_list", "L_list", "gamma_list", "meta_list"}
    assert set(result.keys()) == expected_keys


def test_simulate_many_shapes():
    """Test that output arrays have correct shapes."""
    d = 16
    T = 20
    n_trials = 5
    
    grid_cfg = GridConfig(d=d)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=T, n_trials=n_trials, save_lambda=True)
    
    result = simulate_many(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert result["X"].shape == (n_trials, T, d)
    assert result["Z"].shape == (n_trials, T, d)
    assert result["Lambda"].shape == (n_trials, T, d)
    assert len(result["Sigma_list"]) == n_trials
    assert len(result["L_list"]) == n_trials
    assert len(result["gamma_list"]) == n_trials


def test_simulate_many_single_trial():
    """Test simulate_many with n_trials=1."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10, n_trials=1, save_lambda=True)
    
    result = simulate_many(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert result["X"].shape == (1, 10, 16)
    assert result["Z"].shape == (1, 10, 16)


def test_simulate_many_multiple_trials():
    """Test simulate_many with multiple trials."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=15, n_trials=10, save_lambda=False)
    
    result = simulate_many(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert result["X"].shape == (10, 15, 16)
    assert "Lambda" not in result


def test_simulate_many_seed_offset():
    """Test that trials have different seeds."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10, n_trials=5, save_lambda=True)
    
    result = simulate_many(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    # Each trial should be different
    for i in range(sim_cfg.n_trials - 1):
        assert not np.allclose(result["X"][i], result["X"][i + 1])


def test_simulate_many_reproducibility():
    """Test that same base seed produces reproducible multi-trial results."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=999)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10, n_trials=5, save_lambda=True)
    
    result1 = simulate_many(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    dyn_cfg2 = DynConfig(seed=999)
    result2 = simulate_many(grid_cfg, dyn_cfg2, meas_cfg, sim_cfg)
    
    np.testing.assert_array_equal(result1["X"], result2["X"])
    np.testing.assert_array_equal(result1["Z"], result2["Z"])


def test_simulate_many_consistency():
    """Test that all trials have consistent structure."""
    grid_cfg = GridConfig(d=25)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=20, n_trials=8, save_lambda=True)
    
    result = simulate_many(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    # All Sigma matrices should be the same 
    for i in range(1, sim_cfg.n_trials):
        np.testing.assert_array_equal(result["Sigma_list"][0], result["Sigma_list"][i])


def test_simulate_many_no_nan_or_inf():
    """Test that multi-trial simulation produces no NaN or Inf values."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=30, n_trials=10, save_lambda=True)
    
    result = simulate_many(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert not np.any(np.isnan(result["X"]))
    assert not np.any(np.isinf(result["X"]))
    assert not np.any(np.isnan(result["Lambda"]))


def test_simulate_many_large_n_trials():
    """Test simulate_many with many trials."""
    grid_cfg = GridConfig(d=9)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=5, n_trials=100, save_lambda=False)
    
    result = simulate_many(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    assert result["X"].shape == (100, 5, 9)
    assert len(result["gamma_list"]) == 100


# ===== Statistical property tests =====

def test_simulate_trial_ar1_behavior():
    """Test that dynamics exhibit AR(1) behavior."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(alpha=0.9, nu=1000, gamma_scale=0.0, seed=42)  # High nu => near-Gaussian
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=500)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    # Compute autocorrelation for one sensor
    x = result["X"][:, 0]
    x_mean = np.mean(x)
    x_centered = x - x_mean
    
    # Lag-1 autocorrelation
    autocorr_1 = np.corrcoef(x_centered[:-1], x_centered[1:])[0, 1]
    
    # Should be close to alpha for AR(1)
    assert abs(autocorr_1 - dyn_cfg.alpha) < 0.15


def test_simulate_trial_poisson_counts_distribution():
    """Test that counts follow approximately Poisson distribution."""
    grid_cfg = GridConfig(d=1)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig(m1=5.0, m2=0.0)  # Constant rate = 5
    sim_cfg = SimConfig(T=1000, save_lambda=True)
    
    result = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    # With m2=0, lambda should be constant = m1
    mean_lambda = np.mean(result["Lambda"])
    assert abs(mean_lambda - 5.0) < 0.1
    
    # Count mean should match lambda
    mean_count = np.mean(result["Z"])
    assert abs(mean_count - mean_lambda) < 0.5
    
    # Variance should approximately equal mean for Poisson
    var_count = np.var(result["Z"])
    assert abs(var_count - mean_count) / mean_count < 0.3
