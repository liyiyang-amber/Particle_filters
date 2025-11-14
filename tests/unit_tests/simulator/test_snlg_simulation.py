import numpy as np
import pytest
from simulator.simulator_sensor_network_linear_gaussian import (
    SimConfig,
    simulate_dataset,
)


def test_simulate_dataset_output_shapes_default():
    """Test that simulate_dataset returns arrays with correct shapes."""
    cfg = SimConfig(d=16, T=10, trials=5, sigmas=(2.0, 1.0))
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    S = len(cfg.sigmas)  # 2
    R = cfg.trials       # 5
    T = cfg.T            # 10
    d = cfg.d            # 16
    
    assert X.shape == (S, R, T + 1, d)  # (2, 5, 11, 16)
    assert Z.shape == (S, R, T, d)      # (2, 5, 10, 16)
    assert coords.shape == (d, 2)       # (16, 2)
    assert Sigma.shape == (d, d)        # (16, 16)


def test_simulate_dataset_output_shapes_custom():
    """Test shapes with custom configuration."""
    cfg = SimConfig(d=25, T=15, trials=10, sigmas=(1.0, 0.5, 0.25))
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    assert X.shape == (3, 10, 16, 25)
    assert Z.shape == (3, 10, 15, 25)
    assert coords.shape == (25, 2)
    assert Sigma.shape == (25, 25)


def test_simulate_dataset_output_dtypes():
    """Test that output arrays have correct data types."""
    cfg = SimConfig(d=16, T=5, trials=3, sigmas=(1.0,))
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    assert X.dtype == np.float64
    assert Z.dtype == np.float64
    assert coords.dtype in (np.float64, float)
    assert Sigma.dtype in (np.float64, float)


def test_simulate_dataset_initial_state():
    """Test that initial state (t=0) is zero."""
    cfg = SimConfig(d=16, T=10, trials=5, sigmas=(1.0,))
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    # All x0 should be zero
    assert np.allclose(X[:, :, 0, :], 0.0)


def test_simulate_dataset_seed_reproducibility():
    """Test that same seed produces identical results."""
    cfg1 = SimConfig(d=16, T=8, trials=4, sigmas=(2.0, 1.0), seed=42)
    cfg2 = SimConfig(d=16, T=8, trials=4, sigmas=(2.0, 1.0), seed=42)
    
    X1, Z1, coords1, Sigma1 = simulate_dataset(cfg1)
    X2, Z2, coords2, Sigma2 = simulate_dataset(cfg2)
    
    assert np.allclose(X1, X2)
    assert np.allclose(Z1, Z2)
    assert np.allclose(coords1, coords2)
    assert np.allclose(Sigma1, Sigma2)


def test_simulate_dataset_different_seeds():
    """Test that different seeds produce different results."""
    cfg1 = SimConfig(d=16, T=10, trials=5, sigmas=(1.0,), seed=1)
    cfg2 = SimConfig(d=16, T=10, trials=5, sigmas=(1.0,), seed=2)
    
    X1, Z1, _, _ = simulate_dataset(cfg1)
    X2, Z2, _, _ = simulate_dataset(cfg2)
    
    # States should be different
    assert not np.allclose(X1, X2)
    assert not np.allclose(Z1, Z2)
    
    # Quantify the difference
    diff_X = np.linalg.norm(X1 - X2)
    diff_Z = np.linalg.norm(Z1 - Z2)
    assert diff_X > 1.0
    assert diff_Z > 1.0


def test_simulate_dataset_coords_deterministic():
    """Test that coordinates are the same regardless of seed."""
    cfg1 = SimConfig(d=16, seed=1)
    cfg2 = SimConfig(d=16, seed=2)
    
    _, _, coords1, _ = simulate_dataset(cfg1)
    _, _, coords2, _ = simulate_dataset(cfg2)
    
    assert np.array_equal(coords1, coords2)


def test_simulate_dataset_sigma_deterministic():
    """Test that Sigma (covariance) is the same regardless of seed."""
    cfg1 = SimConfig(d=16, alpha0=3.0, alpha1=0.01, beta=20.0, seed=1)
    cfg2 = SimConfig(d=16, alpha0=3.0, alpha1=0.01, beta=20.0, seed=999)
    
    _, _, _, Sigma1 = simulate_dataset(cfg1)
    _, _, _, Sigma2 = simulate_dataset(cfg2)
    
    assert np.allclose(Sigma1, Sigma2)


def test_simulate_dataset_sigma_symmetric():
    """Test that process covariance Sigma is symmetric."""
    cfg = SimConfig(d=16)
    _, _, _, Sigma = simulate_dataset(cfg)
    
    assert np.allclose(Sigma, Sigma.T)


def test_simulate_dataset_sigma_positive_definite():
    """Test that process covariance Sigma is positive definite."""
    cfg = SimConfig(d=16)
    _, _, _, Sigma = simulate_dataset(cfg)
    
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals > 0)


def test_simulate_dataset_single_sigma():
    """Test simulation with a single observation noise level."""
    cfg = SimConfig(d=9, T=5, trials=3, sigmas=(1.0,))
    X, Z, _, _ = simulate_dataset(cfg)
    
    assert X.shape == (1, 3, 6, 9)
    assert Z.shape == (1, 3, 5, 9)


def test_simulate_dataset_multiple_sigmas():
    """Test simulation with multiple observation noise levels."""
    cfg = SimConfig(d=9, T=5, trials=3, sigmas=(2.0, 1.0, 0.5, 0.25))
    X, Z, _, _ = simulate_dataset(cfg)
    
    assert X.shape == (4, 3, 6, 9)
    assert Z.shape == (4, 3, 5, 9)


def test_simulate_dataset_single_trial():
    """Test simulation with a single trial."""
    cfg = SimConfig(d=16, T=10, trials=1, sigmas=(1.0,))
    X, Z, _, _ = simulate_dataset(cfg)
    
    assert X.shape == (1, 1, 11, 16)
    assert Z.shape == (1, 1, 10, 16)


def test_simulate_dataset_minimal_config():
    """Test simulation with minimal valid configuration."""
    cfg = SimConfig(d=1, T=1, trials=1, sigmas=(1.0,))
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    assert X.shape == (1, 1, 2, 1)
    assert Z.shape == (1, 1, 1, 1)
    assert coords.shape == (1, 2)
    assert Sigma.shape == (1, 1)


def test_simulate_dataset_dynamics_scaling():
    """Test that alpha parameter affects state evolution."""
    # High alpha (slow decay)
    cfg_high = SimConfig(d=16, T=10, trials=10, alpha=0.99, seed=42)
    X_high, _, _, _ = simulate_dataset(cfg_high)
    
    # Low alpha (fast decay)
    cfg_low = SimConfig(d=16, T=10, trials=10, alpha=0.5, seed=42)
    X_low, _, _, _ = simulate_dataset(cfg_low)
    
    # With high alpha, states should have larger magnitude over time
    # (since they decay more slowly)
    magnitude_high = np.mean(np.abs(X_high[:, :, -1, :]))
    magnitude_low = np.mean(np.abs(X_low[:, :, -1, :]))
    
    # This is stochastic, but generally true
    assert magnitude_high > magnitude_low


def test_simulate_dataset_observation_noise_levels():
    """Test that different sigma values produce observations with different noise levels."""
    cfg = SimConfig(d=16, T=20, trials=50, sigmas=(0.1, 1.0, 10.0), seed=42)
    X, Z, _, _ = simulate_dataset(cfg)
    
    # Compute empirical noise variance for each sigma level
    # Noise = Z - X[:, :, 1:, :] (observations minus true states)
    for s_idx in range(3):
        noise = Z[s_idx] - X[s_idx, :, 1:, :]
        empirical_std = np.std(noise)
        expected_std = cfg.sigmas[s_idx]
        
        # Allow 20% tolerance due to finite sample size
        assert abs(empirical_std - expected_std) / expected_std < 0.2


def test_simulate_dataset_time_evolution():
    """Test that states evolve over time (not constant)."""
    cfg = SimConfig(d=16, T=20, trials=5, sigmas=(1.0,), seed=42)
    X, _, _, _ = simulate_dataset(cfg)
    
    # Check that states change over time
    for r in range(cfg.trials):
        for t in range(1, cfg.T):
            # State at time t should differ from state at time t-1
            diff = np.linalg.norm(X[0, r, t, :] - X[0, r, t - 1, :])
            assert diff > 0.01  # Should have some change


def test_simulate_dataset_no_nan_or_inf():
    """Test that simulation produces no NaN or Inf values."""
    cfg = SimConfig(d=25, T=15, trials=10, sigmas=(2.0, 1.0))
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    assert not np.any(np.isnan(X))
    assert not np.any(np.isinf(X))
    assert not np.any(np.isnan(Z))
    assert not np.any(np.isinf(Z))
    assert not np.any(np.isnan(coords))
    assert not np.any(np.isnan(Sigma))
