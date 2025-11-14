"""Unit tests for Lorenz 96 simulator: shapes, seeds, and reproducibility."""

import numpy as np
import pytest
from simulator.simulator_Lorenz_96 import simulate_lorenz96


def test_basic_shapes():
    """Test that simulation returns correct shapes."""
    nx = 40
    total_steps = 100
    Np = 20
    obs_interval = 10
    obs_fraction = 4
    
    result = simulate_lorenz96(
        nx=nx,
        total_steps=total_steps,
        Np=Np,
        obs_interval=obs_interval,
        obs_fraction=obs_fraction,
        seed=42,
    )
    
    # Truth trajectory: (T+1, nx)
    assert result.truth_traj.shape == (total_steps + 1, nx)
    
    # Ensemble trajectory: (Np, T+1, nx)
    assert result.ensemble_traj.shape == (Np, total_steps + 1, nx)
    
    # Observations: number of obs times x number of observed variables
    ny = nx // obs_fraction
    n_obs_times = len(np.arange(0, total_steps + 1, obs_interval))
    assert result.observations.shape == (n_obs_times, ny)
    
    # Observation times array
    assert result.obs_times.shape == (n_obs_times,)
    
    # Observation indices
    assert result.H_idx.shape == (ny,)
    
    # Observation covariance
    assert result.R.shape == (ny, ny)


def test_reproducibility_with_seed():
    """Test that same seed produces identical results."""
    nx = 40
    total_steps = 50
    seed = 123
    
    result1 = simulate_lorenz96(nx=nx, total_steps=total_steps, seed=seed)
    result2 = simulate_lorenz96(nx=nx, total_steps=total_steps, seed=seed)
    
    np.testing.assert_array_equal(result1.truth_traj, result2.truth_traj)
    np.testing.assert_array_equal(result1.ensemble_traj, result2.ensemble_traj)
    np.testing.assert_array_equal(result1.observations, result2.observations)
    np.testing.assert_array_equal(result1.obs_times, result2.obs_times)


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results."""
    nx = 40
    total_steps = 50
    
    result1 = simulate_lorenz96(nx=nx, total_steps=total_steps, seed=1)
    result2 = simulate_lorenz96(nx=nx, total_steps=total_steps, seed=2)
    
    # Ensemble and observations should be different 
    # Truth trajectory may be the same if spinup is deterministic, so we check ensemble
    assert not np.allclose(result1.ensemble_traj, result2.ensemble_traj)
    assert not np.allclose(result1.observations, result2.observations)


def test_config_stored_correctly():
    """Test that configuration is properly stored in result."""
    nx = 60
    F = 10.0
    dt = 0.02
    total_steps = 100
    Np = 30
    obs_interval = 15
    obs_fraction = 5
    obs_error_std = 1.5
    seed = 999
    
    result = simulate_lorenz96(
        nx=nx,
        F=F,
        dt=dt,
        total_steps=total_steps,
        Np=Np,
        obs_interval=obs_interval,
        obs_fraction=obs_fraction,
        obs_error_std=obs_error_std,
        seed=seed,
    )
    
    assert result.config['nx'] == nx
    assert result.config['F'] == F
    assert result.config['dt'] == dt
    assert result.config['total_steps'] == total_steps
    assert result.config['Np'] == Np
    assert result.config['obs_interval'] == obs_interval
    assert result.config['obs_fraction'] == obs_fraction
    assert result.config['obs_error_std'] == obs_error_std
    assert result.config['seed'] == seed


def test_default_parameters():
    """Test simulation with default parameters."""
    result = simulate_lorenz96(seed=42)
    
    # Check default values from function signature
    assert result.config['nx'] == 1000
    assert result.config['F'] == 8.0
    assert result.config['dt'] == 0.01
    assert result.config['Np'] == 20
    
    # Check shapes with defaults
    assert result.truth_traj.shape[1] == 1000
    assert result.ensemble_traj.shape[0] == 20


def test_custom_initial_condition():
    """Test simulation with custom initial condition."""
    nx = 40
    x0 = np.ones(nx) * 5.0
    
    result = simulate_lorenz96(nx=nx, x0=x0, total_steps=50, seed=42)
    
    # The spinup should start from x0, then evolve
    # Truth and ensemble should be initialized from spinup endpoint
    assert result.truth_traj.shape == (51, nx)
    assert result.ensemble_traj.shape[2] == nx


def test_invalid_x0_shape_raises_error():
    """Test that invalid x0 shape raises ValueError."""
    nx = 40
    x0 = np.ones(50)  # Wrong size
    
    with pytest.raises(ValueError, match="x0 must have shape"):
        simulate_lorenz96(nx=nx, x0=x0, seed=42)


def test_observation_indices_correct():
    """Test that observation indices are correct."""
    nx = 40
    obs_fraction = 4  # Observe every 4th variable
    
    result = simulate_lorenz96(nx=nx, obs_fraction=obs_fraction, total_steps=50, seed=42)
    
    expected_idx = np.arange(0, nx, obs_fraction)
    np.testing.assert_array_equal(result.H_idx, expected_idx)


def test_observation_covariance_diagonal():
    """Test that observation covariance is diagonal with correct variance."""
    obs_error_std = 2.0
    nx = 40
    obs_fraction = 4
    
    result = simulate_lorenz96(
        nx=nx,
        obs_fraction=obs_fraction,
        obs_error_std=obs_error_std,
        total_steps=50,
        seed=42,
    )
    
    ny = nx // obs_fraction
    expected_R = (obs_error_std ** 2) * np.eye(ny)
    np.testing.assert_array_equal(result.R, expected_R)


def test_dtypes_are_float():
    """Test that all arrays have float dtype."""
    result = simulate_lorenz96(nx=40, total_steps=50, seed=42)
    
    assert result.truth_traj.dtype == np.float64
    assert result.ensemble_traj.dtype == np.float64
    assert result.observations.dtype == np.float64
    assert result.H_idx.dtype in [np.int32, np.int64, int]
    assert result.R.dtype == np.float64


def test_observation_times_correct():
    """Test that observation times are correctly spaced."""
    total_steps = 100
    obs_interval = 20
    
    result = simulate_lorenz96(total_steps=total_steps, obs_interval=obs_interval, seed=42)
    
    expected_times = np.arange(0, total_steps + 1, obs_interval)
    np.testing.assert_array_equal(result.obs_times, expected_times)


def test_small_system_fast():
    """Test that small system runs quickly."""
    result = simulate_lorenz96(nx=20, total_steps=10, Np=5, seed=42)
    
    assert result.truth_traj.shape == (11, 20)
    assert result.ensemble_traj.shape == (5, 11, 20)


def test_perturbation_std_default():
    """Test that default perturbation std is sqrt(2)."""
    result = simulate_lorenz96(nx=40, total_steps=50, seed=42)
    
    assert result.config['perturbation_std'] == np.sqrt(2.0)


def test_custom_perturbation_std():
    """Test simulation with custom perturbation std."""
    perturbation_std = 1.5
    
    result = simulate_lorenz96(
        nx=40,
        total_steps=50,
        perturbation_std=perturbation_std,
        seed=42,
    )
    
    assert result.config['perturbation_std'] == perturbation_std
