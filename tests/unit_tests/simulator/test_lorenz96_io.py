"""Unit tests for Lorenz 96 I/O: save and load functionality."""

import numpy as np
import pytest
from pathlib import Path
from simulator.simulator_Lorenz_96 import simulate_lorenz96


def test_save_creates_files(tmp_path):
    """Test that save creates .npz and .json files."""
    result = simulate_lorenz96(nx=40, total_steps=50, seed=42)
    
    filepath = tmp_path / "test_lorenz96"
    result.save(str(filepath))
    
    # Check both files exist
    assert (tmp_path / "test_lorenz96.npz").exists()
    assert (tmp_path / "test_lorenz96.json").exists()


def test_save_with_npz_extension(tmp_path):
    """Test that save works with .npz extension in filename."""
    result = simulate_lorenz96(nx=40, total_steps=50, seed=42)
    
    filepath = tmp_path / "test_lorenz96.npz"
    result.save(str(filepath))
    
    assert filepath.exists()
    assert (tmp_path / "test_lorenz96.json").exists()


def test_save_overwrite_false_raises_error(tmp_path):
    """Test that save with overwrite=False raises error if file exists."""
    result = simulate_lorenz96(nx=40, total_steps=50, seed=42)
    
    filepath = tmp_path / "test_lorenz96"
    result.save(str(filepath))
    
    # Try to save again without overwrite
    with pytest.raises(FileExistsError):
        result.save(str(filepath), overwrite=False)


def test_save_overwrite_true_succeeds(tmp_path):
    """Test that save with overwrite=True succeeds even if file exists."""
    result = simulate_lorenz96(nx=40, total_steps=50, seed=42)
    
    filepath = tmp_path / "test_lorenz96"
    result.save(str(filepath))
    result.save(str(filepath), overwrite=True)  


def test_load_restores_data(tmp_path):
    """Test that load correctly restores all data."""
    nx = 40
    total_steps = 50
    Np = 15
    obs_error_std = 1.5
    seed = 123
    
    result_original = simulate_lorenz96(
        nx=nx,
        total_steps=total_steps,
        Np=Np,
        obs_error_std=obs_error_std,
        seed=seed,
    )
    
    filepath = tmp_path / "test_lorenz96"
    result_original.save(str(filepath))
    
    # Load it back
    from simulator.simulator_Lorenz_96 import Lorenz96SimulationResult
    result_loaded = Lorenz96SimulationResult.load(str(filepath))
    
    # Check all arrays match
    np.testing.assert_array_equal(result_loaded.truth_traj, result_original.truth_traj)
    np.testing.assert_array_equal(result_loaded.ensemble_traj, result_original.ensemble_traj)
    np.testing.assert_array_equal(result_loaded.observations, result_original.observations)
    np.testing.assert_array_equal(result_loaded.obs_times, result_original.obs_times)
    np.testing.assert_array_equal(result_loaded.H_idx, result_original.H_idx)
    np.testing.assert_array_equal(result_loaded.R, result_original.R)
    
    # Check config matches
    assert result_loaded.config == result_original.config


def test_load_with_npz_extension(tmp_path):
    """Test that load works with .npz extension in filename."""
    result_original = simulate_lorenz96(nx=40, total_steps=50, seed=42)
    
    filepath = tmp_path / "test_lorenz96.npz"
    result_original.save(str(filepath))
    
    from simulator.simulator_Lorenz_96 import Lorenz96SimulationResult
    result_loaded = Lorenz96SimulationResult.load(str(filepath))
    
    np.testing.assert_array_equal(result_loaded.truth_traj, result_original.truth_traj)


def test_load_without_extension(tmp_path):
    """Test that load works without .npz extension in filename."""
    result_original = simulate_lorenz96(nx=40, total_steps=50, seed=42)
    
    filepath = tmp_path / "test_lorenz96"
    result_original.save(str(filepath))
    
    from simulator.simulator_Lorenz_96 import Lorenz96SimulationResult
    result_loaded = Lorenz96SimulationResult.load(str(filepath))
    
    np.testing.assert_array_equal(result_loaded.truth_traj, result_original.truth_traj)


def test_load_missing_config_graceful(tmp_path):
    """Test that load works even if .json config file is missing."""
    result_original = simulate_lorenz96(nx=40, total_steps=50, seed=42)
    
    filepath = tmp_path / "test_lorenz96"
    result_original.save(str(filepath))
    
    # Remove the JSON file
    (tmp_path / "test_lorenz96.json").unlink()
    
    from simulator.simulator_Lorenz_96 import Lorenz96SimulationResult
    result_loaded = Lorenz96SimulationResult.load(str(filepath))
    
    # Data should still load
    np.testing.assert_array_equal(result_loaded.truth_traj, result_original.truth_traj)
    
    # Config should be empty dict
    assert result_loaded.config == {}


def test_roundtrip_preserves_all_attributes(tmp_path):
    """Test complete save/load roundtrip preserves everything."""
    result_original = simulate_lorenz96(
        nx=60,
        F=10.0,
        dt=0.02,
        spinup_steps=500,
        total_steps=100,
        Np=25,
        obs_interval=15,
        obs_fraction=3,
        obs_error_std=2.0,
        perturbation_std=1.5,
        seed=999,
    )
    
    filepath = tmp_path / "roundtrip_test"
    result_original.save(str(filepath))
    
    from simulator.simulator_Lorenz_96 import Lorenz96SimulationResult
    result_loaded = Lorenz96SimulationResult.load(str(filepath))
    
    # Verify shapes
    assert result_loaded.truth_traj.shape == result_original.truth_traj.shape
    assert result_loaded.ensemble_traj.shape == result_original.ensemble_traj.shape
    assert result_loaded.observations.shape == result_original.observations.shape
    
    # Verify values
    np.testing.assert_array_equal(result_loaded.truth_traj, result_original.truth_traj)
    np.testing.assert_array_equal(result_loaded.ensemble_traj, result_original.ensemble_traj)
    np.testing.assert_array_equal(result_loaded.observations, result_original.observations)
    np.testing.assert_array_equal(result_loaded.obs_times, result_original.obs_times)
    np.testing.assert_array_equal(result_loaded.H_idx, result_original.H_idx)
    np.testing.assert_array_equal(result_loaded.R, result_original.R)
    
    # Verify config
    for key in result_original.config:
        assert result_loaded.config[key] == result_original.config[key]
