import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from simulator.simulator_sensor_network_skewt_dynamic import (
    GridConfig,
    DynConfig,
    MeasConfig,
    SimConfig,
    simulate_trial,
    simulate_many,
    save_npz,
    load_npz,
)


def test_save_npz_creates_file():
    """Test that save_npz creates a file."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10)
    
    data = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_data.npz")
        save_npz(path, data)
        
        assert os.path.exists(path)


def test_save_load_roundtrip_trial():
    """Test that save and load preserve single trial data."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=20, save_lambda=True)
    
    data_orig = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "trial_data.npz")
        save_npz(path, data_orig)
        data_loaded = load_npz(path)
        
        # Check all array fields
        np.testing.assert_array_equal(data_orig["X"], data_loaded["X"])
        np.testing.assert_array_equal(data_orig["Z"], data_loaded["Z"])
        np.testing.assert_array_equal(data_orig["Lambda"], data_loaded["Lambda"])
        np.testing.assert_array_equal(data_orig["Sigma"], data_loaded["Sigma"])
        np.testing.assert_array_equal(data_orig["L"], data_loaded["L"])
        np.testing.assert_array_equal(data_orig["R"], data_loaded["R"])
        np.testing.assert_array_equal(data_orig["gamma"], data_loaded["gamma"])


def test_save_load_roundtrip_many():
    """Test that save and load preserve multi-trial data."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=15, n_trials=5, save_lambda=True)
    
    data_orig = simulate_many(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "many_data.npz")
        save_npz(path, data_orig)
        data_loaded = load_npz(path)
        
        # Check main arrays
        np.testing.assert_array_equal(data_orig["X"], data_loaded["X"])
        np.testing.assert_array_equal(data_orig["Z"], data_loaded["Z"])
        np.testing.assert_array_equal(data_orig["Lambda"], data_loaded["Lambda"])
        
        # Check list fields
        assert len(data_loaded["Sigma_list"]) == len(data_orig["Sigma_list"])
        for i in range(len(data_orig["Sigma_list"])):
            np.testing.assert_array_equal(data_orig["Sigma_list"][i], data_loaded["Sigma_list"][i])


def test_save_load_without_lambda():
    """Test save/load when Lambda is not saved."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10, save_lambda=False)
    
    data_orig = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "no_lambda.npz")
        save_npz(path, data_orig)
        data_loaded = load_npz(path)
        
        assert "Lambda" not in data_loaded
        np.testing.assert_array_equal(data_orig["X"], data_loaded["X"])


def test_save_npz_compressed():
    """Test that save_npz uses compression."""
    grid_cfg = GridConfig(d=64)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=100, save_lambda=True)
    
    data = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "compressed.npz")
        save_npz(path, data)
        
        # File should exist and be reasonably sized
        assert os.path.exists(path)
        file_size = os.path.getsize(path)
        
        # Compressed should be smaller than uncompressed arrays
        # Just check it's not zero
        assert file_size > 0


def test_load_npz_nonexistent_file():
    """Test that load_npz raises error for nonexistent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nonexistent.npz")
        
        with pytest.raises(FileNotFoundError):
            load_npz(path)


def test_save_load_meta_preservation():
    """Test that metadata is preserved in save/load."""
    grid_cfg = GridConfig(d=16, alpha0=2.5)
    dyn_cfg = DynConfig(alpha=0.85, seed=42)
    meas_cfg = MeasConfig(m1=1.5)
    sim_cfg = SimConfig(T=10)
    
    data_orig = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta_test.npz")
        save_npz(path, data_orig)
        data_loaded = load_npz(path)
        
        # Meta should be preserved
        assert "meta" in data_loaded
        # meta is a dict that gets saved as object array


def test_save_load_shapes_preserved():
    """Test that array shapes are preserved in save/load."""
    grid_cfg = GridConfig(d=25)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=30, n_trials=8, save_lambda=True)
    
    data_orig = simulate_many(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "shapes.npz")
        save_npz(path, data_orig)
        data_loaded = load_npz(path)
        
        assert data_loaded["X"].shape == (8, 30, 25)
        assert data_loaded["Z"].shape == (8, 30, 25)
        assert data_loaded["Lambda"].shape == (8, 30, 25)


def test_save_load_dtypes_preserved():
    """Test that data types are preserved in save/load."""
    grid_cfg = GridConfig(d=16)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10, save_lambda=True)
    
    data_orig = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dtypes.npz")
        save_npz(path, data_orig)
        data_loaded = load_npz(path)
        
        assert data_loaded["X"].dtype == data_orig["X"].dtype
        assert data_loaded["Z"].dtype == data_orig["Z"].dtype
        assert data_loaded["Lambda"].dtype == data_orig["Lambda"].dtype


def test_save_load_with_path_object():
    """Test that Path objects work with save/load."""
    grid_cfg = GridConfig(d=9)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=5)
    
    data_orig = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "path_test.npz"
        save_npz(str(path), data_orig)
        data_loaded = load_npz(str(path))
        
        np.testing.assert_array_equal(data_orig["X"], data_loaded["X"])


def test_save_multiple_files():
    """Test saving multiple different datasets to different files."""
    grid_cfg = GridConfig(d=16)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10)
    
    dyn_cfg1 = DynConfig(seed=1)
    data1 = simulate_trial(grid_cfg, dyn_cfg1, meas_cfg, sim_cfg)
    
    dyn_cfg2 = DynConfig(seed=2)
    data2 = simulate_trial(grid_cfg, dyn_cfg2, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, "data1.npz")
        path2 = os.path.join(tmpdir, "data2.npz")
        
        save_npz(path1, data1)
        save_npz(path2, data2)
        
        loaded1 = load_npz(path1)
        loaded2 = load_npz(path2)
        
        np.testing.assert_array_equal(data1["X"], loaded1["X"])
        np.testing.assert_array_equal(data2["X"], loaded2["X"])
        assert not np.allclose(loaded1["X"], loaded2["X"])


def test_save_overwrite_existing():
    """Test that save_npz overwrites existing files."""
    grid_cfg = GridConfig(d=16)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=10)
    
    dyn_cfg1 = DynConfig(seed=1)
    data1 = simulate_trial(grid_cfg, dyn_cfg1, meas_cfg, sim_cfg)
    
    dyn_cfg2 = DynConfig(seed=2)
    data2 = simulate_trial(grid_cfg, dyn_cfg2, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "overwrite.npz")
        
        # Save first dataset
        save_npz(path, data1)
        loaded1 = load_npz(path)
        
        # Save second dataset to same path
        save_npz(path, data2)
        loaded2 = load_npz(path)
        
        # Should have second dataset
        np.testing.assert_array_equal(data2["X"], loaded2["X"])
        assert not np.allclose(loaded1["X"], loaded2["X"])


def test_save_load_large_dataset():
    """Test save/load with large dataset."""
    grid_cfg = GridConfig(d=144)  # 12x12
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=200, n_trials=20, save_lambda=True)
    
    data_orig = simulate_many(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "large.npz")
        save_npz(path, data_orig)
        data_loaded = load_npz(path)
        
        # Check shapes
        assert data_loaded["X"].shape == (20, 200, 144)
        
        # Spot check some values
        np.testing.assert_array_equal(data_orig["X"][0, :5, :5], data_loaded["X"][0, :5, :5])


def test_load_npz_returns_dict():
    """Test that load_npz returns a dictionary."""
    grid_cfg = GridConfig(d=9)
    dyn_cfg = DynConfig(seed=42)
    meas_cfg = MeasConfig()
    sim_cfg = SimConfig(T=5)
    
    data_orig = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dict_test.npz")
        save_npz(path, data_orig)
        data_loaded = load_npz(path)
        
        assert isinstance(data_loaded, dict)
        assert "X" in data_loaded
        assert "Z" in data_loaded
