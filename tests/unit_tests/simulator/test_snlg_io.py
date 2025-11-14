import json
import numpy as np
import pytest
from pathlib import Path
from simulator.simulator_sensor_network_linear_gaussian import (
    SimConfig,
    simulate_dataset,
    save_npz,
    dump_config_json,
)


def test_save_npz_creates_file(tmp_path):
    """Test that save_npz creates a file."""
    cfg = SimConfig(d=16, T=5, trials=3, sigmas=(1.0,), seed=42)
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    output_path = tmp_path / "test_data.npz"
    save_npz(str(output_path), X, Z, coords, Sigma, cfg)
    
    assert output_path.exists()


def test_save_npz_roundtrip(tmp_path):
    """Test that saved data can be loaded back correctly."""
    cfg = SimConfig(d=16, T=8, trials=5, sigmas=(2.0, 1.0), seed=99)
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    output_path = tmp_path / "roundtrip_data.npz"
    save_npz(str(output_path), X, Z, coords, Sigma, cfg)
    
    # Load the data back
    data = np.load(str(output_path))
    
    # Verify all arrays
    assert np.allclose(data["X"], X)
    assert np.allclose(data["Z"], Z)
    assert np.allclose(data["coords"], coords)
    assert np.allclose(data["Sigma"], Sigma)


def test_save_npz_metadata(tmp_path):
    """Test that metadata is correctly saved."""
    cfg = SimConfig(
        d=25,
        alpha=0.95,
        T=12,
        trials=7,
        sigmas=(2.0, 1.0, 0.5),
        seed=123,
    )
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    output_path = tmp_path / "metadata_test.npz"
    save_npz(str(output_path), X, Z, coords, Sigma, cfg)
    
    data = np.load(str(output_path))
    
    # Check metadata arrays
    assert data["alpha"][0] == 0.95
    assert data["T"][0] == 12
    assert data["trials"][0] == 7
    assert data["d"][0] == 25
    assert data["seed"][0] == 123
    assert np.allclose(data["sigmas"], [2.0, 1.0, 0.5])


def test_save_npz_sigmas_array(tmp_path):
    """Test that sigmas are saved as an array."""
    cfg = SimConfig(d=16, sigmas=(3.0, 2.0, 1.0, 0.5))
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    output_path = tmp_path / "sigmas_test.npz"
    save_npz(str(output_path), X, Z, coords, Sigma, cfg)
    
    data = np.load(str(output_path))
    
    assert data["sigmas"].shape == (4,)
    assert np.allclose(data["sigmas"], [3.0, 2.0, 1.0, 0.5])


def test_save_npz_compressed(tmp_path):
    """Test that the file is compressed (has .npz extension)."""
    cfg = SimConfig(d=64, T=20, trials=10)
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    output_path = tmp_path / "compressed_test.npz"
    save_npz(str(output_path), X, Z, coords, Sigma, cfg)
    
    # The file should exist and be a valid .npz file
    assert output_path.exists()
    data = np.load(str(output_path))
    assert "X" in data
    assert "Z" in data


def test_save_npz_overwrite(tmp_path):
    """Test that save_npz can overwrite existing files."""
    cfg1 = SimConfig(d=16, T=5, trials=2, seed=1)
    cfg2 = SimConfig(d=16, T=5, trials=2, seed=2)
    
    X1, Z1, coords1, Sigma1 = simulate_dataset(cfg1)
    X2, Z2, coords2, Sigma2 = simulate_dataset(cfg2)
    
    output_path = tmp_path / "overwrite_test.npz"
    
    # Save first dataset
    save_npz(str(output_path), X1, Z1, coords1, Sigma1, cfg1)
    
    # Overwrite with second dataset
    save_npz(str(output_path), X2, Z2, coords2, Sigma2, cfg2)
    
    # Load and verify it's the second dataset
    data = np.load(str(output_path))
    assert np.allclose(data["X"], X2)
    assert data["seed"][0] == 2


def test_dump_config_json_creates_file(tmp_path):
    """Test that dump_config_json creates a JSON file."""
    cfg = SimConfig(d=16, T=10, trials=5)
    
    output_path = tmp_path / "config.json"
    dump_config_json(str(output_path), cfg)
    
    assert output_path.exists()


def test_dump_config_json_content(tmp_path):
    """Test that JSON file contains correct configuration."""
    cfg = SimConfig(
        d=25,
        alpha=0.88,
        alpha0=2.5,
        alpha1=0.02,
        beta=15.0,
        T=20,
        trials=50,
        sigmas=(2.0, 1.0),
        seed=999,
    )
    
    output_path = tmp_path / "config_content.json"
    dump_config_json(str(output_path), cfg)
    
    with open(output_path, "r") as f:
        loaded_cfg = json.load(f)
    
    assert loaded_cfg["d"] == 25
    assert loaded_cfg["alpha"] == 0.88
    assert loaded_cfg["alpha0"] == 2.5
    assert loaded_cfg["alpha1"] == 0.02
    assert loaded_cfg["beta"] == 15.0
    assert loaded_cfg["T"] == 20
    assert loaded_cfg["trials"] == 50
    assert loaded_cfg["sigmas"] == [2.0, 1.0]
    assert loaded_cfg["seed"] == 999


def test_dump_config_json_is_valid_json(tmp_path):
    """Test that the output is valid JSON."""
    cfg = SimConfig()
    
    output_path = tmp_path / "valid.json"
    dump_config_json(str(output_path), cfg)
    
    # Should be able to parse without error
    with open(output_path, "r") as f:
        data = json.load(f)
    
    assert isinstance(data, dict)


def test_dump_config_json_roundtrip(tmp_path):
    """Test that config can be saved and reconstructed."""
    cfg_original = SimConfig(
        d=36,
        alpha=0.92,
        alpha0=3.5,
        alpha1=0.05,
        beta=25.0,
        T=15,
        trials=30,
        sigmas=(1.5, 0.75),
        seed=777,
    )
    
    output_path = tmp_path / "roundtrip_config.json"
    dump_config_json(str(output_path), cfg_original)
    
    # Load and reconstruct
    with open(output_path, "r") as f:
        cfg_dict = json.load(f)
    
    # Convert sigmas list back to tuple
    cfg_dict["sigmas"] = tuple(cfg_dict["sigmas"])
    cfg_reconstructed = SimConfig(**cfg_dict)
    
    # Verify all fields match
    assert cfg_reconstructed.d == cfg_original.d
    assert cfg_reconstructed.alpha == cfg_original.alpha
    assert cfg_reconstructed.alpha0 == cfg_original.alpha0
    assert cfg_reconstructed.alpha1 == cfg_original.alpha1
    assert cfg_reconstructed.beta == cfg_original.beta
    assert cfg_reconstructed.T == cfg_original.T
    assert cfg_reconstructed.trials == cfg_original.trials
    assert cfg_reconstructed.sigmas == cfg_original.sigmas
    assert cfg_reconstructed.seed == cfg_original.seed


def test_dump_config_json_overwrite(tmp_path):
    """Test that dump_config_json overwrites existing files."""
    cfg1 = SimConfig(d=16, seed=1)
    cfg2 = SimConfig(d=25, seed=2)
    
    output_path = tmp_path / "overwrite_config.json"
    
    # Save first config
    dump_config_json(str(output_path), cfg1)
    
    # Overwrite with second config
    dump_config_json(str(output_path), cfg2)
    
    # Load and verify it's the second config
    with open(output_path, "r") as f:
        cfg_dict = json.load(f)
    
    assert cfg_dict["d"] == 25
    assert cfg_dict["seed"] == 2


def test_save_npz_and_json_together(tmp_path):
    """Test saving both .npz and .json files together (typical workflow)."""
    cfg = SimConfig(d=16, T=10, trials=5, sigmas=(1.0, 0.5), seed=42)
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    npz_path = tmp_path / "dataset.npz"
    json_path = tmp_path / "dataset_config.json"
    
    save_npz(str(npz_path), X, Z, coords, Sigma, cfg)
    dump_config_json(str(json_path), cfg)
    
    # Both files should exist
    assert npz_path.exists()
    assert json_path.exists()
    
    # Load and verify consistency
    data = np.load(str(npz_path))
    with open(json_path, "r") as f:
        cfg_dict = json.load(f)
    
    # Metadata should match
    assert data["d"][0] == cfg_dict["d"]
    assert data["T"][0] == cfg_dict["T"]
    assert data["trials"][0] == cfg_dict["trials"]
    assert data["seed"][0] == cfg_dict["seed"]


def test_npz_file_without_extension(tmp_path):
    """Test that save_npz works without .npz extension."""
    cfg = SimConfig(d=16, T=5, trials=2)
    X, Z, coords, Sigma = simulate_dataset(cfg)
    
    # Provide path without extension
    output_path = tmp_path / "data_no_ext"
    save_npz(str(output_path), X, Z, coords, Sigma, cfg)
    
    # NumPy should add .npz extension
    expected_path = tmp_path / "data_no_ext.npz"
    assert expected_path.exists()


def test_json_indent_formatting(tmp_path):
    """Test that JSON file is properly indented for readability."""
    cfg = SimConfig()
    
    output_path = tmp_path / "formatted.json"
    dump_config_json(str(output_path), cfg)
    
    with open(output_path, "r") as f:
        content = f.read()
    
    # Should have multiple lines (not a single line)
    lines = content.strip().split("\n")
    assert len(lines) > 5  # At least several lines
    
    # Should have indentation
    assert any("  " in line for line in lines)
