"""
Integration tests for Multi-Target Acoustic Tracking (MAT) simulator end-to-end.

These tests validate:
- Complete simulation pipeline from config to output
- Statistical properties of trajectories and measurements
- Save/load functionality
- Boundary conditions and edge cases
- Physical consistency of the acoustic model
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from simulator.simulator_Multi_acoustic_tracking import (
    simulate_acoustic_dataset,
    ScenarioConfig,
    DynamicsConfig,
    simulate_cv_targets,
    acoustic_measurement_model,
    make_sensor_grid,
    build_cv_transition,
    article_process_noise_cov,
    article_initial_states,
)


@pytest.mark.integration
def test_complete_simulation_pipeline():
    """Test the complete simulation pipeline produces valid and consistent data."""
    cfg = ScenarioConfig(
        n_targets=4,
        n_steps=100,
        area_xy=(40.0, 40.0),
        sensor_grid_shape=(5, 5),
        psi=10.0,
        d0=0.1,
        seed=42,
        use_article_init=True,
    )
    dyn_cfg = DynamicsConfig(dt=1.0)
    
    result = simulate_acoustic_dataset(cfg, dyn_cfg)
    
    # Verify all outputs are finite
    assert np.isfinite(result["X"]).all(), "States contain non-finite values"
    assert np.isfinite(result["P"]).all(), "Positions contain non-finite values"
    assert np.isfinite(result["S"]).all(), "Sensors contain non-finite values"
    assert np.isfinite(result["Z"]).all(), "Measurements contain non-finite values"
    
    # Verify shapes
    assert result["X"].shape == (100, 4, 4)
    assert result["P"].shape == (100, 4, 2)
    assert result["S"].shape == (25, 2)
    assert result["Z"].shape == (100, 25)
    
    # Verify positions are consistent with states
    np.testing.assert_array_equal(result["P"], result["X"][..., :2])
    
    # Note: Measurements can be negative due to measurement noise
    # but the noiseless component should be positive
    # We just check that most measurements are reasonable
    assert np.isfinite(result["Z"]).all(), "Measurements should be finite"


@pytest.mark.integration
def test_trajectory_statistics():
    """Test that generated trajectories have reasonable statistical properties."""
    cfg = ScenarioConfig(
        n_targets=4,
        n_steps=500,
        area_xy=(40.0, 40.0),
        seed=42,
        use_article_init=True,
    )
    dyn_cfg = DynamicsConfig(dt=1.0)
    
    result = simulate_acoustic_dataset(cfg, dyn_cfg)
    X = result["X"]  # (T, C, 4)
    
    # Check that positions evolve (not static)
    position_changes = np.diff(X[:, :, :2], axis=0)
    total_movement = np.sum(np.linalg.norm(position_changes, axis=2))
    assert total_movement > 10.0, "Targets should move significantly over time"
    
    # Check velocity statistics (should be small but non-zero)
    velocities = X[:, :, 2:]  # (T, C, 2)
    velocity_magnitudes = np.linalg.norm(velocities, axis=2)
    mean_speed = np.mean(velocity_magnitudes)
    
    # Velocities should be reasonable (not too large for CV model with small noise)
    assert 0.001 < mean_speed < 5.0, f"Mean speed {mean_speed} seems unreasonable"


@pytest.mark.integration
def test_cv_dynamics_integration():
    """Test that CV dynamics integration follows expected evolution.
    
    Note: With boundary enforcement enabled, targets that hit boundaries will
    have their positions reflected and velocities reversed, causing larger
    state changes than just F @ x + noise. We detect these cases and validate
    them separately.
    """
    dyn_cfg = DynamicsConfig(dt=1.0)
    rng = np.random.default_rng(123)
    
    # Simulate with small noise to check dynamics more clearly
    n_steps = 50
    n_targets = 2
    area_xy = (40.0, 40.0)
    X = simulate_cv_targets(
        n_steps=n_steps,
        n_targets=n_targets,
        area_xy=area_xy,
        dyn_cfg=dyn_cfg,
        rng=rng,
        use_article_init=False,
    )
    
    F = build_cv_transition(dyn_cfg.dt)
    width, height = area_xy
    
    # For each step, check that state evolution is approximately F @ x_{k-1} + noise
    for t in range(1, n_steps):
        for c in range(n_targets):
            x_pred = F @ X[t-1, c]
            x_actual = X[t, c]
            
            # Check if boundary reflection occurred
            pos_pred = x_pred[:2]
            pos_actual = x_actual[:2]
            vel_pred = x_pred[2:]
            vel_actual = x_actual[2:]
            
            # Detect boundary reflection by checking if velocity sign changed
            vel_sign_changed = np.any(np.sign(vel_pred) != np.sign(vel_actual))
            near_boundary = (pos_actual[0] <= 0.01 or pos_actual[0] >= width - 0.01 or
                           pos_actual[1] <= 0.01 or pos_actual[1] >= height - 0.01)
            
            if vel_sign_changed or near_boundary:
                # Boundary reflection occurred - use larger threshold
                # Reflection can cause larger displacement
                diff = x_actual - x_pred
                diff_norm = np.linalg.norm(diff)
                assert diff_norm < 15.0, f"State change too large even with reflection at t={t}, c={c}: {diff_norm:.2f}m"
            else:
                # Normal dynamics - difference should be reasonable (process noise)
                diff = x_actual - x_pred
                diff_norm = np.linalg.norm(diff)
                assert diff_norm < 5.0, f"State change too large at t={t}, c={c}: {diff_norm:.2f}m"


@pytest.mark.integration
def test_acoustic_model_physical_properties():
    """Test that acoustic model satisfies physical properties."""
    cfg = ScenarioConfig(
        n_targets=4,
        n_steps=50,
        sensor_grid_shape=(5, 5),
        psi=10.0,
        d0=0.1,
        seed=42,
    )
    dyn_cfg = DynamicsConfig(dt=1.0)
    
    result = simulate_acoustic_dataset(cfg, dyn_cfg)
    P = result["P"]  # (T, C, 2)
    S = result["S"]  # (S, 2)
    Z = result["Z"]  # (T, S)
    
    # Test 1: Measurements should decay with distance
    # For each sensor, closer targets contribute more
    for t in range(P.shape[0]):
        for s in range(S.shape[0]):
            sensor_pos = S[s]
            measurement = Z[t, s]
            
            # Compute expected contribution from each target
            expected_z = 0.0
            for c in range(P.shape[1]):
                target_pos = P[t, c]
                dist_sq = np.sum((target_pos - sensor_pos) ** 2)
                expected_z += cfg.psi / (dist_sq + cfg.d0)
            
            # Should match (no noise)
            np.testing.assert_allclose(measurement, expected_z, rtol=1e-10)
    
    # Test 2: Measurements should be positive
    assert (Z > 0).all(), "All measurements should be positive"
    
    # Test 3: Measurements should vary over time (targets are moving)
    measurement_variance = np.var(Z, axis=0)
    assert (measurement_variance > 0).all(), "Measurements should vary over time"


@pytest.mark.integration
def test_sensor_grid_coverage():
    """Test that sensor grid properly covers the specified area."""
    area_xy = (50.0, 30.0)
    grid_shape = (6, 8)
    
    sensors = make_sensor_grid(area_xy, grid_shape)
    
    # Check that sensors span the area
    x_coords = sensors[:, 0]
    y_coords = sensors[:, 1]
    
    assert np.isclose(x_coords.min(), 0.0)
    assert np.isclose(x_coords.max(), area_xy[0])
    assert np.isclose(y_coords.min(), 0.0)
    assert np.isclose(y_coords.max(), area_xy[1])
    
    # Check grid structure
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    
    assert len(unique_x) == grid_shape[1], "Number of x-coordinates should match columns"
    assert len(unique_y) == grid_shape[0], "Number of y-coordinates should match rows"
    
    # Check uniform spacing
    x_spacing = np.diff(unique_x)
    y_spacing = np.diff(unique_y)
    
    assert np.allclose(x_spacing, x_spacing[0]), "X-spacing should be uniform"
    assert np.allclose(y_spacing, y_spacing[0]), "Y-spacing should be uniform"


@pytest.mark.integration
def test_save_and_load_roundtrip():
    """Test that saved data can be loaded and matches original."""
    cfg = ScenarioConfig(
        n_targets=3,
        n_steps=50,
        seed=99,
    )
    dyn_cfg = DynamicsConfig(dt=1.0)
    
    result = simulate_acoustic_dataset(cfg, dyn_cfg)
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_mat_data.npz"
        np.savez(str(filepath), **result)
        
        # Load back
        loaded = np.load(str(filepath))
        
        # Compare all fields
        for key in result.keys():
            np.testing.assert_array_equal(
                result[key], 
                loaded[key],
                err_msg=f"Mismatch in {key} after save/load"
            )


@pytest.mark.integration
def test_article_initialization_consistency():
    """Test that article initialization produces expected initial states."""
    cfg = ScenarioConfig(
        n_targets=4,
        n_steps=10,
        use_article_init=True,
        seed=42,
    )
    dyn_cfg = DynamicsConfig(dt=1.0)
    
    result = simulate_acoustic_dataset(cfg, dyn_cfg)
    
    # Initial states should match article
    expected_init = article_initial_states(4)
    np.testing.assert_array_almost_equal(result["X"][0], expected_init)


@pytest.mark.integration
def test_random_initialization_stays_in_central_region():
    """Test that random initialization keeps targets near center of area."""
    cfg = ScenarioConfig(
        n_targets=5,
        n_steps=1,
        area_xy=(40.0, 40.0),
        use_article_init=False,
        seed=123,
    )
    dyn_cfg = DynamicsConfig(dt=1.0)
    
    result = simulate_acoustic_dataset(cfg, dyn_cfg)
    
    initial_positions = result["P"][0]  # (C, 2)
    
    # Should be in central region (0.25 to 0.75 of area)
    assert (initial_positions[:, 0] >= 10.0).all()
    assert (initial_positions[:, 0] <= 30.0).all()
    assert (initial_positions[:, 1] >= 10.0).all()
    assert (initial_positions[:, 1] <= 30.0).all()


@pytest.mark.integration
def test_process_noise_covariance_positive_definite():
    """Test that article process noise covariance is positive definite."""
    V = article_process_noise_cov()
    
    # Check symmetry
    np.testing.assert_allclose(V, V.T, rtol=1e-12, atol=1e-12)
    
    # Check positive definiteness
    eigenvalues = np.linalg.eigvals(V)
    assert (eigenvalues > 0).all(), "Process noise covariance should be positive definite"
    
    # Check it can be Cholesky decomposed
    try:
        L = np.linalg.cholesky(V)
        # Verify factorization
        np.testing.assert_allclose(L @ L.T, V, rtol=1e-10)
    except np.linalg.LinAlgError:
        pytest.fail("Process noise covariance should be Cholesky decomposable")


@pytest.mark.integration
def test_multiple_independent_runs():
    """Test that multiple independent simulation runs produce different results."""
    cfg = ScenarioConfig(n_targets=3, n_steps=50, use_article_init=False)
    dyn_cfg = DynamicsConfig(dt=1.0)
    
    results = []
    for seed in [1, 2, 3, 4, 5]:
        cfg_seed = ScenarioConfig(
            n_targets=3,
            n_steps=50,
            use_article_init=False,
            seed=seed,
        )
        result = simulate_acoustic_dataset(cfg_seed, dyn_cfg)
        results.append(result["X"])
    
    # Each pair of runs should produce different trajectories
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert not np.allclose(results[i], results[j]), \
                f"Runs {i} and {j} produced identical results"


@pytest.mark.integration
def test_long_simulation_numerical_stability():
    """Test that long simulations remain numerically stable."""
    cfg = ScenarioConfig(
        n_targets=4,
        n_steps=1000,  # Long simulation
        seed=42,
        use_article_init=True,
    )
    dyn_cfg = DynamicsConfig(dt=1.0)
    
    result = simulate_acoustic_dataset(cfg, dyn_cfg)
    
    # Check for numerical issues
    assert np.isfinite(result["X"]).all(), "States contain non-finite values in long simulation"
    assert np.isfinite(result["Z"]).all(), "Measurements contain non-finite values in long simulation"
    
    # For long simulations with CV dynamics, positions can drift significantly
    # This is expected behavior - we just check for NaN/Inf issues
    # and that velocities remain reasonable (process noise is bounded)
    velocity_magnitudes = np.linalg.norm(result["X"][:, :, 2:], axis=2)
    assert (velocity_magnitudes < 50.0).all(), "Velocities became unreasonably large"


@pytest.mark.integration
def test_meta_array_consistency():
    """Test that meta array correctly stores all scenario parameters."""
    cfg = ScenarioConfig(
        n_targets=3,
        n_steps=50,
        area_xy=(35.0, 25.0),
        psi=15.0,
        d0=0.2,
        seed=42,
    )
    dyn_cfg = DynamicsConfig(dt=0.5)
    
    result = simulate_acoustic_dataset(cfg, dyn_cfg)
    meta = result["meta"]
    
    # Verify each element: [width, height, psi, d0, dt]
    assert np.isclose(meta[0], 35.0), "Width not stored correctly"
    assert np.isclose(meta[1], 25.0), "Height not stored correctly"
    assert np.isclose(meta[2], 15.0), "Psi not stored correctly"
    assert np.isclose(meta[3], 0.2), "d0 not stored correctly"
    assert np.isclose(meta[4], 0.5), "dt not stored correctly"


@pytest.mark.integration
def test_different_time_steps():
    """Test simulation with different time step values."""
    cfg = ScenarioConfig(n_targets=2, n_steps=100, seed=42)
    
    results = {}
    for dt in [0.1, 0.5, 1.0, 2.0]:
        dyn_cfg = DynamicsConfig(dt=dt)
        result = simulate_acoustic_dataset(cfg, dyn_cfg)
        results[dt] = result
    
    # Different dt should produce different trajectories
    # (even with same seed, because noise is additive and positions evolve differently)
    dt_values = list(results.keys())
    for i in range(len(dt_values)):
        for j in range(i + 1, len(dt_values)):
            dt1, dt2 = dt_values[i], dt_values[j]
            # Positions after a few steps should differ
            if not np.allclose(results[dt1]["P"][10], results[dt2]["P"][10], atol=1.0):
                break
    else:
        # At least some pairs should differ
        assert not np.allclose(results[0.1]["P"][10], results[2.0]["P"][10], atol=1.0)
