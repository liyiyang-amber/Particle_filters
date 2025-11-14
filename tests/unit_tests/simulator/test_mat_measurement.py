import numpy as np
import pytest

from simulator.simulator_Multi_acoustic_tracking import (
    acoustic_measurement_model,
    ScenarioConfig,
    DynamicsConfig,
    simulate_acoustic_dataset,
)

def test_acoustic_monotonic_with_distance_single_sensor():
    """Test that amplitude decreases with increasing distance from sensor."""
    # Single time, single target, one sensor at origin
    sensors = np.array([[0.0, 0.0]])
    psi = 10.0
    d0 = 0.1

    # Position at distance 1 vs distance 2 along x-axis
    P_close = np.array([[[1.0, 0.0]]])  # (T=1, C=1, 2)
    P_far   = np.array([[[2.0, 0.0]]])

    Z_close = acoustic_measurement_model(P_close, sensors, psi=psi, d0=d0)
    Z_far   = acoustic_measurement_model(P_far, sensors, psi=psi, d0=d0)

    assert Z_close.shape == (1, 1)
    assert Z_far.shape == (1, 1)
    assert Z_close[0, 0] > Z_far[0, 0]  # closer => larger amplitude

def test_vectorized_equals_naive_sum_noiseless():
    """Test that vectorized acoustic model matches naive summation."""
    rng = np.random.default_rng(123)
    T, C, S = 5, 3, 4
    positions = rng.uniform(low=-5.0, high=5.0, size=(T, C, 2))
    sensors = rng.uniform(low=-5.0, high=5.0, size=(S, 2))
    psi, d0 = 7.5, 0.2

    Z_vec = acoustic_measurement_model(positions, sensors, psi=psi, d0=d0)
    assert Z_vec.shape == (T, S)

    # Naive computation
    Z_naive = np.zeros((T, S))
    for t in range(T):
        for s in range(S):
            z = 0.0
            for c in range(C):
                diff = positions[t, c] - sensors[s]
                d2 = np.sum(diff**2)
                z += psi / (d2 + d0)
            Z_naive[t, s] = z

    np.testing.assert_allclose(Z_vec, Z_naive, rtol=1e-12, atol=1e-12)

def test_noiseless_measurement_consistency():
    """Test that acoustic model produces consistent noiseless measurements."""
    rng = np.random.default_rng(42)
    T, C, S = 3, 2, 3
    positions = rng.normal(size=(T, C, 2))
    sensors = rng.normal(size=(S, 2))
    psi, d0 = 5.0, 0.1

    Z1 = acoustic_measurement_model(positions, sensors, psi=psi, d0=d0)
    Z2 = acoustic_measurement_model(positions, sensors, psi=psi, d0=d0)
    np.testing.assert_allclose(Z1, Z2)

def test_dataset_matches_model():
    """Ensure simulator Z equals direct model output."""
    cfg = ScenarioConfig(n_targets=2, n_steps=6, sensor_grid_shape=(2, 3), seed=2024)
    dyn = DynamicsConfig(dt=1.0)
    data = simulate_acoustic_dataset(cfg, dyn)

    Z = data["Z"]
    P = data["P"]
    S = data["S"]
    psi, d0 = cfg.psi, cfg.d0

    Z_model = acoustic_measurement_model(P, S, psi=psi, d0=d0)
    np.testing.assert_allclose(Z, Z_model, rtol=1e-12, atol=1e-12)