import numpy as np
import pytest

from simulator.simulator_Multi_acoustic_tracking import (
    ScenarioConfig,
    DynamicsConfig,
    simulate_acoustic_dataset,
    make_sensor_grid,
    article_initial_states,
    build_cv_transition,
    article_process_noise_cov,
)

def test_shapes_and_meta_basic():
    """Test that simulated dataset has correct shapes and metadata."""
    cfg = ScenarioConfig(
        n_targets=3,
        n_steps=12,
        area_xy=(40.0, 40.0),
        sensor_grid_shape=(3, 4),
        psi=10.0,
        d0=0.1,
        seed=123,
        use_article_init=False,
    )
    dyn = DynamicsConfig(dt=0.5)
    data = simulate_acoustic_dataset(cfg, dyn)
    assert set(data.keys()) == {"X", "P", "S", "Z", "meta"}

    X, P, S, Z, meta = data["X"], data["P"], data["S"], data["Z"], data["meta"]

    assert X.shape == (12, 3, 4)
    assert P.shape == (12, 3, 2)
    assert S.shape == (3 * 4, 2)
    assert Z.shape == (12, 12)
    assert meta.shape == (5,)
    np.testing.assert_allclose(meta, np.array([40.0, 40.0, 10.0, 0.1, 0.5]))

    # Finite
    assert np.isfinite(X).all()
    assert np.isfinite(P).all()
    assert np.isfinite(S).all()
    assert np.isfinite(Z).all()
    assert np.isfinite(meta).all()

def test_determinism_with_seed():
    """Test that simulation is deterministic with the same seed."""
    cfg1 = ScenarioConfig(n_targets=3, n_steps=10, seed=999, use_article_init=False)
    dyn1 = DynamicsConfig(dt=1.0)
    d1 = simulate_acoustic_dataset(cfg1, dyn1)

    cfg2 = ScenarioConfig(n_targets=3, n_steps=10, seed=999, use_article_init=False)
    dyn2 = DynamicsConfig(dt=1.0)
    d2 = simulate_acoustic_dataset(cfg2, dyn2)

    # Same seed => same outputs
    np.testing.assert_allclose(d1["X"], d2["X"])
    np.testing.assert_allclose(d1["P"], d2["P"])
    np.testing.assert_allclose(d1["S"], d2["S"])
    np.testing.assert_allclose(d1["Z"], d2["Z"])
    np.testing.assert_allclose(d1["meta"], d2["meta"])

    # Different seed => different X or Z (very likely)
    cfg3 = ScenarioConfig(n_targets=3, n_steps=10, seed=1000, use_article_init=False)
    d3 = simulate_acoustic_dataset(cfg3, dyn1)
    assert not np.allclose(d1["X"], d3["X"]) or not np.allclose(d1["Z"], d3["Z"])

def test_article_initial_states_applied_when_requested():
    """Test that article initial states are used when requested."""
    cfg = ScenarioConfig(n_targets=4, n_steps=1, use_article_init=True)
    dyn = DynamicsConfig(dt=1.0)
    data = simulate_acoustic_dataset(cfg, dyn)
    np.testing.assert_allclose(data["X"][0], article_initial_states(4))

def test_sensor_grid_geometry_and_bounds():
    """Test that sensor grid is created with correct shape and bounds."""
    sensors = make_sensor_grid((20.0, 10.0), (2, 5))
    assert sensors.shape == (10, 2)
    xs, ys = sensors[:, 0], sensors[:, 1]
    assert np.isclose(xs.min(), 0.0)
    assert np.isclose(xs.max(), 20.0)
    assert np.isclose(ys.min(), 0.0)
    assert np.isclose(ys.max(), 10.0)
    # Grid should include boundaries and be a Cartesian product
    assert len(np.unique(xs)) == 5
    assert len(np.unique(ys)) == 2

def test_transition_and_noise_covariance_properties():
    """Test properties of CV transition matrix and process noise covariance."""
    F = build_cv_transition(0.25)
    assert F.shape == (4, 4)
    # Check structure
    dt = 0.25
    F_true = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
    np.testing.assert_allclose(F, F_true)

    V = article_process_noise_cov()
    assert V.shape == (4, 4)
    # Symmetry and PSD
    np.testing.assert_allclose(V, V.T, rtol=1e-12, atol=1e-12)
    w, _ = np.linalg.eigh(V)
    assert (w >= -1e-12).all()