"""
Integration tests for all filter types with Multi-Target Acoustic Tracking simulator.

Tests the integration of EKF, UKF, EDH, and LEDH filters with the MAT simulator.
Each filter tracks individual targets in the multi-target scenario.
"""

import numpy as np
import pytest

from simulator.simulator_Multi_acoustic_tracking import (
    ScenarioConfig,
    DynamicsConfig,
    simulate_acoustic_dataset,
)

from models.extended_kalman_filter import ExtendedKalmanFilter, EKFState
from models.unscented_kalman_filter import UnscentedKalmanFilter, UKFState
from models.LEDH_particle_filter import LEDHFlowPF, LEDHConfig
from models.EDH_particle_filter import EDHFlowPF, EDHConfig

# Small filter process noise (more uncertain than simulator's)
Q_FILTER = np.array([
    [3.0, 0.0, 0.1, 0.0],
    [0.0, 3.0, 0.0, 0.1],
    [0.1, 0.0, 0.03, 0.0],
    [0.0, 0.1, 0.0, 0.03],
])

def build_F(dt=1.0):
    return np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

def g_filter(x, u=None, v=None):
    F = build_F(1.0)
    if v is None:
        v = np.zeros_like(x)
    return F @ x + v

def jac_g(x, u=None):
    return build_F(1.0)

def make_h_and_jac(S, psi, d0):
    n_sensors = S.shape[0]
    def h_acoustic(x):
        pos = x[:2]
        z = np.zeros(n_sensors)
        for s in range(n_sensors):
            dist_sq = np.sum((pos - S[s])**2)
            z[s] = psi / (dist_sq + d0)
        return z
    def jac_h_acoustic(x):
        pos = x[:2]
        H = np.zeros((n_sensors, 4))
        for s in range(n_sensors):
            diff = pos - S[s]
            dist_sq = np.sum(diff**2)
            denom = (dist_sq + d0) ** 2
            H[s, 0] = -2.0 * psi * diff[0] / denom
            H[s, 1] = -2.0 * psi * diff[1] / denom
        return H
    return h_acoustic, jac_h_acoustic

def log_trans_pdf(xk, xkm1):
    F = build_F(1.0)
    diff = xk - F @ xkm1
    log_det = np.log(np.linalg.det(2 * np.pi * Q_FILTER))
    return -0.5 * (diff.T @ np.linalg.solve(Q_FILTER, diff) + log_det)

def make_log_like_pdf(h, R):
    def log_like_pdf(z, x):
        diff = z - h(x)
        log_det = np.log(np.linalg.det(2 * np.pi * R))
        return -0.5 * (diff.T @ np.linalg.solve(R, diff) + log_det)
    return log_like_pdf

class EKFTracker:
    def __init__(self, ekf, initial_state):
        self.ekf = ekf
        self.state = initial_state
        self.past_mean = initial_state.mean.copy()
    def predict(self):
        self.past_mean = self.state.mean.copy()
        self.state = self.ekf.predict(self.state, u=None)
        return self.state.mean, self.state.cov
    def update(self, z_k):
        self.state = self.ekf.update(self.state, z_k)
        return self.state.mean, self.state.cov
    def get_past_mean(self):
        return self.past_mean

@pytest.mark.integration
def test_simulator_noiseless_matches_sum_of_single_target_contributions():
    # With zero noise, Z should equal the sum over targets of per-target contributions
    cfg = ScenarioConfig(n_targets=3, n_steps=6, sensor_grid_shape=(3, 3), seed=7)
    dyn = DynamicsConfig(dt=1.0)
    data = simulate_acoustic_dataset(cfg, dyn)
    P, S, Z = data["P"], data["S"], data["Z"]
    psi, d0 = cfg.psi, cfg.d0

    # Recompute Z manually
    T, C = P.shape[:2]
    S_count = S.shape[0]
    Z_manual = np.zeros((T, S_count))
    for t in range(T):
        for s in range(S_count):
            z_sum = 0.0
            for c in range(C):
                diff = P[t, c] - S[s]
                d2 = np.sum(diff**2)
                z_sum += psi / (d2 + d0)
            Z_manual[t, s] = z_sum

    np.testing.assert_allclose(Z, Z_manual, rtol=1e-12, atol=1e-12)

@pytest.mark.integration
def test_small_ledh_and_edh_runs_produce_finite_results():
    # Small scenario
    cfg = ScenarioConfig(n_targets=3, n_steps=8, sensor_grid_shape=(3, 3), seed=11)
    dyn = DynamicsConfig(dt=1.0)
    data = simulate_acoustic_dataset(cfg, dyn)
    X_true, P_true, S, Z, meta = data["X"], data["P"], data["S"], data["Z"], data["meta"]

    psi, d0 = meta[2], meta[3]
    n_sensors = S.shape[0]
    R = (0.1 ** 2) * np.eye(n_sensors)  # Use fixed noise std of 0.1

    h, jac_h = make_h_and_jac(S, psi, d0)
    log_like_pdf = make_log_like_pdf(h, R)

    # Create per-target trackers and filters (LEDH and EDH)
    n_particles = 100
    P_init = np.diag([10.0**2, 10.0**2, 1.0**2, 1.0**2])

    ledh_filters = []
    edh_filters = []

    for c in range(cfg.n_targets):
        ekf = ExtendedKalmanFilter(g=g_filter, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q_FILTER, R=R)
        ekf_state = EKFState(mean=X_true[0, c].copy(), cov=P_init.copy(), t=0)  # init near truth to keep test stable
        tracker = EKFTracker(ekf, ekf_state)

        ledh_cfg = LEDHConfig(n_particles=n_particles, n_lambda_steps=3, resample_ess_ratio=0.5, rng=np.random.default_rng(100 + c))
        ledh_pf = LEDHFlowPF(
            tracker=tracker, g=g_filter, h=h, jacobian_h=jac_h,
            log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=ledh_cfg
        )
        state_ledh = ledh_pf.init_from_gaussian(ekf_state.mean, P_init.copy())
        ledh_filters.append((ledh_pf, state_ledh, ledh_cfg))

        edh_cfg = EDHConfig(n_particles=n_particles, n_lambda_steps=3, resample_ess_ratio=0.5, flow_integrator="rk4", rng=np.random.default_rng(200 + c))
        edh_pf = EDHFlowPF(
            tracker=tracker, g=g_filter, h=h, jacobian_h=jac_h,
            log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=edh_cfg
        )
        state_edh = edh_pf.init_from_gaussian(ekf_state.mean, P_init.copy())
        edh_filters.append((edh_pf, state_edh, edh_cfg))

    # Run for a few steps and check numerical sanity
    steps = min(5, cfg.n_steps - 1)
    for t in range(1, 1 + steps):
        z_t = Z[t]

        for i in range(len(ledh_filters)):
            pf, state, cfg_obj = ledh_filters[i]
            def proc_noise_sampler(N, nx):
                return cfg_obj.rng.multivariate_normal(np.zeros(nx), Q_FILTER, size=N)
            state = pf.step(state, z_t, process_noise_sampler=proc_noise_sampler)
            ledh_filters[i] = (pf, state, cfg_obj)

            # Sanity checks
            assert np.isfinite(state.particles).all()
            assert np.isfinite(state.weights).all()
            assert np.isfinite(state.mean).all()
            assert np.isfinite(state.cov).all()
            np.testing.assert_allclose(np.sum(state.weights), 1.0, rtol=1e-9, atol=1e-9)

        for i in range(len(edh_filters)):
            pf, state, cfg_obj = edh_filters[i]
            def proc_noise_sampler(N, nx):
                return cfg_obj.rng.multivariate_normal(np.zeros(nx), Q_FILTER, size=N)
            state = pf.step(state, z_t, process_noise_sampler=proc_noise_sampler)
            edh_filters[i] = (pf, state, cfg_obj)

            # Sanity checks
            assert np.isfinite(state.particles).all()
            assert np.isfinite(state.weights).all()
            assert np.isfinite(state.mean).all()
            assert np.isfinite(state.cov).all()
            np.testing.assert_allclose(np.sum(state.weights), 1.0, rtol=1e-9, atol=1e-9)


@pytest.mark.integration
def test_ekf_mat_single_target_tracking():
    """Test EKF on MAT simulator with single target for tracking accuracy."""
    # Single target scenario for clean tracking (no interference from other targets)
    cfg = ScenarioConfig(
        n_targets=1,
        n_steps=50,
        sensor_grid_shape=(4, 4),
        seed=42,
        use_article_init=False,
    )
    dyn = DynamicsConfig(dt=1.0)
    data = simulate_acoustic_dataset(cfg, dyn)
    
    X_true, P_true, S, Z, meta = data["X"], data["P"], data["S"], data["Z"], data["meta"]
    psi, d0 = meta[2], meta[3]
    n_sensors = S.shape[0]
    R = (0.05 ** 2) * np.eye(n_sensors)  # Use fixed noise std of 0.05
    
    h, jac_h = make_h_and_jac(S, psi, d0)
    
    # Initialize EKF close to truth for stable tracking
    x0_init = X_true[0, 0].copy()
    P0 = np.diag([5.0, 5.0, 1.0, 1.0])
    
    ekf = ExtendedKalmanFilter(
        g=g_filter,
        h=h,
        jac_g=jac_g,
        jac_h=jac_h,
        Q=Q_FILTER,
        R=R,
        joseph=True,
        jitter=1e-8,
    )
    
    state = EKFState(mean=x0_init, cov=P0, t=0)
    
    # Track for several steps
    position_errors = []
    for t in range(1, min(30, cfg.n_steps)):
        z_t = Z[t]
        state = ekf.step(state, z_t, u=None)
        
        # Compute position error
        pos_error = np.linalg.norm(state.mean[:2] - X_true[t, 0, :2])
        position_errors.append(pos_error)
        
        # Sanity checks
        assert np.isfinite(state.mean).all()
        assert np.isfinite(state.cov).all()
        assert np.all(np.linalg.eigvals(state.cov) > -1e-10)  # PSD
    
    # Check that EKF produces finite results and maintains reasonable error
    assert np.all(np.isfinite(position_errors))
    mean_error = np.mean(position_errors)
    assert mean_error < 10.0, f"EKF mean tracking error {mean_error} should be reasonable for single target"


@pytest.mark.integration
def test_ukf_mat_single_target_tracking():
    """Test UKF on MAT simulator with single target for tracking accuracy."""
    # Single target scenario
    cfg = ScenarioConfig(
        n_targets=1,
        n_steps=50,
        sensor_grid_shape=(4, 4),
        seed=42,
        use_article_init=False,
    )
    dyn = DynamicsConfig(dt=1.0)
    data = simulate_acoustic_dataset(cfg, dyn)
    
    X_true, P_true, S, Z, meta = data["X"], data["P"], data["S"], data["Z"], data["meta"]
    psi, d0 = meta[2], meta[3]
    n_sensors = S.shape[0]
    R = (0.05 ** 2) * np.eye(n_sensors)  # Use fixed noise std of 0.05
    
    h, _ = make_h_and_jac(S, psi, d0)
    
    # Initialize UKF close to truth
    x0_init = X_true[0, 0].copy()
    P0 = np.diag([5.0, 5.0, 1.0, 1.0])
    
    ukf = UnscentedKalmanFilter(
        g=g_filter,
        h=h,
        Q=Q_FILTER,
        R=R,
        alpha=1e-3,
        beta=2.0,
        kappa=0.0,
        jitter=1e-9,
    )
    
    state = UKFState(mean=x0_init, cov=P0, t=0)
    
    # Track for several steps
    position_errors = []
    for t in range(1, min(30, cfg.n_steps)):
        z_t = Z[t]
        state = ukf.step(state, z_t, u=None)
        
        # Compute position error
        pos_error = np.linalg.norm(state.mean[:2] - X_true[t, 0, :2])
        position_errors.append(pos_error)
        
        # Sanity checks
        assert np.isfinite(state.mean).all()
        assert np.isfinite(state.cov).all()
        assert np.all(np.linalg.eigvals(state.cov) > -1e-10)  # PSD
    
    # Check that UKF produces finite results and maintains reasonable error
    assert np.all(np.isfinite(position_errors))
    mean_error = np.mean(position_errors)
    assert mean_error < 10.0, f"UKF mean tracking error {mean_error} should be reasonable for single target"


@pytest.mark.integration
def test_ekf_ukf_comparison_mat():
    """Compare EKF and UKF tracking performance on MAT simulator."""
    # Use single target configuration for fair comparison (avoid multi-target interference)
    cfg = ScenarioConfig(
        n_targets=1,
        n_steps=40,
        sensor_grid_shape=(4, 4),
        seed=42,  # Use same seed as other single-target tests
        use_article_init=False,
    )
    dyn = DynamicsConfig(dt=1.0)
    data = simulate_acoustic_dataset(cfg, dyn)
    
    X_true, S, Z, meta = data["X"], data["S"], data["Z"], data["meta"]
    psi, d0 = meta[2], meta[3]
    n_sensors = S.shape[0]
    R = (0.05 ** 2) * np.eye(n_sensors)  # Use fixed noise std of 0.05
    
    h, jac_h = make_h_and_jac(S, psi, d0)
    
    # Initialize both filters from true state
    x0_init = X_true[0, 0].copy()
    P0 = np.diag([5.0, 5.0, 1.0, 1.0])
    
    # EKF
    ekf = ExtendedKalmanFilter(
        g=g_filter, h=h, jac_g=jac_g, jac_h=jac_h,
        Q=Q_FILTER, R=R, joseph=True, jitter=1e-8
    )
    ekf_state = EKFState(mean=x0_init.copy(), cov=P0.copy(), t=0)
    
    # UKF
    ukf = UnscentedKalmanFilter(
        g=g_filter, h=h, Q=Q_FILTER, R=R,
        alpha=1e-3, beta=2.0, kappa=0.0, jitter=1e-9
    )
    ukf_state = UKFState(mean=x0_init.copy(), cov=P0.copy(), t=0)
    
    ekf_errors = []
    ukf_errors = []
    
    for t in range(1, cfg.n_steps):
        z_t = Z[t]
        
        ekf_state = ekf.step(ekf_state, z_t, u=None)
        ukf_state = ukf.step(ukf_state, z_t, u=None)
        
        ekf_err = np.linalg.norm(ekf_state.mean[:2] - X_true[t, 0, :2])
        ukf_err = np.linalg.norm(ukf_state.mean[:2] - X_true[t, 0, :2])
        
        ekf_errors.append(ekf_err)
        ukf_errors.append(ukf_err)
    
    # Both should produce reasonable results
    mean_ekf_error = np.mean(ekf_errors)
    mean_ukf_error = np.mean(ukf_errors)
    
    assert mean_ekf_error < 10.0, f"EKF mean error {mean_ekf_error} should be reasonable"
    assert mean_ukf_error < 10.0, f"UKF mean error {mean_ukf_error} should be reasonable"
    
    # Both should produce finite results throughout
    assert np.all(np.isfinite(ekf_errors))
    assert np.all(np.isfinite(ukf_errors))


@pytest.mark.integration
def test_all_filters_produce_finite_results_mat():
    """Test that all four filter types (EKF, UKF, EDH, LEDH) produce finite results on MAT data."""
    cfg = ScenarioConfig(
        n_targets=2,
        n_steps=15,
        sensor_grid_shape=(3, 3),
        seed=999,
        use_article_init=False,
    )
    dyn = DynamicsConfig(dt=1.0)
    data = simulate_acoustic_dataset(cfg, dyn)
    
    X_true, S, Z, meta = data["X"], data["S"], data["Z"], data["meta"]
    psi, d0 = meta[2], meta[3]
    n_sensors = S.shape[0]
    R = (0.1 ** 2) * np.eye(n_sensors)  # Use fixed noise std of 0.1
    
    h, jac_h = make_h_and_jac(S, psi, d0)
    log_like_pdf = make_log_like_pdf(h, R)
    
    # Initialize all filters for first target
    x0 = X_true[0, 0].copy()
    P0 = np.diag([10.0, 10.0, 2.0, 2.0])
    
    # EKF
    ekf = ExtendedKalmanFilter(g=g_filter, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q_FILTER, R=R)
    ekf_state = EKFState(mean=x0.copy(), cov=P0.copy(), t=0)
    
    # UKF
    ukf = UnscentedKalmanFilter(g=g_filter, h=h, Q=Q_FILTER, R=R, alpha=1e-3, beta=2.0, kappa=0.0)
    ukf_state = UKFState(mean=x0.copy(), cov=P0.copy(), t=0)
    
    # EDH
    ekf_tracker = EKFTracker(ekf, ekf_state)
    edh_cfg = EDHConfig(n_particles=50, n_lambda_steps=3, resample_ess_ratio=0.5, 
                        flow_integrator="rk4", rng=np.random.default_rng(500))
    edh = EDHFlowPF(tracker=ekf_tracker, g=g_filter, h=h, jacobian_h=jac_h,
                    log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=edh_cfg)
    edh_state = edh.init_from_gaussian(x0.copy(), P0.copy())
    
    # LEDH
    ekf_tracker2 = EKFTracker(ekf, EKFState(mean=x0.copy(), cov=P0.copy(), t=0))
    ledh_cfg = LEDHConfig(n_particles=50, n_lambda_steps=3, resample_ess_ratio=0.5,
                          rng=np.random.default_rng(600))
    ledh = LEDHFlowPF(tracker=ekf_tracker2, g=g_filter, h=h, jacobian_h=jac_h,
                      log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=ledh_cfg)
    ledh_state = ledh.init_from_gaussian(x0.copy(), P0.copy())
    
    # Run all filters for a few steps
    for t in range(1, min(10, cfg.n_steps)):
        z_t = Z[t]
        
        # EKF
        ekf_state = ekf.step(ekf_state, z_t, u=None)
        assert np.isfinite(ekf_state.mean).all()
        assert np.isfinite(ekf_state.cov).all()
        
        # UKF
        ukf_state = ukf.step(ukf_state, z_t, u=None)
        assert np.isfinite(ukf_state.mean).all()
        assert np.isfinite(ukf_state.cov).all()
        
        # EDH
        def proc_noise_sampler(N, nx):
            return edh_cfg.rng.multivariate_normal(np.zeros(nx), Q_FILTER, size=N)
        edh_state = edh.step(edh_state, z_t, process_noise_sampler=proc_noise_sampler)
        assert np.isfinite(edh_state.particles).all()
        assert np.isfinite(edh_state.weights).all()
        assert np.isfinite(edh_state.mean).all()
        
        # LEDH
        def proc_noise_sampler2(N, nx):
            return ledh_cfg.rng.multivariate_normal(np.zeros(nx), Q_FILTER, size=N)
        ledh_state = ledh.step(ledh_state, z_t, process_noise_sampler=proc_noise_sampler2)
        assert np.isfinite(ledh_state.particles).all()
        assert np.isfinite(ledh_state.weights).all()
        assert np.isfinite(ledh_state.mean).all()
