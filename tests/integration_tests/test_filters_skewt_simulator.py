"""
Consolidated integration tests for skew-t simulator covering EKF/UKF trackers
and EDH/LEDH particle filters.
"""

import numpy as np
import pytest

from simulator.simulator_sensor_network_skewt_dynamic import (
    GridConfig, DynConfig, MeasConfig, SimConfig, simulate_trial
)
from models.extended_kalman_filter import ExtendedKalmanFilter, EKFState
from models.unscented_kalman_filter import UnscentedKalmanFilter, UKFState
from models.EDH_particle_filter import EDHFlowPF, EDHConfig
from models.LEDH_particle_filter import LEDHFlowPF, LEDHConfig


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


class UKFTracker:
    def __init__(self, ukf, initial_state):
        self.ukf = ukf
        self.state = initial_state
        self.past_mean = initial_state.mean.copy()

    def predict(self):
        self.past_mean = self.state.mean.copy()
        self.state = self.ukf.predict(self.state, u=None)
        return self.state.mean, self.state.cov

    def update(self, z_k):
        self.state = self.ukf.update(self.state, z_k)
        return self.state.mean, self.state.cov

    def get_past_mean(self):
        return self.past_mean


def make_skewt_models(dyn_cfg, data, d):
    def g_ekf(x, u):
        return dyn_cfg.alpha * x

    def g_edh(x, u, v):
        if v is None:
            v = np.zeros_like(x)
        return dyn_cfg.alpha * x + v

    def h(x):
        return x

    def jac_h(x):
        return np.eye(d)

    Q = data["Sigma"]
    R = np.eye(d) * 10.0

    def log_trans_pdf(xk, xkm1):
        diff = xk - dyn_cfg.alpha * xkm1
        return -0.5 * (diff.T @ np.linalg.solve(Q, diff) + np.log((2 * np.pi) ** d * np.linalg.det(Q)))

    def log_like_pdf(z, x):
        diff = z - h(x)
        return -0.5 * (diff.T @ np.linalg.solve(R, diff) + np.log((2 * np.pi) ** d * np.linalg.det(R)))

    return g_ekf, g_edh, h, jac_h, Q, R, log_trans_pdf, log_like_pdf


@pytest.mark.integration
def test_edh_ukf_skewt_basic():
    grid_cfg = GridConfig(d=4, alpha0=2.0, alpha1=0.1, beta=8.0)
    dyn_cfg = DynConfig(alpha=0.9, nu=8.0, gamma_scale=0.1, seed=42)
    meas_cfg = MeasConfig(m1=1.0, m2=0.3)
    sim_cfg = SimConfig(T=20, save_lambda=True)

    data = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    Z = data["Z"]
    d = grid_cfg.d

    g_ekf, g_edh, h, jac_h, Q, R, log_trans_pdf, log_like_pdf = make_skewt_models(dyn_cfg, data, d)

    ukf = UnscentedKalmanFilter(g=g_ekf, h=h, Q=Q, R=R, alpha=0.001, beta=2.0, kappa=0.0)
    ukf_state = UKFState(mean=np.zeros(d), cov=Q.copy(), t=0)
    tracker = UKFTracker(ukf, ukf_state)

    cfg = EDHConfig(n_particles=200, n_lambda_steps=4, resample_ess_ratio=0.0, rng=np.random.default_rng(123))
    edh_pf = EDHFlowPF(tracker=tracker, g=g_edh, h=h, jacobian_h=jac_h,
                       log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=cfg)

    state = edh_pf.init_from_gaussian(np.zeros(d), Q.copy())

    for t in range(1, min(15, len(Z))):
        z_t = Z[t].astype(float)
        def proc_noise(N, nx):
            return cfg.rng.multivariate_normal(np.zeros(nx), Q, size=N)
        state = edh_pf.step(state, z_t, process_noise_sampler=proc_noise)

    assert np.isfinite(state.mean).all()
    assert np.isclose(np.sum(state.weights), 1.0)


@pytest.mark.integration
def test_ledh_ukf_skewt_basic():
    grid_cfg = GridConfig(d=4, alpha0=2.0, alpha1=0.1, beta=8.0)
    dyn_cfg = DynConfig(alpha=0.9, nu=8.0, gamma_scale=0.1, seed=42)
    meas_cfg = MeasConfig(m1=1.0, m2=0.3)
    sim_cfg = SimConfig(T=20, save_lambda=True)

    data = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    Z = data["Z"]
    d = grid_cfg.d

    g_ekf, g_edh, h, jac_h, Q, R, log_trans_pdf, log_like_pdf = make_skewt_models(dyn_cfg, data, d)

    ukf = UnscentedKalmanFilter(g=g_ekf, h=h, Q=Q, R=R, alpha=0.001, beta=2.0, kappa=0.0)
    ukf_state = UKFState(mean=np.zeros(d), cov=Q.copy(), t=0)
    tracker = UKFTracker(ukf, ukf_state)

    cfg = LEDHConfig(n_particles=200, n_lambda_steps=4, resample_ess_ratio=0.0, rng=np.random.default_rng(123))
    ledh_pf = LEDHFlowPF(tracker=tracker, g=g_edh, h=h, jacobian_h=jac_h,
                         log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=cfg)

    state = ledh_pf.init_from_gaussian(np.zeros(d), Q.copy())

    for t in range(1, min(15, len(Z))):
        z_t = Z[t].astype(float)
        def proc_noise(N, nx):
            return cfg.rng.multivariate_normal(np.zeros(nx), Q, size=N)
        state = ledh_pf.step(state, z_t, process_noise_sampler=proc_noise)

    assert np.isfinite(state.mean).all()
    assert np.isclose(np.sum(state.weights), 1.0)


@pytest.mark.integration
def test_edh_ekf_skewt_basic():
    grid_cfg = GridConfig(d=4, alpha0=2.0, alpha1=0.1, beta=8.0)
    dyn_cfg = DynConfig(alpha=0.9, nu=8.0, gamma_scale=0.1, seed=42)
    meas_cfg = MeasConfig(m1=1.0, m2=0.3)
    sim_cfg = SimConfig(T=20, save_lambda=True)

    data = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    Z = data["Z"]
    d = grid_cfg.d

    g_ekf, g_edh, h, jac_h, Q, R, log_trans_pdf, log_like_pdf = make_skewt_models(dyn_cfg, data, d)

    ekf = ExtendedKalmanFilter(g=g_ekf, h=h, jac_g=lambda x, u: dyn_cfg.alpha * np.eye(d), jac_h=jac_h, Q=Q, R=R)
    ekf_state = EKFState(mean=np.zeros(d), cov=Q.copy(), t=0)
    tracker = EKFTracker(ekf, ekf_state)

    cfg = EDHConfig(n_particles=200, n_lambda_steps=4, resample_ess_ratio=0.0, rng=np.random.default_rng(321))
    edh_pf = EDHFlowPF(tracker=tracker, g=g_edh, h=h, jacobian_h=jac_h,
                       log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=cfg)

    state = edh_pf.init_from_gaussian(np.zeros(d), Q.copy())

    for t in range(1, min(15, len(Z))):
        z_t = Z[t].astype(float)
        def proc_noise(N, nx):
            return cfg.rng.multivariate_normal(np.zeros(nx), Q, size=N)
        state = edh_pf.step(state, z_t, process_noise_sampler=proc_noise)

    assert np.isfinite(state.mean).all()
    assert np.isclose(np.sum(state.weights), 1.0)


@pytest.mark.integration
def test_ledh_ekf_skewt_basic():
    grid_cfg = GridConfig(d=4, alpha0=2.0, alpha1=0.1, beta=8.0)
    dyn_cfg = DynConfig(alpha=0.9, nu=8.0, gamma_scale=0.1, seed=42)
    meas_cfg = MeasConfig(m1=1.0, m2=0.3)
    sim_cfg = SimConfig(T=20, save_lambda=True)

    data = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    Z = data["Z"]
    d = grid_cfg.d

    g_ekf, g_edh, h, jac_h, Q, R, log_trans_pdf, log_like_pdf = make_skewt_models(dyn_cfg, data, d)

    ekf = ExtendedKalmanFilter(g=g_ekf, h=h, jac_g=lambda x, u: dyn_cfg.alpha * np.eye(d), jac_h=jac_h, Q=Q, R=R)
    ekf_state = EKFState(mean=np.zeros(d), cov=Q.copy(), t=0)
    tracker = EKFTracker(ekf, ekf_state)

    cfg = LEDHConfig(n_particles=200, n_lambda_steps=4, resample_ess_ratio=0.0, rng=np.random.default_rng(654))
    ledh_pf = LEDHFlowPF(tracker=tracker, g=g_edh, h=h, jacobian_h=jac_h,
                         log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=cfg)

    state = ledh_pf.init_from_gaussian(np.zeros(d), Q.copy())

    for t in range(1, min(15, len(Z))):
        z_t = Z[t].astype(float)
        def proc_noise(N, nx):
            return cfg.rng.multivariate_normal(np.zeros(nx), Q, size=N)
        state = ledh_pf.step(state, z_t, process_noise_sampler=proc_noise)

    assert np.isfinite(state.mean).all()
    assert np.isclose(np.sum(state.weights), 1.0)


@pytest.mark.integration
def test_all_filters_comparison_skewt():
    # Smaller trial for comparison
    grid_cfg = GridConfig(d=4, alpha0=2.0, alpha1=0.1, beta=8.0)
    dyn_cfg = DynConfig(alpha=0.9, nu=8.0, gamma_scale=0.1, seed=11)
    meas_cfg = MeasConfig(m1=1.0, m2=0.3)
    sim_cfg = SimConfig(T=25, save_lambda=True)

    data = simulate_trial(grid_cfg, dyn_cfg, meas_cfg, sim_cfg)
    Z = data["Z"]
    d = grid_cfg.d

    g_ekf, g_edh, h, jac_h, Q, R, log_trans_pdf, log_like_pdf = make_skewt_models(dyn_cfg, data, d)

    # EKF
    ekf = ExtendedKalmanFilter(g=g_ekf, h=h, jac_g=lambda x, u: dyn_cfg.alpha * np.eye(d), jac_h=jac_h, Q=Q, R=R)
    ekf_state = EKFState(mean=np.zeros(d), cov=Q.copy(), t=0)

    # UKF
    ukf = UnscentedKalmanFilter(g=g_ekf, h=h, Q=Q, R=R)
    ukf_state = UKFState(mean=np.zeros(d), cov=Q.copy(), t=0)

    # EDH with UKF tracker
    ukf_for_edh = UnscentedKalmanFilter(g=g_ekf, h=h, Q=Q, R=R)
    edh_tracker = UKFTracker(ukf_for_edh, ukf_state)
    edh_cfg = EDHConfig(n_particles=100, n_lambda_steps=4, resample_ess_ratio=0.5, rng=np.random.default_rng(101))
    edh = EDHFlowPF(tracker=edh_tracker, g=g_edh, h=h, jacobian_h=jac_h,
                    log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=edh_cfg)
    edh_state = edh.init_from_gaussian(np.zeros(d), Q.copy())

    # LEDH with EKF tracker
    ekf_for_ledh = ExtendedKalmanFilter(g=g_ekf, h=h, jac_g=lambda x,u: dyn_cfg.alpha * np.eye(d), jac_h=jac_h, Q=Q, R=R)
    ledh_tracker = EKFTracker(ekf_for_ledh, ekf_state)
    ledh_cfg = LEDHConfig(n_particles=100, n_lambda_steps=4, resample_ess_ratio=0.5, rng=np.random.default_rng(202))
    ledh = LEDHFlowPF(tracker=ledh_tracker, g=g_edh, h=h, jacobian_h=jac_h,
                      log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=ledh_cfg)
    ledh_state = ledh.init_from_gaussian(np.zeros(d), Q.copy())

    # Run a short comparison
    for t in range(1, min(12, len(Z))):
        z_t = Z[t].astype(float)

        # EKF update
        ekf_state = ekf.step(ekf_state, z_t, u=None)
        assert np.isfinite(ekf_state.mean).all()

        # UKF update
        ukf_state = ukf.step(ukf_state, z_t, u=None)
        assert np.isfinite(ukf_state.mean).all()

        # EDH step
        def proc_edh(N, nx):
            return edh_cfg.rng.multivariate_normal(np.zeros(nx), Q, size=N)
        edh_state = edh.step(edh_state, z_t, process_noise_sampler=proc_edh)
        assert np.isfinite(edh_state.mean).all()

        # LEDH step
        def proc_ledh(N, nx):
            return ledh_cfg.rng.multivariate_normal(np.zeros(nx), Q, size=N)
        ledh_state = ledh.step(ledh_state, z_t, process_noise_sampler=proc_ledh)
        assert np.isfinite(ledh_state.mean).all()
