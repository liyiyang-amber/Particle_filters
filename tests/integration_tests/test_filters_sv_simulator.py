"""
Integration tests for all filter types with Stochastic Volatility (SV) simulator.

Tests the integration of EKF, UKF, EDH, and LEDH filters with the SV simulator.
Consolidates tests previously split across multiple files.
"""

import numpy as np
import pytest

from simulator.simulator_sto_volatility_model import simulate_sv_1d
from models.extended_kalman_filter import ExtendedKalmanFilter, EKFState
from models.unscented_kalman_filter import UnscentedKalmanFilter, UKFState
from models.EDH_particle_filter import EDHFlowPF, EDHConfig
from models.LEDH_particle_filter import LEDHFlowPF, LEDHConfig


# Standard SV model parameters
ALPHA = 0.9
SIGMA = 0.2
BETA = 1.0
Q_FILTER = np.array([[SIGMA**2]])
R_FILTER = np.array([[0.1]])


def g_filter(x, u=None, v=None):
    """Process model: x_k = alpha * x_{k-1} + v."""
    if v is None:
        v = np.zeros_like(x)
    return np.array([ALPHA * x[0] + v[0]])


def jac_g(x, u=None):
    """Jacobian of process model."""
    return np.array([[ALPHA]])


def make_h_and_jac(beta):
    """Factory for observation model and its Jacobian."""
    def h(x):
        """Observation model: y_k = beta * exp(0.5 * x_k)."""
        return np.array([beta * np.exp(0.5 * x[0])])
    
    def jac_h(x):
        """Jacobian of observation model."""
        return np.array([[0.5 * beta * np.exp(0.5 * x[0])]])
    
    return h, jac_h


def log_trans_pdf(xk, xkm1):
    """Log p(x_k | x_{k-1})."""
    diff = xk - ALPHA * xkm1
    return -0.5 * (diff.T @ np.linalg.solve(Q_FILTER, diff) + np.log(2 * np.pi * Q_FILTER[0, 0]))


def make_log_like_pdf(h, R):
    """Factory for log likelihood PDF."""
    def log_like_pdf(z, x):
        """Log p(z | x)."""
        diff = z - h(x)
        return -0.5 * (diff.T @ np.linalg.solve(R, diff) + np.log(2 * np.pi * R[0, 0]))
    return log_like_pdf


class EKFTracker:
    """Wrapper for EKF to match tracker protocol."""
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
    """Wrapper for UKF to match tracker protocol."""
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


@pytest.fixture
def sv_data():
    """Generate SV simulation data."""
    results = simulate_sv_1d(n=200, alpha=ALPHA, sigma=SIGMA, beta=BETA, seed=42)
    return results


@pytest.mark.integration
def test_ekf_sv_basic_filtering(sv_data):
    """Test basic EKF filtering on SV data."""
    X_true = sv_data.X
    Y_obs = sv_data.Y
    n = len(X_true)
    
    h, jac_h = make_h_and_jac(BETA)
    
    ekf = ExtendedKalmanFilter(g=g_filter, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q_FILTER, R=R_FILTER)
    state = EKFState(mean=np.array([X_true[0]]), cov=np.array([[0.5]]), t=0)
    
    # Run for 50 steps
    for t in range(1, min(50, n)):
        z_t = np.array([Y_obs[t]])
        state = ekf.step(state, z_t, u=None)
    
    # Check final state
    assert np.isfinite(state.mean).all()
    assert np.isfinite(state.cov).all()
    assert state.mean.shape == (1,)
    assert state.cov.shape == (1, 1)


@pytest.mark.integration
def test_ukf_sv_basic_filtering(sv_data):
    """Test basic UKF filtering on SV data."""
    X_true = sv_data.X
    Y_obs = sv_data.Y
    n = len(X_true)
    
    h, _ = make_h_and_jac(BETA)
    
    ukf = UnscentedKalmanFilter(g=g_filter, h=h, Q=Q_FILTER, R=R_FILTER, alpha=1e-3, beta=2.0, kappa=0.0)
    state = UKFState(mean=np.array([X_true[0]]), cov=np.array([[0.5]]), t=0)
    
    # Run for 50 steps
    for t in range(1, min(50, n)):
        z_t = np.array([Y_obs[t]])
        state = ukf.step(state, z_t, u=None)
    
    # Check final state
    assert np.isfinite(state.mean).all()
    assert np.isfinite(state.cov).all()
    assert state.mean.shape == (1,)
    assert state.cov.shape == (1, 1)


@pytest.mark.integration
def test_edh_ekf_sv_basic_filtering(sv_data):
    """Test EDH-PF with EKF tracker on SV data."""
    X_true = sv_data.X
    Y_obs = sv_data.Y
    n = len(X_true)
    
    h, jac_h = make_h_and_jac(BETA)
    log_like_pdf = make_log_like_pdf(h, R_FILTER)
    
    # Initialize EKF tracker
    ekf = ExtendedKalmanFilter(g=g_filter, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q_FILTER, R=R_FILTER)
    ekf_state = EKFState(mean=np.array([X_true[0]]), cov=np.array([[0.5]]), t=0)
    tracker = EKFTracker(ekf, ekf_state)
    
    # Initialize EDH-PF
    cfg = EDHConfig(n_particles=500, n_lambda_steps=4, resample_ess_ratio=0.0, rng=np.random.default_rng(123))
    edh_pf = EDHFlowPF(
        tracker=tracker, g=g_filter, h=h, jacobian_h=jac_h,
        log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R_FILTER, config=cfg
    )
    
    # Initialize particles
    x0 = np.array([X_true[0]])
    P0 = np.array([[0.5]])
    state = edh_pf.init_from_gaussian(x0, P0)
    
    # Run for 50 steps
    for t in range(1, min(50, n)):
        z_t = np.array([Y_obs[t]])
        
        def process_noise_sampler(N, nx):
            return cfg.rng.multivariate_normal(np.zeros(nx), Q_FILTER, size=N)
        
        state = edh_pf.step(state, z_t, process_noise_sampler=process_noise_sampler)
    
    # Check final state
    assert state.particles.shape == (500, 1)
    assert np.isclose(np.sum(state.weights), 1.0)
    assert np.isfinite(state.particles).all()
    assert np.isfinite(state.mean).all()


@pytest.mark.integration
def test_edh_ukf_sv_basic_filtering(sv_data):
    """Test EDH-PF with UKF tracker on SV data."""
    X_true = sv_data.X
    Y_obs = sv_data.Y
    n = len(X_true)
    
    h, jac_h = make_h_and_jac(BETA)
    log_like_pdf = make_log_like_pdf(h, R_FILTER)
    
    # Initialize UKF tracker
    ukf = UnscentedKalmanFilter(g=g_filter, h=h, Q=Q_FILTER, R=R_FILTER, alpha=1e-3, beta=2.0, kappa=0.0)
    ukf_state = UKFState(mean=np.array([X_true[0]]), cov=np.array([[0.5]]), t=0)
    tracker = UKFTracker(ukf, ukf_state)
    
    # Initialize EDH-PF
    cfg = EDHConfig(n_particles=500, n_lambda_steps=4, resample_ess_ratio=0.0, rng=np.random.default_rng(123))
    edh_pf = EDHFlowPF(
        tracker=tracker, g=g_filter, h=h, jacobian_h=jac_h,
        log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R_FILTER, config=cfg
    )
    
    # Initialize particles
    x0 = np.array([X_true[0]])
    P0 = np.array([[0.5]])
    state = edh_pf.init_from_gaussian(x0, P0)
    
    # Run for 50 steps
    for t in range(1, min(50, n)):
        z_t = np.array([Y_obs[t]])
        
        def process_noise_sampler(N, nx):
            return cfg.rng.multivariate_normal(np.zeros(nx), Q_FILTER, size=N)
        
        state = edh_pf.step(state, z_t, process_noise_sampler=process_noise_sampler)
    
    # Check final state
    assert state.particles.shape == (500, 1)
    assert np.isclose(np.sum(state.weights), 1.0)
    assert np.isfinite(state.particles).all()
    assert np.isfinite(state.mean).all()


@pytest.mark.integration
def test_ledh_ekf_sv_basic_filtering(sv_data):
    """Test LEDH-PF with EKF tracker on SV data."""
    X_true = sv_data.X
    Y_obs = sv_data.Y
    n = len(X_true)
    
    h, jac_h = make_h_and_jac(BETA)
    log_like_pdf = make_log_like_pdf(h, R_FILTER)
    
    # Initialize EKF tracker
    ekf = ExtendedKalmanFilter(g=g_filter, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q_FILTER, R=R_FILTER)
    ekf_state = EKFState(mean=np.array([X_true[0]]), cov=np.array([[0.5]]), t=0)
    tracker = EKFTracker(ekf, ekf_state)
    
    # Initialize LEDH-PF
    cfg = LEDHConfig(n_particles=500, n_lambda_steps=4, resample_ess_ratio=0.0, rng=np.random.default_rng(123))
    ledh_pf = LEDHFlowPF(
        tracker=tracker, g=g_filter, h=h, jacobian_h=jac_h,
        log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R_FILTER, config=cfg
    )
    
    # Initialize particles
    x0 = np.array([X_true[0]])
    P0 = np.array([[0.5]])
    state = ledh_pf.init_from_gaussian(x0, P0)
    
    # Run for 50 steps
    for t in range(1, min(50, n)):
        z_t = np.array([Y_obs[t]])
        
        def process_noise_sampler(N, nx):
            return cfg.rng.multivariate_normal(np.zeros(nx), Q_FILTER, size=N)
        
        state = ledh_pf.step(state, z_t, process_noise_sampler=process_noise_sampler)
    
    # Check final state
    assert state.particles.shape == (500, 1)
    assert np.isclose(np.sum(state.weights), 1.0)
    assert np.isfinite(state.particles).all()
    assert np.isfinite(state.mean).all()


@pytest.mark.integration
def test_ledh_ukf_sv_basic_filtering(sv_data):
    """Test LEDH-PF with UKF tracker on SV data."""
    X_true = sv_data.X
    Y_obs = sv_data.Y
    n = len(X_true)
    
    h, jac_h = make_h_and_jac(BETA)
    log_like_pdf = make_log_like_pdf(h, R_FILTER)
    
    # Initialize UKF tracker
    ukf = UnscentedKalmanFilter(g=g_filter, h=h, Q=Q_FILTER, R=R_FILTER, alpha=1e-3, beta=2.0, kappa=0.0)
    ukf_state = UKFState(mean=np.array([X_true[0]]), cov=np.array([[0.5]]), t=0)
    tracker = UKFTracker(ukf, ukf_state)
    
    # Initialize LEDH-PF
    cfg = LEDHConfig(n_particles=500, n_lambda_steps=4, resample_ess_ratio=0.0, rng=np.random.default_rng(123))
    ledh_pf = LEDHFlowPF(
        tracker=tracker, g=g_filter, h=h, jacobian_h=jac_h,
        log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R_FILTER, config=cfg
    )
    
    # Initialize particles
    x0 = np.array([X_true[0]])
    P0 = np.array([[0.5]])
    state = ledh_pf.init_from_gaussian(x0, P0)
    
    # Run for 50 steps
    for t in range(1, min(50, n)):
        z_t = np.array([Y_obs[t]])
        
        def process_noise_sampler(N, nx):
            return cfg.rng.multivariate_normal(np.zeros(nx), Q_FILTER, size=N)
        
        state = ledh_pf.step(state, z_t, process_noise_sampler=process_noise_sampler)
    
    # Check final state
    assert state.particles.shape == (500, 1)
    assert np.isclose(np.sum(state.weights), 1.0)
    assert np.isfinite(state.particles).all()
    assert np.isfinite(state.mean).all()


@pytest.mark.integration
def test_all_filters_comparison_sv():
    """Compare all four filter types on same SV data."""
    results = simulate_sv_1d(n=100, alpha=ALPHA, sigma=SIGMA, beta=BETA, seed=999)
    X_true = results.X
    Y_obs = results.Y
    n = len(X_true)
    
    h, jac_h = make_h_and_jac(BETA)
    log_like_pdf = make_log_like_pdf(h, R_FILTER)
    
    x0 = np.array([X_true[0]])
    P0 = np.array([[0.5]])
    
    # EKF
    ekf = ExtendedKalmanFilter(g=g_filter, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q_FILTER, R=R_FILTER)
    ekf_state = EKFState(mean=x0.copy(), cov=P0.copy(), t=0)
    
    # UKF
    ukf = UnscentedKalmanFilter(g=g_filter, h=h, Q=Q_FILTER, R=R_FILTER)
    ukf_state = UKFState(mean=x0.copy(), cov=P0.copy(), t=0)
    
    # EDH
    ekf_for_edh = ExtendedKalmanFilter(g=g_filter, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q_FILTER, R=R_FILTER)
    edh_tracker = EKFTracker(ekf_for_edh, EKFState(mean=x0.copy(), cov=P0.copy(), t=0))
    edh_cfg = EDHConfig(n_particles=200, n_lambda_steps=4, resample_ess_ratio=0.5, rng=np.random.default_rng(500))
    edh = EDHFlowPF(tracker=edh_tracker, g=g_filter, h=h, jacobian_h=jac_h,
                    log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R_FILTER, config=edh_cfg)
    edh_state = edh.init_from_gaussian(x0.copy(), P0.copy())
    
    # LEDH
    ekf_for_ledh = ExtendedKalmanFilter(g=g_filter, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q_FILTER, R=R_FILTER)
    ledh_tracker = EKFTracker(ekf_for_ledh, EKFState(mean=x0.copy(), cov=P0.copy(), t=0))
    ledh_cfg = LEDHConfig(n_particles=200, n_lambda_steps=4, resample_ess_ratio=0.5, rng=np.random.default_rng(600))
    ledh = LEDHFlowPF(tracker=ledh_tracker, g=g_filter, h=h, jacobian_h=jac_h,
                      log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R_FILTER, config=ledh_cfg)
    ledh_state = ledh.init_from_gaussian(x0.copy(), P0.copy())
    
    # Run all filters
    for t in range(1, min(50, n)):
        z_t = np.array([Y_obs[t]])
        
        # EKF
        ekf_state = ekf.step(ekf_state, z_t, u=None)
        assert np.isfinite(ekf_state.mean).all()
        
        # UKF
        ukf_state = ukf.step(ukf_state, z_t, u=None)
        assert np.isfinite(ukf_state.mean).all()
        
        # EDH
        def proc_noise_edh(N, nx):
            return edh_cfg.rng.multivariate_normal(np.zeros(nx), Q_FILTER, size=N)
        edh_state = edh.step(edh_state, z_t, process_noise_sampler=proc_noise_edh)
        assert np.isfinite(edh_state.mean).all()
        
        # LEDH
        def proc_noise_ledh(N, nx):
            return ledh_cfg.rng.multivariate_normal(np.zeros(nx), Q_FILTER, size=N)
        ledh_state = ledh.step(ledh_state, z_t, process_noise_sampler=proc_noise_ledh)
        assert np.isfinite(ledh_state.mean).all()
