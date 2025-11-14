"""
Integration tests for all filter types with Sensor Network Linear Gaussian (SNLG) simulator.

Tests the integration of KF, EKF, UKF, EDH, and LEDH filters with the SNLG simulator.
Consolidates tests previously split across multiple files.
"""

import numpy as np
import pytest

from simulator.simulator_sensor_network_linear_gaussian import SimConfig, simulate_dataset
from models.extended_kalman_filter import ExtendedKalmanFilter, EKFState
from models.unscented_kalman_filter import UnscentedKalmanFilter, UKFState
from models.EDH_particle_filter import EDHFlowPF, EDHConfig
from models.LEDH_particle_filter import LEDHFlowPF, LEDHConfig


# Standard SNLG config for testing
def get_snlg_config():
    """Get standard SNLG simulation config."""
    return SimConfig(
        d=9,  # 3x3 grid
        alpha=0.9,
        alpha0=1.5,
        alpha1=0.05,
        beta=8.0,
        T=40,
        trials=1,
        sigmas=(1.0,),
        seed=42,
    )


def make_snlg_models(alpha, Q, R, d):
    """Factory for SNLG model functions."""
    def g_filter(x, u=None, v=None):
        """Process model: x_k = alpha * x_{k-1} + v."""
        if v is None:
            v = np.zeros_like(x)
        return alpha * x + v
    
    def h(x):
        """Measurement model: y_k = x_k."""
        return x
    
    def jac_g(x, u=None):
        """Jacobian of process model."""
        return alpha * np.eye(d)
    
    def jac_h(x):
        """Jacobian of measurement model."""
        return np.eye(d)
    
    def log_trans_pdf(xk, xkm1):
        """Log p(x_k | x_{k-1})."""
        diff = xk - alpha * xkm1
        sign, logdet = np.linalg.slogdet(Q)
        return -0.5 * (diff.T @ np.linalg.solve(Q, diff) + logdet + d * np.log(2 * np.pi))
    
    def log_like_pdf(z, x):
        """Log p(z | x)."""
        diff = z - h(x)
        sign, logdet = np.linalg.slogdet(R)
        return -0.5 * (diff.T @ np.linalg.solve(R, diff) + logdet + d * np.log(2 * np.pi))
    
    return g_filter, h, jac_g, jac_h, log_trans_pdf, log_like_pdf


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
def snlg_data():
    """Generate SNLG simulation data."""
    cfg = get_snlg_config()
    X, Z, coords, Sigma = simulate_dataset(cfg)
    return {
        'X_true': X[0, 0, :, :],
        'Y': Z[0, 0, :, :],
        'Sigma': Sigma,
        'config': cfg
    }


@pytest.mark.integration
def test_ekf_snlg_basic_filtering(snlg_data):
    """Test basic EKF filtering on SNLG data."""
    X_true = snlg_data['X_true']
    Y = snlg_data['Y']
    Sigma = snlg_data['Sigma']
    cfg = snlg_data['config']
    
    d = cfg.d
    alpha = cfg.alpha
    sigma_z = cfg.sigmas[0]
    Q = Sigma
    R = (sigma_z ** 2) * np.eye(d)
    
    g_filter, h, jac_g, jac_h, _, _ = make_snlg_models(alpha, Q, R, d)
    
    ekf = ExtendedKalmanFilter(g=g_filter, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q, R=R)
    state = EKFState(mean=np.zeros(d), cov=Sigma.copy(), t=0)
    
    # Run for all steps
    for t in range(len(Y)):
        state = ekf.step(state, Y[t], u=None)
    
    # Check final state
    assert np.isfinite(state.mean).all()
    assert np.isfinite(state.cov).all()
    assert state.mean.shape == (d,)


@pytest.mark.integration
def test_ukf_snlg_basic_filtering(snlg_data):
    """Test basic UKF filtering on SNLG data."""
    X_true = snlg_data['X_true']
    Y = snlg_data['Y']
    Sigma = snlg_data['Sigma']
    cfg = snlg_data['config']
    
    d = cfg.d
    alpha = cfg.alpha
    sigma_z = cfg.sigmas[0]
    Q = Sigma
    R = (sigma_z ** 2) * np.eye(d)
    
    g_filter, h, _, _, _, _ = make_snlg_models(alpha, Q, R, d)
    
    ukf = UnscentedKalmanFilter(g=g_filter, h=h, Q=Q, R=R, alpha=1e-3, beta=2.0, kappa=0.0)
    state = UKFState(mean=np.zeros(d), cov=Sigma.copy(), t=0)
    
    # Run for all steps
    for t in range(len(Y)):
        state = ukf.step(state, Y[t], u=None)
    
    # Check final state
    assert np.isfinite(state.mean).all()
    assert np.isfinite(state.cov).all()
    assert state.mean.shape == (d,)


@pytest.mark.integration
def test_edh_ukf_snlg_basic_filtering(snlg_data):
    """Test EDH-PF with UKF tracker on SNLG data."""
    X_true = snlg_data['X_true']
    Y = snlg_data['Y']
    Sigma = snlg_data['Sigma']
    cfg = snlg_data['config']
    
    d = cfg.d
    alpha = cfg.alpha
    sigma_z = cfg.sigmas[0]
    Q = Sigma
    R = (sigma_z ** 2) * np.eye(d)
    
    g_filter, h, jac_g, jac_h, log_trans_pdf, log_like_pdf = make_snlg_models(alpha, Q, R, d)
    
    # Initialize UKF tracker
    ukf = UnscentedKalmanFilter(g=g_filter, h=h, Q=Q, R=R, alpha=1e-3, beta=2.0, kappa=0.0)
    ukf_state = UKFState(mean=np.zeros(d), cov=Sigma.copy(), t=0)
    tracker = UKFTracker(ukf, ukf_state)
    
    # Initialize EDH-PF
    edh_cfg = EDHConfig(n_particles=200, n_lambda_steps=6, resample_ess_ratio=0.5, rng=np.random.default_rng(42))
    edh_pf = EDHFlowPF(
        tracker=tracker, g=g_filter, h=h, jacobian_h=jac_h,
        log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=edh_cfg
    )
    
    state = edh_pf.init_from_gaussian(np.zeros(d), Sigma.copy())
    
    # Run for all steps
    for t in range(len(Y)):
        def process_noise_sampler(N, nx):
            return edh_cfg.rng.multivariate_normal(np.zeros(nx), Q, size=N)
        
        state = edh_pf.step(state, Y[t], process_noise_sampler=process_noise_sampler)
    
    # Check final state
    assert np.isclose(np.sum(state.weights), 1.0)
    assert np.isfinite(state.particles).all()
    assert np.isfinite(state.mean).all()


@pytest.mark.integration
def test_ledh_ukf_snlg_basic_filtering(snlg_data):
    """Test LEDH-PF with UKF tracker on SNLG data."""
    X_true = snlg_data['X_true']
    Y = snlg_data['Y']
    Sigma = snlg_data['Sigma']
    cfg = snlg_data['config']
    
    d = cfg.d
    alpha = cfg.alpha
    sigma_z = cfg.sigmas[0]
    Q = Sigma
    R = (sigma_z ** 2) * np.eye(d)
    
    g_filter, h, jac_g, jac_h, log_trans_pdf, log_like_pdf = make_snlg_models(alpha, Q, R, d)
    
    # Initialize UKF tracker
    ukf = UnscentedKalmanFilter(g=g_filter, h=h, Q=Q, R=R, alpha=1e-3, beta=2.0, kappa=0.0)
    ukf_state = UKFState(mean=np.zeros(d), cov=Sigma.copy(), t=0)
    tracker = UKFTracker(ukf, ukf_state)
    
    # Initialize LEDH-PF
    ledh_cfg = LEDHConfig(n_particles=200, n_lambda_steps=6, resample_ess_ratio=0.5, rng=np.random.default_rng(42))
    ledh_pf = LEDHFlowPF(
        tracker=tracker, g=g_filter, h=h, jacobian_h=jac_h,
        log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=ledh_cfg
    )
    
    state = ledh_pf.init_from_gaussian(np.zeros(d), Sigma.copy())
    
    # Run for all steps
    for t in range(len(Y)):
        def process_noise_sampler(N, nx):
            return ledh_cfg.rng.multivariate_normal(np.zeros(nx), Q, size=N)
        
        state = ledh_pf.step(state, Y[t], process_noise_sampler=process_noise_sampler)
    
    # Check final state
    assert np.isclose(np.sum(state.weights), 1.0)
    assert np.isfinite(state.particles).all()
    assert np.isfinite(state.mean).all()


@pytest.mark.integration
def test_all_filters_comparison_snlg():
    """Compare all filter types on same SNLG data."""
    cfg = get_snlg_config()
    X, Z, coords, Sigma = simulate_dataset(cfg)
    X_true = X[0, 0, :, :]
    Y = Z[0, 0, :, :]
    
    d = cfg.d
    alpha = cfg.alpha
    sigma_z = cfg.sigmas[0]
    Q = Sigma
    R = (sigma_z ** 2) * np.eye(d)
    
    g_filter, h, jac_g, jac_h, log_trans_pdf, log_like_pdf = make_snlg_models(alpha, Q, R, d)
    
    # Initialize all filters
    x0 = np.zeros(d)
    P0 = Sigma.copy()
    
    # EKF
    ekf = ExtendedKalmanFilter(g=g_filter, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q, R=R)
    ekf_state = EKFState(mean=x0.copy(), cov=P0.copy(), t=0)
    
    # UKF
    ukf = UnscentedKalmanFilter(g=g_filter, h=h, Q=Q, R=R)
    ukf_state = UKFState(mean=x0.copy(), cov=P0.copy(), t=0)
    
    # EDH
    ukf_for_edh = UnscentedKalmanFilter(g=g_filter, h=h, Q=Q, R=R)
    edh_tracker = UKFTracker(ukf_for_edh, UKFState(mean=x0.copy(), cov=P0.copy(), t=0))
    edh_cfg = EDHConfig(n_particles=100, n_lambda_steps=4, resample_ess_ratio=0.5, rng=np.random.default_rng(500))
    edh = EDHFlowPF(tracker=edh_tracker, g=g_filter, h=h, jacobian_h=jac_h,
                    log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=edh_cfg)
    edh_state = edh.init_from_gaussian(x0.copy(), P0.copy())
    
    # LEDH
    ukf_for_ledh = UnscentedKalmanFilter(g=g_filter, h=h, Q=Q, R=R)
    ledh_tracker = UKFTracker(ukf_for_ledh, UKFState(mean=x0.copy(), cov=P0.copy(), t=0))
    ledh_cfg = LEDHConfig(n_particles=100, n_lambda_steps=4, resample_ess_ratio=0.5, rng=np.random.default_rng(600))
    ledh = LEDHFlowPF(tracker=ledh_tracker, g=g_filter, h=h, jacobian_h=jac_h,
                      log_trans_pdf=log_trans_pdf, log_like_pdf=log_like_pdf, R=R, config=ledh_cfg)
    ledh_state = ledh.init_from_gaussian(x0.copy(), P0.copy())
    
    # Run all filters for first 20 steps
    for t in range(min(20, len(Y))):
        # EKF
        ekf_state = ekf.step(ekf_state, Y[t], u=None)
        assert np.isfinite(ekf_state.mean).all()
        
        # UKF
        ukf_state = ukf.step(ukf_state, Y[t], u=None)
        assert np.isfinite(ukf_state.mean).all()
        
        # EDH
        def proc_noise_edh(N, nx):
            return edh_cfg.rng.multivariate_normal(np.zeros(nx), Q, size=N)
        edh_state = edh.step(edh_state, Y[t], process_noise_sampler=proc_noise_edh)
        assert np.isfinite(edh_state.mean).all()
        
        # LEDH
        def proc_noise_ledh(N, nx):
            return ledh_cfg.rng.multivariate_normal(np.zeros(nx), Q, size=N)
        ledh_state = ledh.step(ledh_state, Y[t], process_noise_sampler=proc_noise_ledh)
        assert np.isfinite(ledh_state.mean).all()
