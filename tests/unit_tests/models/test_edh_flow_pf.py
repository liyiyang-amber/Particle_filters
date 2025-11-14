"""Unit tests for EDHFlowPF (EDH particle-flow particle filter)."""

import numpy as np
import pytest
from models.EDH_particle_filter import EDHFlowPF, EKFTracker, UKFTracker, EDHConfig, PFState
from models.extended_kalman_filter import ExtendedKalmanFilter, EKFState
from models.unscented_kalman_filter import UnscentedKalmanFilter, UKFState


@pytest.fixture
def simple_linear_system():
    """Simple 1D linear system for testing."""
    alpha = 0.9
    sigma = 0.1
    Q = np.array([[sigma**2]])
    R = np.array([[0.1]])
    
    # For EKF/UKF: g(x, u) without v
    def g_ekf(x, u):
        """Process model for EKF/UKF: x_k = alpha * x_{k-1}."""
        return np.array([alpha * x[0]])
    
    # For EDHFlowPF: g(x, u, v) with v
    def g_edh(x, u, v):
        """Process model for EDH: x_k = alpha * x_{k-1} + v."""
        if v is None:
            v = np.zeros_like(x)
        return np.array([alpha * x[0] + v[0]])
    
    def h(x):
        """Observation model: y_k = x_k."""
        return np.array([x[0]])
    
    def jac_g(x, u):
        return np.array([[alpha]])
    
    def jac_h(x):
        return np.array([[1.0]])
    
    def log_trans_pdf(xk, xkm1):
        """Log p(x_k | x_{k-1}) for Gaussian process noise."""
        diff = xk - g_ekf(xkm1, None)
        return -0.5 * (diff.T @ np.linalg.solve(Q, diff) + np.log(2 * np.pi * Q[0, 0]))
    
    def log_like_pdf(z, x):
        """Log p(z | x) for Gaussian observation noise."""
        diff = z - h(x)
        return -0.5 * (diff.T @ np.linalg.solve(R, diff) + np.log(2 * np.pi * R[0, 0]))
    
    return {
        'g_ekf': g_ekf,  # For EKF/UKF
        'g_edh': g_edh,  # For EDHFlowPF
        'h': h,
        'jac_g': jac_g,
        'jac_h': jac_h,
        'Q': Q,
        'R': R,
        'alpha': alpha,
        'log_trans_pdf': log_trans_pdf,
        'log_like_pdf': log_like_pdf
    }


@pytest.fixture
def ekf_tracker_linear(simple_linear_system):
    """Create EKF tracker for linear system."""
    sys = simple_linear_system
    ekf = ExtendedKalmanFilter(
        g=sys['g_ekf'], h=sys['h'],
        jac_g=sys['jac_g'], jac_h=sys['jac_h'],
        Q=sys['Q'], R=sys['R']
    )
    initial_state = EKFState(mean=np.array([0.5]), cov=np.array([[0.3]]), t=0)
    return EKFTracker(ekf, initial_state)


@pytest.fixture
def ukf_tracker_linear(simple_linear_system):
    """Create UKF tracker for linear system."""
    sys = simple_linear_system
    ukf = UnscentedKalmanFilter(
        g=sys['g_ekf'], h=sys['h'],
        Q=sys['Q'], R=sys['R']
    )
    initial_state = UKFState(mean=np.array([0.5]), cov=np.array([[0.3]]), t=0)
    return UKFTracker(ukf, initial_state)


class TestEDHFlowPFInitialization:
    """Tests for EDHFlowPF initialization."""
    
    def test_edh_pf_creation(self, ekf_tracker_linear, simple_linear_system):
        """Test basic EDHFlowPF creation."""
        sys = simple_linear_system
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'],
            h=sys['h'],
            jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R']
        )
        
        assert pf.tracker is ekf_tracker_linear
        assert pf.g is sys['g_edh']
        assert pf.h is sys['h']
        np.testing.assert_array_equal(pf.R, sys['R'])
    
    def test_edh_pf_with_config(self, ekf_tracker_linear, simple_linear_system):
        """Test EDHFlowPF with custom configuration."""
        sys = simple_linear_system
        config = EDHConfig(
            n_particles=100,
            n_lambda_steps=10,
            resample_ess_ratio=0.5,
            flow_integrator="euler",
            rng=np.random.default_rng(42)
        )
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'],
            h=sys['h'],
            jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        assert pf.cfg.n_particles == 100
        assert pf.cfg.n_lambda_steps == 10
        assert pf.cfg.resample_ess_ratio == 0.5
        assert pf.cfg.flow_integrator == "euler"
    
    def test_edh_pf_with_ukf_tracker(self, ukf_tracker_linear, simple_linear_system):
        """Test EDHFlowPF with UKF tracker."""
        sys = simple_linear_system
        
        pf = EDHFlowPF(
            tracker=ukf_tracker_linear,
            g=sys['g_edh'],
            h=sys['h'],
            jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R']
        )
        
        assert pf.tracker is ukf_tracker_linear
        assert isinstance(pf.tracker, UKFTracker)


class TestEDHFlowPFInitFromGaussian:
    """Tests for init_from_gaussian method."""
    
    def test_init_from_gaussian_basic(self, ekf_tracker_linear, simple_linear_system):
        """Test initialization from Gaussian distribution."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=50, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        mean0 = np.array([0.5])
        cov0 = np.array([[0.3]])
        
        state = pf.init_from_gaussian(mean0, cov0)
        
        assert isinstance(state, PFState)
        assert state.particles.shape == (50, 1)
        assert state.weights.shape == (50,)
        assert state.mean.shape == (1,)
        assert state.cov.shape == (1, 1)
    
    def test_init_equal_weights(self, ekf_tracker_linear, simple_linear_system):
        """Test that initial weights are equal."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=100, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        mean0 = np.array([0.5])
        cov0 = np.array([[0.3]])
        state = pf.init_from_gaussian(mean0, cov0)
        
        expected_weight = 1.0 / 100
        np.testing.assert_allclose(state.weights, expected_weight, rtol=1e-10)
        np.testing.assert_allclose(np.sum(state.weights), 1.0)
    
    def test_init_particles_distributed(self, ekf_tracker_linear, simple_linear_system):
        """Test that particles are distributed around mean."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=500, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        mean0 = np.array([2.0])
        cov0 = np.array([[0.5]])
        state = pf.init_from_gaussian(mean0, cov0)
        
        # Particle mean should be close to mean0
        particle_mean = np.mean(state.particles, axis=0)
        np.testing.assert_allclose(particle_mean, mean0, atol=0.1)
        
        # Particle variance should be close to cov0
        particle_var = np.var(state.particles, axis=0)
        np.testing.assert_allclose(particle_var, np.diag(cov0), rtol=0.2)
    
    def test_init_deterministic_with_seed(self, ekf_tracker_linear, simple_linear_system):
        """Test that initialization is deterministic with same seed."""
        sys = simple_linear_system
        
        config1 = EDHConfig(n_particles=100, rng=np.random.default_rng(456))
        pf1 = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config1
        )
        
        # Create new tracker for pf2
        ekf2 = ExtendedKalmanFilter(
            g=sys['g_edh'], h=sys['h'],
            jac_g=sys['jac_g'], jac_h=sys['jac_h'],
            Q=sys['Q'], R=sys['R']
        )
        tracker2 = EKFTracker(ekf2, EKFState(mean=np.array([0.5]), cov=np.array([[0.3]]), t=0))
        
        config2 = EDHConfig(n_particles=100, rng=np.random.default_rng(456))
        pf2 = EDHFlowPF(
            tracker=tracker2,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config2
        )
        
        mean0 = np.array([0.5])
        cov0 = np.array([[0.3]])
        
        state1 = pf1.init_from_gaussian(mean0, cov0)
        state2 = pf2.init_from_gaussian(mean0, cov0)
        
        np.testing.assert_array_equal(state1.particles, state2.particles)


class TestEDHFlowPFStep:
    """Tests for the step method."""

    def test_step_basic_execution(self, ekf_tracker_linear, simple_linear_system):
        """Test that step executes without errors."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=50, n_lambda_steps=4, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        mean0 = np.array([0.5])
        cov0 = np.array([[0.3]])
        state = pf.init_from_gaussian(mean0, cov0)
        
        z = np.array([0.6])
        new_state = pf.step(state, z)
        
        assert isinstance(new_state, PFState)
        assert new_state.particles.shape == (50, 1)
        assert new_state.weights.shape == (50,)
    
    def test_step_output_shapes(self, ekf_tracker_linear, simple_linear_system):
        """Test that step returns correct shapes."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=100, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        z = np.array([0.6])
        new_state = pf.step(state, z)
        
        assert new_state.particles.shape == (100, 1)
        assert new_state.weights.shape == (100,)
        assert new_state.mean.shape == (1,)
        assert new_state.cov.shape == (1, 1)
    
    def test_step_weights_normalized(self, ekf_tracker_linear, simple_linear_system):
        """Test that weights sum to 1 after step."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=100, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        z = np.array([0.6])
        new_state = pf.step(state, z)
        
        np.testing.assert_allclose(np.sum(new_state.weights), 1.0, rtol=1e-10)
    
    def test_step_particles_evolve(self, ekf_tracker_linear, simple_linear_system):
        """Test that particles change after step."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=100, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        particles_before = state.particles.copy()
        
        z = np.array([0.6])
        new_state = pf.step(state, z)
        
        # Particles should have evolved
        assert not np.allclose(new_state.particles, particles_before)
    
    def test_step_multiple_sequential(self, ekf_tracker_linear, simple_linear_system):
        """Test multiple sequential steps."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=100, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        
        observations = [0.6, 0.7, 0.65, 0.8]
        for z_val in observations:
            z = np.array([z_val])
            state = pf.step(state, z)
            
            # Check validity at each step
            assert np.isfinite(state.particles).all()
            assert np.isfinite(state.weights).all()
            np.testing.assert_allclose(np.sum(state.weights), 1.0, rtol=1e-10)


class TestEDHFlowPFIntegrators:
    """Tests for different flow integrators."""
    
    def test_euler_integrator(self, ekf_tracker_linear, simple_linear_system):
        """Test EDHFlowPF with Euler integrator."""
        sys = simple_linear_system
        config = EDHConfig(
            n_particles=50,
            n_lambda_steps=8,
            flow_integrator="euler",
            rng=np.random.default_rng(123)
        )
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        z = np.array([0.6])
        new_state = pf.step(state, z)
        
        assert np.isfinite(new_state.particles).all()
        assert np.isfinite(new_state.weights).all()
    
    def test_rk4_integrator(self, ekf_tracker_linear, simple_linear_system):
        """Test EDHFlowPF with RK4 integrator."""
        sys = simple_linear_system
        config = EDHConfig(
            n_particles=50,
            n_lambda_steps=8,
            flow_integrator="rk4",
            rng=np.random.default_rng(123)
        )
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        z = np.array([0.6])
        new_state = pf.step(state, z)
        
        assert np.isfinite(new_state.particles).all()
        assert np.isfinite(new_state.weights).all()
    
    def test_euler_vs_rk4_comparison(self, simple_linear_system):
        """Compare Euler and RK4 integrators."""
        sys = simple_linear_system
        
        # Create two separate trackers - use g_ekf for EKF, not g_edh
        ekf1 = ExtendedKalmanFilter(
            g=sys['g_ekf'], h=sys['h'],
            jac_g=sys['jac_g'], jac_h=sys['jac_h'],
            Q=sys['Q'], R=sys['R']
        )
        tracker1 = EKFTracker(ekf1, EKFState(mean=np.array([0.5]), cov=np.array([[0.3]]), t=0))
        
        ekf2 = ExtendedKalmanFilter(
            g=sys['g_ekf'], h=sys['h'],
            jac_g=sys['jac_g'], jac_h=sys['jac_h'],
            Q=sys['Q'], R=sys['R']
        )
        tracker2 = EKFTracker(ekf2, EKFState(mean=np.array([0.5]), cov=np.array([[0.3]]), t=0))
        
        config_euler = EDHConfig(n_particles=100, flow_integrator="euler", rng=np.random.default_rng(789))
        config_rk4 = EDHConfig(n_particles=100, flow_integrator="rk4", rng=np.random.default_rng(789))
        
        pf_euler = EDHFlowPF(
            tracker=tracker1, g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'], log_like_pdf=sys['log_like_pdf'],
            R=sys['R'], config=config_euler
        )
        
        pf_rk4 = EDHFlowPF(
            tracker=tracker2, g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'], log_like_pdf=sys['log_like_pdf'],
            R=sys['R'], config=config_rk4
        )
        
        state_euler = pf_euler.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        state_rk4 = pf_rk4.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        
        z = np.array([0.6])
        state_euler = pf_euler.step(state_euler, z)
        state_rk4 = pf_rk4.step(state_rk4, z)
        
        # Both should produce valid results
        assert np.isfinite(state_euler.mean).all()
        assert np.isfinite(state_rk4.mean).all()
        
        # Results should be similar but not identical
        np.testing.assert_allclose(state_euler.mean, state_rk4.mean, rtol=0.5)


class TestEDHFlowPFResampling:
    """Tests for resampling functionality."""
    
    def test_resampling_disabled_by_default(self, ekf_tracker_linear, simple_linear_system):
        """Test that resampling is disabled by default."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=100, resample_ess_ratio=0.0, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        
        # Force weight degeneracy
        state.weights[0] = 0.9
        state.weights[1:] = 0.1 / 99
        
        z = np.array([0.6])
        new_state = pf.step(state, z)
        
        # Weights should not be reset to uniform (no resampling)
        assert not np.allclose(new_state.weights, 1.0/100)
    
    def test_resampling_enabled(self, ekf_tracker_linear, simple_linear_system):
        """Test resampling when enabled."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=100, resample_ess_ratio=0.8, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        z = np.array([0.6])
        new_state = pf.step(state, z)
        
        # Should still have valid weights
        np.testing.assert_allclose(np.sum(new_state.weights), 1.0)


class TestEDHFlowPFWithProcessNoise:
    """Tests for process noise handling."""
    
    def test_with_custom_process_noise_sampler(self, ekf_tracker_linear, simple_linear_system):
        """Test EDHFlowPF with custom process noise sampler."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=50, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        
        def process_noise_sampler(N, nx):
            """Sample from N(0, Q)."""
            return config.rng.multivariate_normal(np.zeros(nx), sys['Q'], size=N)
        
        z = np.array([0.6])
        new_state = pf.step(state, z, process_noise_sampler=process_noise_sampler)
        
        assert np.isfinite(new_state.particles).all()
        assert np.isfinite(new_state.weights).all()


class TestEDHFlowPFCovarianceProperties:
    """Tests for covariance properties."""

    def test_covariance_is_symmetric(self, ekf_tracker_linear, simple_linear_system):
        """Test that output covariance is symmetric."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=100, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        z = np.array([0.6])
        new_state = pf.step(state, z)
        
        np.testing.assert_allclose(new_state.cov, new_state.cov.T, rtol=1e-10)
    
    def test_covariance_positive_definite(self, ekf_tracker_linear, simple_linear_system):
        """Test that output covariance is positive definite."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=100, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        z = np.array([0.6])
        new_state = pf.step(state, z)
        
        # Check eigenvalues are positive
        eigvals = np.linalg.eigvalsh(new_state.cov)
        assert np.all(eigvals > -1e-10)


class TestEDHFlowPFEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_with_different_lambda_steps(self, simple_linear_system):
        """Test EDHFlowPF with different numbers of lambda steps."""
        sys = simple_linear_system
        
        for n_lambda in [1, 4, 8, 16]:
            config = EDHConfig(n_particles=50, n_lambda_steps=n_lambda, rng=np.random.default_rng(123))
            
            # Create fresh tracker - use g_ekf for EKF, not g_edh
            ekf = ExtendedKalmanFilter(
                g=sys['g_ekf'], h=sys['h'],
                jac_g=sys['jac_g'], jac_h=sys['jac_h'],
                Q=sys['Q'], R=sys['R']
            )
            tracker = EKFTracker(ekf, EKFState(mean=np.array([0.5]), cov=np.array([[0.3]]), t=0))
            
            pf = EDHFlowPF(
                tracker=tracker,
                g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
                log_trans_pdf=sys['log_trans_pdf'],
                log_like_pdf=sys['log_like_pdf'],
                R=sys['R'],
                config=config
            )
            
            state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
            z = np.array([0.6])
            new_state = pf.step(state, z)
            
            assert np.isfinite(new_state.mean).all()
            assert np.isfinite(new_state.cov).all()
    
    def test_numerical_stability(self, ekf_tracker_linear, simple_linear_system):
        """Test numerical stability over long sequences."""
        sys = simple_linear_system
        config = EDHConfig(n_particles=100, rng=np.random.default_rng(123))
        
        pf = EDHFlowPF(
            tracker=ekf_tracker_linear,
            g=sys['g_edh'], h=sys['h'], jacobian_h=sys['jac_h'],
            log_trans_pdf=sys['log_trans_pdf'],
            log_like_pdf=sys['log_like_pdf'],
            R=sys['R'],
            config=config
        )
        
        state = pf.init_from_gaussian(np.array([0.5]), np.array([[0.3]]))
        
        # Run for 50 steps
        for t in range(50):
            z = np.array([0.5 + 0.1 * np.sin(t * 0.1)])
            state = pf.step(state, z)
            
            # Check for numerical issues
            assert np.isfinite(state.particles).all(), f"Non-finite particles at step {t}"
            assert np.isfinite(state.weights).all(), f"Non-finite weights at step {t}"
            assert np.isfinite(state.mean).all(), f"Non-finite mean at step {t}"
            assert np.isfinite(state.cov).all(), f"Non-finite cov at step {t}"
            np.testing.assert_allclose(np.sum(state.weights), 1.0, rtol=1e-9)
