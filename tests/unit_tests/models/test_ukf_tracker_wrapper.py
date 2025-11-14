"""Unit tests for UKFTracker wrapper in EDH particle filter."""

import numpy as np
import pytest
from models.EDH_particle_filter import UKFTracker
from models.unscented_kalman_filter import UnscentedKalmanFilter, UKFState


@pytest.fixture
def simple_sv_system_ukf():
    """Simple 1D stochastic volatility system for UKF testing."""
    alpha = 0.9
    sigma = 0.2
    beta = 1.0
    
    Q = np.array([[sigma**2]])
    R = np.array([[0.1]])
    
    def g(x, u):
        """Process model: x_k = alpha * x_{k-1}."""
        return np.array([alpha * x[0]])
    
    def h(x):
        """Observation model: y_k = beta * exp(0.5 * x_k)."""
        return np.array([beta * np.exp(0.5 * x[0])])
    
    ukf = UnscentedKalmanFilter(g=g, h=h, Q=Q, R=R, alpha=1e-3, beta=2.0, kappa=0.0)
    
    return {
        'ukf': ukf,
        'g': g,
        'h': h,
        'Q': Q,
        'R': R,
        'alpha_param': alpha,
        'beta_param': beta
    }


@pytest.fixture
def ukf_initial_state():
    """Initial UKF state."""
    mean0 = np.array([0.5])
    cov0 = np.array([[0.3]])
    return UKFState(mean=mean0, cov=cov0, t=0)


class TestUKFTrackerInitialization:
    """Tests for UKFTracker initialization."""
    
    def test_tracker_creation(self, simple_sv_system_ukf, ukf_initial_state):
        """Test basic UKFTracker creation."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        assert tracker.ukf is ukf
        assert tracker.state is ukf_initial_state
        assert tracker.state.t == 0
        np.testing.assert_array_equal(tracker.state.mean, ukf_initial_state.mean)
        np.testing.assert_array_equal(tracker.state.cov, ukf_initial_state.cov)
    
    def test_past_mean_initialized(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that past_mean is initialized correctly."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        np.testing.assert_array_equal(tracker.past_mean, ukf_initial_state.mean)
        # Should be a copy, not the same object
        assert tracker.past_mean is not ukf_initial_state.mean
    
    def test_tracker_with_different_dimensions(self):
        """Test UKFTracker with multi-dimensional state."""
        # 2D state, 1D observation
        nx, nz = 2, 1
        
        def g(x, u):
            return np.array([0.9 * x[0] + 0.1 * x[1], 0.8 * x[1]])
        
        def h(x):
            return np.array([x[0] + 0.5 * x[1]])
        
        Q = np.diag([0.05, 0.02])
        R = np.array([[0.1]])
        
        ukf = UnscentedKalmanFilter(g=g, h=h, Q=Q, R=R)
        initial_state = UKFState(mean=np.array([0.5, -0.3]), cov=np.eye(2) * 0.3, t=0)
        
        tracker = UKFTracker(ukf, initial_state)
        
        assert tracker.state.mean.shape == (2,)
        assert tracker.state.cov.shape == (2, 2)
        assert tracker.past_mean.shape == (2,)


class TestUKFTrackerPredict:
    """Tests for UKFTracker.predict() method."""
    
    def test_predict_returns_mean_and_cov(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that predict returns mean and covariance."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        m_pred, P_pred = tracker.predict()
        
        assert isinstance(m_pred, np.ndarray)
        assert isinstance(P_pred, np.ndarray)
        assert m_pred.shape == (1,)
        assert P_pred.shape == (1, 1)
    
    def test_predict_updates_state(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that predict updates internal state."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        original_mean = tracker.state.mean.copy()
        original_t = tracker.state.t
        
        m_pred, P_pred = tracker.predict()
        
        # State should be updated
        np.testing.assert_array_equal(tracker.state.mean, m_pred)
        np.testing.assert_array_equal(tracker.state.cov, P_pred)
        assert tracker.state.t == original_t + 1
    
    def test_predict_stores_past_mean(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that predict stores the previous mean."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        original_mean = tracker.state.mean.copy()
        m_pred, P_pred = tracker.predict()
        
        # past_mean should contain the mean before prediction
        np.testing.assert_array_equal(tracker.past_mean, original_mean)
    
    def test_predict_propagates_state_correctly(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that predict propagates state according to process model."""
        ukf = simple_sv_system_ukf['ukf']
        alpha = simple_sv_system_ukf['alpha_param']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        x0 = ukf_initial_state.mean[0]
        m_pred, P_pred = tracker.predict()
        
        # Expected: x_k = alpha * x_{k-1}
        expected_mean = alpha * x0
        np.testing.assert_allclose(m_pred[0], expected_mean, rtol=1e-5)
    
    def test_predict_adds_uncertainty(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that predict adds process noise to covariance."""
        ukf = simple_sv_system_ukf['ukf']
        alpha = simple_sv_system_ukf['alpha_param']
        Q = simple_sv_system_ukf['Q']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        P0 = ukf_initial_state.cov[0, 0]
        m_pred, P_pred = tracker.predict()
        
        # For linear system: P_pred ≈ alpha^2 * P0 + Q
        # Check that Q contribution is present
        assert P_pred[0, 0] >= Q[0, 0] * 0.9  # Allow numerical error
    
    def test_multiple_predicts(self, simple_sv_system_ukf, ukf_initial_state):
        """Test multiple consecutive predictions."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        means = [tracker.state.mean.copy()]
        
        for _ in range(5):
            m_pred, P_pred = tracker.predict()
            means.append(m_pred.copy())
        
        # All means should be different
        for i in range(len(means) - 1):
            assert not np.allclose(means[i], means[i+1])
        
        # Time should advance
        assert tracker.state.t == 5


class TestUKFTrackerUpdate:
    """Tests for UKFTracker.update() method."""
    
    def test_update_returns_mean_and_cov(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that update returns mean and covariance."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        # Do prediction first
        tracker.predict()
        
        # Generate observation
        z = np.array([1.5])
        
        m_post, P_post = tracker.update(z)
        
        assert isinstance(m_post, np.ndarray)
        assert isinstance(P_post, np.ndarray)
        assert m_post.shape == (1,)
        assert P_post.shape == (1, 1)
    
    def test_update_updates_state(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that update modifies internal state."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        tracker.predict()
        m_before = tracker.state.mean.copy()
        
        z = np.array([1.5])
        m_post, P_post = tracker.update(z)
        
        # State should be updated
        np.testing.assert_array_equal(tracker.state.mean, m_post)
        np.testing.assert_array_equal(tracker.state.cov, P_post)
        # Mean should change after update
        assert not np.allclose(m_post, m_before)
    
    def test_update_reduces_uncertainty(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that update reduces covariance (information from measurement)."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        tracker.predict()
        P_before = tracker.state.cov[0, 0]
        
        z = np.array([1.5])
        m_post, P_post = tracker.update(z)
        
        # Covariance should decrease after update
        assert P_post[0, 0] < P_before
    
    def test_update_without_predict_works(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that update can be called without prior predict."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        z = np.array([1.5])
        m_post, P_post = tracker.update(z)
        
        assert m_post.shape == (1,)
        assert P_post.shape == (1, 1)
        assert np.isfinite(m_post).all()
        assert np.isfinite(P_post).all()


class TestUKFTrackerGetPastMean:
    """Tests for UKFTracker.get_past_mean() method."""
    
    def test_get_past_mean_returns_array(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that get_past_mean returns numpy array."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        past_mean = tracker.get_past_mean()
        
        assert isinstance(past_mean, np.ndarray)
        assert past_mean.shape == (1,)
    
    def test_get_past_mean_initial_value(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that get_past_mean returns initial mean before predict."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        past_mean = tracker.get_past_mean()
        
        np.testing.assert_array_equal(past_mean, ukf_initial_state.mean)
    
    def test_get_past_mean_after_predict(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that get_past_mean returns mean before last predict."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        mean_before_predict = tracker.state.mean.copy()
        tracker.predict()
        past_mean = tracker.get_past_mean()
        
        np.testing.assert_array_equal(past_mean, mean_before_predict)
    
    def test_get_past_mean_unchanged_by_update(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that get_past_mean is not changed by update."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        tracker.predict()
        past_mean_before = tracker.get_past_mean().copy()
        
        z = np.array([1.5])
        tracker.update(z)
        past_mean_after = tracker.get_past_mean()
        
        np.testing.assert_array_equal(past_mean_after, past_mean_before)
    
    def test_get_past_mean_multiple_cycles(self, simple_sv_system_ukf, ukf_initial_state):
        """Test get_past_mean over multiple predict-update cycles."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        current_means = []
        
        for i in range(3):
            tracker.predict()
            z = np.array([1.0 + 0.1 * i])
            tracker.update(z)
            
            current_means.append(tracker.state.mean.copy())
        
        # Each cycle should produce different results
        for i in range(len(current_means) - 1):
            assert not np.allclose(current_means[i], current_means[i+1])


class TestUKFTrackerFullCycle:
    """Tests for complete predict-update cycles."""
    
    def test_predict_update_cycle(self, simple_sv_system_ukf, ukf_initial_state):
        """Test a complete predict-update cycle."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        # Predict
        m_pred, P_pred = tracker.predict()
        assert tracker.state.t == 1
        
        # Update
        z = np.array([1.5])
        m_post, P_post = tracker.update(z)
        
        # Time should remain at 1 after update
        assert tracker.state.t == 1
        
        # Posterior uncertainty should be less than prior
        assert P_post[0, 0] < P_pred[0, 0]
    
    def test_multiple_cycles(self, simple_sv_system_ukf, ukf_initial_state):
        """Test multiple predict-update cycles."""
        ukf = simple_sv_system_ukf['ukf']
        tracker = UKFTracker(ukf, ukf_initial_state)
        
        observations = [1.5, 1.6, 1.4, 1.7, 1.3]
        
        means = []
        covs = []
        
        for z_val in observations:
            tracker.predict()
            m, P = tracker.update(np.array([z_val]))
            means.append(m[0])
            covs.append(P[0, 0])
        
        assert len(means) == 5
        assert len(covs) == 5
        
        # All results should be finite
        assert all(np.isfinite(m) for m in means)
        assert all(np.isfinite(c) and c > 0 for c in covs)
    
    def test_tracker_matches_direct_ukf_usage(self, simple_sv_system_ukf, ukf_initial_state):
        """Test that tracker produces same results as direct UKF usage."""
        ukf1 = simple_sv_system_ukf['ukf']
        ukf2 = UnscentedKalmanFilter(
            g=simple_sv_system_ukf['g'],
            h=simple_sv_system_ukf['h'],
            Q=simple_sv_system_ukf['Q'],
            R=simple_sv_system_ukf['R'],
            alpha=1e-3, beta=2.0, kappa=0.0
        )
        
        state1 = UKFState(
            mean=ukf_initial_state.mean.copy(),
            cov=ukf_initial_state.cov.copy(),
            t=ukf_initial_state.t
        )
        state2 = UKFState(
            mean=ukf_initial_state.mean.copy(),
            cov=ukf_initial_state.cov.copy(),
            t=ukf_initial_state.t
        )
        
        tracker = UKFTracker(ukf1, state1)
        
        observations = [1.5, 1.6, 1.4]
        
        for z_val in observations:
            z = np.array([z_val])
            
            # Using tracker
            m_tracker, P_tracker = tracker.predict()
            m_tracker, P_tracker = tracker.update(z)
            
            # Using UKF directly
            state2 = ukf2.predict(state2, u=None)
            state2 = ukf2.update(state2, z)
            
            # Should produce identical results
            np.testing.assert_allclose(m_tracker, state2.mean, rtol=1e-10)
            np.testing.assert_allclose(P_tracker, state2.cov, rtol=1e-10)


class TestUKFTrackerEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_tracker_with_zero_initial_covariance(self, simple_sv_system_ukf):
        """Test tracker behavior with zero initial covariance."""
        ukf = simple_sv_system_ukf['ukf']
        initial_state = UKFState(mean=np.array([0.5]), cov=np.zeros((1, 1)), t=0)
        
        tracker = UKFTracker(ukf, initial_state)
        
        # Should still work (process noise will add uncertainty)
        m_pred, P_pred = tracker.predict()
        Q = simple_sv_system_ukf['Q']
        assert P_pred[0, 0] >= Q[0, 0] * 0.9  # At minimum, Q is added
    
    def test_tracker_with_large_state_values(self, simple_sv_system_ukf, ukf_initial_state):
        """Test tracker with large state values."""
        ukf = simple_sv_system_ukf['ukf']
        large_state = UKFState(mean=np.array([10.0]), cov=np.array([[5.0]]), t=0)
        
        tracker = UKFTracker(ukf, large_state)
        
        m_pred, P_pred = tracker.predict()
        
        # Should handle large values without overflow
        assert np.isfinite(m_pred).all()
        assert np.isfinite(P_pred).all()
    
    def test_tracker_state_management(self, simple_sv_system_ukf):
        """Test that tracker manages state correctly."""
        ukf = simple_sv_system_ukf['ukf']
        original_state = UKFState(mean=np.array([0.5]), cov=np.array([[0.3]]), t=0)
        original_mean_value = original_state.mean[0]
        
        tracker = UKFTracker(ukf, original_state)
        
        # Verify tracker stored the initial mean correctly
        np.testing.assert_array_equal(tracker.past_mean, np.array([original_mean_value]))
        
        # After predict, past_mean should still be original value
        tracker.predict()
        assert tracker.past_mean[0] == original_mean_value


class TestUKFTrackerVsEKFTracker:
    """Compare UKF and EKF tracker behavior."""
    
    def test_ukf_and_ekf_tracker_api_consistency(self):
        """Test that UKF and EKF trackers have consistent APIs."""
        from models.EDH_particle_filter import EKFTracker
        from models.extended_kalman_filter import ExtendedKalmanFilter, EKFState
        
        # Define system
        alpha = 0.9
        Q = np.array([[0.04]])
        R = np.array([[0.1]])
        
        def g(x, u):
            return np.array([alpha * x[0]])
        
        def h(x):
            return np.array([x[0]])
        
        def jac_g(x, u):
            return np.array([[alpha]])
        
        def jac_h(x):
            return np.array([[1.0]])
        
        # Create EKF and UKF trackers
        ekf = ExtendedKalmanFilter(g=g, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q, R=R)
        ukf = UnscentedKalmanFilter(g=g, h=h, Q=Q, R=R)
        
        ekf_state = EKFState(mean=np.array([0.5]), cov=np.array([[0.3]]), t=0)
        ukf_state = UKFState(mean=np.array([0.5]), cov=np.array([[0.3]]), t=0)
        
        ekf_tracker = EKFTracker(ekf, ekf_state)
        ukf_tracker = UKFTracker(ukf, ukf_state)
        
        # Both should have same methods
        assert hasattr(ekf_tracker, 'predict')
        assert hasattr(ukf_tracker, 'predict')
        assert hasattr(ekf_tracker, 'update')
        assert hasattr(ukf_tracker, 'update')
        assert hasattr(ekf_tracker, 'get_past_mean')
        assert hasattr(ukf_tracker, 'get_past_mean')
        
        # Both should work the same way
        z = np.array([0.8])
        
        ekf_m_pred, ekf_P_pred = ekf_tracker.predict()
        ukf_m_pred, ukf_P_pred = ukf_tracker.predict()
        
        ekf_m_post, ekf_P_post = ekf_tracker.update(z)
        ukf_m_post, ukf_P_post = ukf_tracker.update(z)
        
        # Should produce similar (but not identical) results for linear system
        np.testing.assert_allclose(ekf_m_post, ukf_m_post, rtol=0.1)
        np.testing.assert_allclose(ekf_P_post, ukf_P_post, rtol=0.1)
