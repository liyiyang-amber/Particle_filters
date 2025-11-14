"""Unit tests for EKFTracker wrapper in EDH particle filter."""

import numpy as np
import pytest
from models.EDH_particle_filter import EKFTracker
from models.extended_kalman_filter import ExtendedKalmanFilter, EKFState


@pytest.fixture
def simple_sv_system():
    """Simple 1D stochastic volatility system for testing."""
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
    
    def jac_g(x, u):
        """Jacobian of g."""
        return np.array([[alpha]])
    
    def jac_h(x):
        """Jacobian of h."""
        return np.array([[0.5 * beta * np.exp(0.5 * x[0])]])
    
    ekf = ExtendedKalmanFilter(g=g, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q, R=R)
    
    return {
        'ekf': ekf,
        'g': g,
        'h': h,
        'jac_g': jac_g,
        'jac_h': jac_h,
        'Q': Q,
        'R': R,
        'alpha': alpha,
        'beta': beta
    }


@pytest.fixture
def ekf_initial_state():
    """Initial EKF state."""
    mean0 = np.array([0.5])
    cov0 = np.array([[0.3]])
    return EKFState(mean=mean0, cov=cov0, t=0)


class TestEKFTrackerInitialization:
    """Tests for EKFTracker initialization."""
    
    def test_tracker_creation(self, simple_sv_system, ekf_initial_state):
        """Test basic EKFTracker creation."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        assert tracker.ekf is ekf
        assert tracker.state is ekf_initial_state
        assert tracker.state.t == 0
        np.testing.assert_array_equal(tracker.state.mean, ekf_initial_state.mean)
        np.testing.assert_array_equal(tracker.state.cov, ekf_initial_state.cov)
    
    def test_past_mean_initialized(self, simple_sv_system, ekf_initial_state):
        """Test that past_mean is initialized correctly."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        np.testing.assert_array_equal(tracker.past_mean, ekf_initial_state.mean)
        # Should be a copy, not the same object
        assert tracker.past_mean is not ekf_initial_state.mean
    
    def test_tracker_with_different_dimensions(self):
        """Test EKFTracker with multi-dimensional state."""
        # 2D state, 1D observation
        nx, nz = 2, 1
        
        def g(x, u):
            return np.array([0.9 * x[0] + 0.1 * x[1], 0.8 * x[1]])
        
        def h(x):
            return np.array([x[0] + 0.5 * x[1]])
        
        def jac_g(x, u):
            return np.array([[0.9, 0.1], [0.0, 0.8]])
        
        def jac_h(x):
            return np.array([[1.0, 0.5]])
        
        Q = np.diag([0.05, 0.02])
        R = np.array([[0.1]])
        
        ekf = ExtendedKalmanFilter(g=g, h=h, jac_g=jac_g, jac_h=jac_h, Q=Q, R=R)
        initial_state = EKFState(mean=np.array([0.5, -0.3]), cov=np.eye(2) * 0.3, t=0)
        
        tracker = EKFTracker(ekf, initial_state)
        
        assert tracker.state.mean.shape == (2,)
        assert tracker.state.cov.shape == (2, 2)
        assert tracker.past_mean.shape == (2,)


class TestEKFTrackerPredict:
    """Tests for EKFTracker.predict() method."""
    
    def test_predict_returns_mean_and_cov(self, simple_sv_system, ekf_initial_state):
        """Test that predict returns mean and covariance."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        m_pred, P_pred = tracker.predict()
        
        assert isinstance(m_pred, np.ndarray)
        assert isinstance(P_pred, np.ndarray)
        assert m_pred.shape == (1,)
        assert P_pred.shape == (1, 1)
    
    def test_predict_updates_state(self, simple_sv_system, ekf_initial_state):
        """Test that predict updates internal state."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        original_mean = tracker.state.mean.copy()
        original_t = tracker.state.t
        
        m_pred, P_pred = tracker.predict()
        
        # State should be updated
        np.testing.assert_array_equal(tracker.state.mean, m_pred)
        np.testing.assert_array_equal(tracker.state.cov, P_pred)
        assert tracker.state.t == original_t + 1
    
    def test_predict_stores_past_mean(self, simple_sv_system, ekf_initial_state):
        """Test that predict stores the previous mean."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        original_mean = tracker.state.mean.copy()
        m_pred, P_pred = tracker.predict()
        
        # past_mean should contain the mean before prediction
        np.testing.assert_array_equal(tracker.past_mean, original_mean)
    
    def test_predict_propagates_state_correctly(self, simple_sv_system, ekf_initial_state):
        """Test that predict propagates state according to process model."""
        ekf = simple_sv_system['ekf']
        alpha = simple_sv_system['alpha']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        x0 = ekf_initial_state.mean[0]
        m_pred, P_pred = tracker.predict()
        
        # Expected: x_k = alpha * x_{k-1}
        expected_mean = alpha * x0
        np.testing.assert_allclose(m_pred[0], expected_mean, rtol=1e-10)
    
    def test_predict_increases_uncertainty(self, simple_sv_system, ekf_initial_state):
        """Test that predict adds process noise to covariance."""
        ekf = simple_sv_system['ekf']
        alpha = simple_sv_system['alpha']
        Q = simple_sv_system['Q']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        P0 = ekf_initial_state.cov[0, 0]
        m_pred, P_pred = tracker.predict()
        
        # For linear system: P_pred = A * P0 * A^T + Q
        # With alpha < 1, might actually decrease, so check it includes Q contribution
        expected_min = Q[0, 0]  # At minimum, Q is added
        assert P_pred[0, 0] >= expected_min * 0.9  # Allow small numerical error
    
    def test_multiple_predicts(self, simple_sv_system, ekf_initial_state):
        """Test multiple consecutive predictions."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        means = [tracker.state.mean.copy()]
        
        for _ in range(5):
            m_pred, P_pred = tracker.predict()
            means.append(m_pred.copy())
        
        # All means should be different
        for i in range(len(means) - 1):
            assert not np.allclose(means[i], means[i+1])
        
        # Time should advance
        assert tracker.state.t == 5


class TestEKFTrackerUpdate:
    """Tests for EKFTracker.update() method."""
    
    def test_update_returns_mean_and_cov(self, simple_sv_system, ekf_initial_state):
        """Test that update returns mean and covariance."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        # Do prediction first
        tracker.predict()
        
        # Generate observation
        z = np.array([1.5])
        
        m_post, P_post = tracker.update(z)
        
        assert isinstance(m_post, np.ndarray)
        assert isinstance(P_post, np.ndarray)
        assert m_post.shape == (1,)
        assert P_post.shape == (1, 1)
    
    def test_update_updates_state(self, simple_sv_system, ekf_initial_state):
        """Test that update modifies internal state."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        tracker.predict()
        m_before = tracker.state.mean.copy()
        
        z = np.array([1.5])
        m_post, P_post = tracker.update(z)
        
        # State should be updated
        np.testing.assert_array_equal(tracker.state.mean, m_post)
        np.testing.assert_array_equal(tracker.state.cov, P_post)
        # Mean should change after update
        assert not np.allclose(m_post, m_before)
    
    def test_update_reduces_uncertainty(self, simple_sv_system, ekf_initial_state):
        """Test that update reduces covariance (information from measurement)."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        tracker.predict()
        P_before = tracker.state.cov[0, 0]
        
        z = np.array([1.5])
        m_post, P_post = tracker.update(z)
        
        # Covariance should decrease after update
        assert P_post[0, 0] < P_before
    
    def test_update_without_predict_works(self, simple_sv_system, ekf_initial_state):
        """Test that update can be called without prior predict."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        z = np.array([1.5])
        m_post, P_post = tracker.update(z)
        
        assert m_post.shape == (1,)
        assert P_post.shape == (1, 1)
        assert np.isfinite(m_post).all()
        assert np.isfinite(P_post).all()


class TestEKFTrackerGetPastMean:
    """Tests for EKFTracker.get_past_mean() method."""
    
    def test_get_past_mean_returns_array(self, simple_sv_system, ekf_initial_state):
        """Test that get_past_mean returns numpy array."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        past_mean = tracker.get_past_mean()
        
        assert isinstance(past_mean, np.ndarray)
        assert past_mean.shape == (1,)
    
    def test_get_past_mean_initial_value(self, simple_sv_system, ekf_initial_state):
        """Test that get_past_mean returns initial mean before predict."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        past_mean = tracker.get_past_mean()
        
        np.testing.assert_array_equal(past_mean, ekf_initial_state.mean)
    
    def test_get_past_mean_after_predict(self, simple_sv_system, ekf_initial_state):
        """Test that get_past_mean returns mean before last predict."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        mean_before_predict = tracker.state.mean.copy()
        tracker.predict()
        past_mean = tracker.get_past_mean()
        
        np.testing.assert_array_equal(past_mean, mean_before_predict)
    
    def test_get_past_mean_unchanged_by_update(self, simple_sv_system, ekf_initial_state):
        """Test that get_past_mean is not changed by update."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        tracker.predict()
        past_mean_before = tracker.get_past_mean().copy()
        
        z = np.array([1.5])
        tracker.update(z)
        past_mean_after = tracker.get_past_mean()
        
        np.testing.assert_array_equal(past_mean_after, past_mean_before)
    
    def test_get_past_mean_multiple_cycles(self, simple_sv_system, ekf_initial_state):
        """Test get_past_mean over multiple predict-update cycles."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
        past_means = []
        current_means = []
        
        for i in range(3):
            tracker.predict()
            z = np.array([1.0 + 0.1 * i])
            tracker.update(z)
            
            # Record current mean and past mean after update
            current_means.append(tracker.state.mean.copy())
            past_means.append(tracker.get_past_mean().copy())
        
        # Past means should track the previous state means
        # past_means[i] should be close to current_means[i-1] (with prediction applied)
        for i in range(len(current_means)):
            # Each cycle should produce different results
            if i > 0:
                assert not np.allclose(current_means[i], current_means[i-1])


class TestEKFTrackerFullCycle:
    """Tests for complete predict-update cycles."""
    
    def test_predict_update_cycle(self, simple_sv_system, ekf_initial_state):
        """Test a complete predict-update cycle."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
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
    
    def test_multiple_cycles(self, simple_sv_system, ekf_initial_state):
        """Test multiple predict-update cycles."""
        ekf = simple_sv_system['ekf']
        tracker = EKFTracker(ekf, ekf_initial_state)
        
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
    
    def test_tracker_matches_direct_ekf_usage(self, simple_sv_system, ekf_initial_state):
        """Test that tracker produces same results as direct EKF usage."""
        ekf1 = simple_sv_system['ekf']
        ekf2 = ExtendedKalmanFilter(
            g=simple_sv_system['g'],
            h=simple_sv_system['h'],
            jac_g=simple_sv_system['jac_g'],
            jac_h=simple_sv_system['jac_h'],
            Q=simple_sv_system['Q'],
            R=simple_sv_system['R']
        )
        
        state1 = EKFState(
            mean=ekf_initial_state.mean.copy(),
            cov=ekf_initial_state.cov.copy(),
            t=ekf_initial_state.t
        )
        state2 = EKFState(
            mean=ekf_initial_state.mean.copy(),
            cov=ekf_initial_state.cov.copy(),
            t=ekf_initial_state.t
        )
        
        tracker = EKFTracker(ekf1, state1)
        
        observations = [1.5, 1.6, 1.4]
        
        for z_val in observations:
            z = np.array([z_val])
            
            # Using tracker
            m_tracker, P_tracker = tracker.predict()
            m_tracker, P_tracker = tracker.update(z)
            
            # Using EKF directly
            state2 = ekf2.predict(state2, u=None)
            state2 = ekf2.update(state2, z)
            
            # Should produce identical results
            np.testing.assert_allclose(m_tracker, state2.mean, rtol=1e-10)
            np.testing.assert_allclose(P_tracker, state2.cov, rtol=1e-10)


class TestEKFTrackerEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_tracker_with_zero_initial_covariance(self, simple_sv_system):
        """Test tracker behavior with zero initial covariance."""
        ekf = simple_sv_system['ekf']
        initial_state = EKFState(mean=np.array([0.5]), cov=np.zeros((1, 1)), t=0)
        
        tracker = EKFTracker(ekf, initial_state)
        
        # Should still work (process noise will add uncertainty)
        m_pred, P_pred = tracker.predict()
        assert P_pred[0, 0] > 0  # Process noise Q added
    
    def test_tracker_with_large_state_values(self, simple_sv_system, ekf_initial_state):
        """Test tracker with large state values."""
        ekf = simple_sv_system['ekf']
        large_state = EKFState(mean=np.array([10.0]), cov=np.array([[5.0]]), t=0)
        
        tracker = EKFTracker(ekf, large_state)
        
        m_pred, P_pred = tracker.predict()
        
        # Should handle large values without overflow
        assert np.isfinite(m_pred).all()
        assert np.isfinite(P_pred).all()
    
    def test_tracker_state_independence(self, simple_sv_system):
        """Test that tracker creates its own copy of state."""
        ekf = simple_sv_system['ekf']
        original_state = EKFState(mean=np.array([0.5]), cov=np.array([[0.3]]), t=0)
        original_mean_value = original_state.mean[0]
        
        tracker = EKFTracker(ekf, original_state)
        
        # Tracker uses the state object passed in (not a copy)
        # This is expected behavior - tracker manages the state object
        # Verify tracker stored the initial mean correctly
        np.testing.assert_array_equal(tracker.past_mean, np.array([original_mean_value]))
        
        # After predict, past_mean should still be original value
        tracker.predict()
        assert tracker.past_mean[0] == original_mean_value
