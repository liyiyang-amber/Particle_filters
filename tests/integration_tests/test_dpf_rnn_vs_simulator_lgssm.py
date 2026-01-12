"""Integration tests for DPF with RNN resampling on LGSSM."""

import numpy as np
import pytest
import tensorflow as tf


pytestmark = [pytest.mark.integration, pytest.mark.tensorflow]

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.DPF_RNN_resampling import DifferentiableParticleFilterRNN
from simulator.simulator_LGSSM import simulate_lgssm


def linear_transition_fn(x_prev, params):
    """Linear transition for LGSSM compatible with DPF_RNN."""
    a = params.get('a', 0.9)
    sigma_q = params.get('sigma_q', 0.5)
    
    a = tf.convert_to_tensor(a, dtype=tf.float32)
    sigma_q = tf.convert_to_tensor(sigma_q, dtype=tf.float32)
    
    a = tf.reshape(a, (1, 1, 1))
    sigma_q = tf.reshape(sigma_q, (1, 1, 1))
    
    eps = tf.random.normal(tf.shape(x_prev), dtype=tf.float32)
    x_new = a * x_prev + sigma_q * eps
    return x_new


def linear_log_likelihood_fn(x, y, params):
    """Linear Gaussian observation likelihood compatible with DPF_RNN."""
    sigma_r = params.get('sigma_r', 0.5)
    sigma_r = tf.convert_to_tensor(sigma_r, dtype=tf.float32)
    
    # Broadcast y to match x
    y = tf.expand_dims(y, axis=1)
    y = tf.broadcast_to(y, tf.shape(x))
    
    diff = y - x
    var = sigma_r ** 2
    log_norm_const = -0.5 * tf.math.log(2.0 * np.pi * var)
    log_lik = log_norm_const - 0.5 * (diff ** 2) / var
    return tf.squeeze(log_lik, axis=-1)


@pytest.fixture
def lgssm_1d_simple():
    """Simple 1D LGSSM for testing."""
    # State: x_t = 0.9 * x_{t-1} + noise(0, 0.5^2)
    # Obs:   y_t = x_t + noise(0, 0.7^2)
    a_true = 0.9
    sigma_q = 0.5
    sigma_r = 0.7
    
    T = 50
    x = np.zeros((T + 1, 1), dtype=np.float32)
    y = np.zeros((T, 1), dtype=np.float32)
    
    np.random.seed(42)
    x[0] = 0.0
    for t in range(T):
        x[t + 1] = a_true * x[t] + sigma_q * np.random.randn(1)
        y[t] = x[t + 1] + sigma_r * np.random.randn(1)
    
    return {
        'x': x, 'y': y, 'a': a_true,
        'sigma_q': sigma_q, 'sigma_r': sigma_r, 'T': T
    }


@pytest.mark.integration
class TestDPFRNNSimple1D:
    """Integration tests for DPF RNN on simple 1D system."""
    
    def test_dpf_rnn_tracks_1d_system(self, lgssm_1d_simple):
        """Test that DPF RNN can track 1D linear system."""
        data = lgssm_1d_simple
        
        dpf_rnn = DifferentiableParticleFilterRNN(
            n_particles=50,
            state_dim=1,
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            rnn_type='lstm',
            rnn_hidden_dim=32,
            rnn_num_layers=1,
            use_weight_features=True,
            use_particle_features=True,
            temperature=1.0,
            use_baseline_resampling=False
        )
        
        # Prepare data
        y_batch = data['y'][None, :, :]  # (1, T, 1)
        init_mean = np.array([0.0], dtype=np.float32)
        init_cov_chol = np.array([[1.0]], dtype=np.float32)
        
        params = {
            'a': data['a'],
            'sigma_q': data['sigma_q'],
            'sigma_r': data['sigma_r']
        }
        
        # Run filter
        particles_seq, logw_seq, assign_mats, ess_stats = dpf_rnn.filter(
            y_batch, init_mean, init_cov_chol, params, return_ess=True
        )
        
        # Compute posterior means
        w = tf.exp(logw_seq)
        posterior_means = tf.reduce_sum(w[..., None] * particles_seq, axis=2).numpy()[0]
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((posterior_means - data['x']) ** 2))
        
        # Should achieve tracking (may not be as good as optimal filter without training)
        assert rmse < 2.0, f"RMSE too high: {rmse}"
        
        # Check shapes
        assert particles_seq.shape == (1, data['T'] + 1, 50, 1)
        assert assign_mats.shape == (1, data['T'], 50, 50)
    
    def test_dpf_rnn_with_ess_monitoring(self, lgssm_1d_simple):
        """Test DPF RNN with ESS monitoring."""
        data = lgssm_1d_simple
        
        dpf_rnn = DifferentiableParticleFilterRNN(
            n_particles=40,
            state_dim=1,
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            rnn_type='gru',
            rnn_hidden_dim=24,
            use_baseline_resampling=False
        )
        
        y_batch = data['y'][None, :30, :]  # Use first 30 timesteps
        init_mean = np.array([0.0], dtype=np.float32)
        init_cov_chol = np.array([[1.0]], dtype=np.float32)
        
        params = {
            'a': data['a'],
            'sigma_q': data['sigma_q'],
            'sigma_r': data['sigma_r']
        }
        
        particles_seq, logw_seq, assign_mats, ess_stats = dpf_rnn.filter(
            y_batch, init_mean, init_cov_chol, params, return_ess=True
        )
        
        # Check ESS statistics
        assert 'mean_ess_before' in ess_stats
        assert 'mean_ess_after' in ess_stats
        assert 'mean_entropy_before' in ess_stats
        assert 'mean_entropy_after' in ess_stats
        
        # ESS after resampling should be high (uniform weights)
        assert ess_stats['mean_ess_after'].numpy() > 30  # Most particles have equal weight
    
    def test_dpf_rnn_baseline_mode(self, lgssm_1d_simple):
        """Test DPF RNN in baseline resampling mode."""
        data = lgssm_1d_simple
        
        dpf_baseline = DifferentiableParticleFilterRNN(
            n_particles=50,
            state_dim=1,
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            use_baseline_resampling=True  # Use simple weight-based resampling
        )
        
        y_batch = data['y'][None, :20, :]
        init_mean = np.array([0.0], dtype=np.float32)
        init_cov_chol = np.array([[1.0]], dtype=np.float32)
        
        params = {
            'a': data['a'],
            'sigma_q': data['sigma_q'],
            'sigma_r': data['sigma_r']
        }
        
        particles_seq, logw_seq, assign_mats, ess_stats = dpf_baseline.filter(
            y_batch, init_mean, init_cov_chol, params, return_ess=True
        )
        
        # Should complete without errors
        assert particles_seq.shape == (1, 21, 50, 1)
        
        # Compute posterior mean and RMSE
        w = tf.exp(logw_seq)
        posterior_means = tf.reduce_sum(w[..., None] * particles_seq, axis=2).numpy()[0]
        rmse = np.sqrt(np.mean((posterior_means - data['x'][:21]) ** 2))
        
        # Baseline should still work reasonably
        assert rmse < 2.5


@pytest.mark.integration
class TestDPFRNNFeatureConfigurations:
    """Test different feature configurations."""
    
    def test_weights_only_features(self, lgssm_1d_simple):
        """Test RNN with only weight features."""
        data = lgssm_1d_simple
        
        dpf = DifferentiableParticleFilterRNN(
            n_particles=40,
            state_dim=1,
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            use_weight_features=True,
            use_particle_features=False,
            use_baseline_resampling=False
        )
        
        y_batch = data['y'][None, :20, :]
        init_mean = np.array([0.0], dtype=np.float32)
        init_cov_chol = np.array([[1.0]], dtype=np.float32)
        
        params = {
            'a': data['a'],
            'sigma_q': data['sigma_q'],
            'sigma_r': data['sigma_r']
        }
        
        particles_seq, logw_seq, _, _ = dpf.filter(
            y_batch, init_mean, init_cov_chol, params, return_ess=True
        )
        
        # Should complete without errors
        assert particles_seq.shape == (1, 21, 40, 1)
    
    def test_particles_only_features(self, lgssm_1d_simple):
        """Test RNN with only particle features."""
        data = lgssm_1d_simple
        
        dpf = DifferentiableParticleFilterRNN(
            n_particles=40,
            state_dim=1,
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            use_weight_features=False,
            use_particle_features=True,
            use_baseline_resampling=False
        )
        
        y_batch = data['y'][None, :20, :]
        init_mean = np.array([0.0], dtype=np.float32)
        init_cov_chol = np.array([[1.0]], dtype=np.float32)
        
        params = {
            'a': data['a'],
            'sigma_q': data['sigma_q'],
            'sigma_r': data['sigma_r']
        }
        
        particles_seq, logw_seq, _, _ = dpf.filter(
            y_batch, init_mean, init_cov_chol, params, return_ess=True
        )
        
        # Should complete without errors
        assert particles_seq.shape == (1, 21, 40, 1)


@pytest.mark.integration
class TestDPFRNNRNNTypes:
    """Test different RNN architectures."""
    
    def test_lstm_architecture(self, lgssm_1d_simple):
        """Test DPF with LSTM."""
        data = lgssm_1d_simple
        
        dpf = DifferentiableParticleFilterRNN(
            n_particles=35,
            state_dim=1,
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            rnn_type='lstm',
            rnn_hidden_dim=28,
            rnn_num_layers=2
        )
        
        y_batch = data['y'][None, :15, :]
        init_mean = np.array([0.0], dtype=np.float32)
        init_cov_chol = np.array([[1.0]], dtype=np.float32)
        
        params = {
            'a': data['a'],
            'sigma_q': data['sigma_q'],
            'sigma_r': data['sigma_r']
        }
        
        particles_seq, logw_seq, _, _ = dpf.filter(
            y_batch, init_mean, init_cov_chol, params, return_ess=True
        )
        
        assert particles_seq.shape == (1, 16, 35, 1)
    
    def test_gru_architecture(self, lgssm_1d_simple):
        """Test DPF with GRU."""
        data = lgssm_1d_simple
        
        dpf = DifferentiableParticleFilterRNN(
            n_particles=35,
            state_dim=1,
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            rnn_type='gru',
            rnn_hidden_dim=28,
            rnn_num_layers=2
        )
        
        y_batch = data['y'][None, :15, :]
        init_mean = np.array([0.0], dtype=np.float32)
        init_cov_chol = np.array([[1.0]], dtype=np.float32)
        
        params = {
            'a': data['a'],
            'sigma_q': data['sigma_q'],
            'sigma_r': data['sigma_r']
        }
        
        particles_seq, logw_seq, _, _ = dpf.filter(
            y_batch, init_mean, init_cov_chol, params, return_ess=True
        )
        
        assert particles_seq.shape == (1, 16, 35, 1)


@pytest.mark.integration
class TestDPFRNNAssignmentMatrices:
    """Test assignment matrix properties."""
    
    def test_assignment_matrix_properties(self, lgssm_1d_simple):
        """Test that assignment matrices have correct properties."""
        data = lgssm_1d_simple
        
        dpf = DifferentiableParticleFilterRNN(
            n_particles=30,
            state_dim=1,
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            use_baseline_resampling=False
        )
        
        y_batch = data['y'][None, :10, :]
        init_mean = np.array([0.0], dtype=np.float32)
        init_cov_chol = np.array([[1.0]], dtype=np.float32)
        
        params = {
            'a': data['a'],
            'sigma_q': data['sigma_q'],
            'sigma_r': data['sigma_r']
        }
        
        particles_seq, logw_seq, assign_mats, _ = dpf.filter(
            y_batch, init_mean, init_cov_chol, params, return_ess=True
        )
        
        # Check assignment matrix properties
        assign_np = assign_mats.numpy()[0]  # (T, N, N)
        
        for t in range(10):
            A_t = assign_np[t]  # (N, N)
            
            # Each row should sum to 1 (each new particle is a convex combination)
            row_sums = np.sum(A_t, axis=1)
            np.testing.assert_allclose(row_sums, 1.0, rtol=1e-4)
            
            # All entries should be non-negative
            assert np.all(A_t >= -1e-6)
    
    def test_assignment_diversity_with_rnn(self, lgssm_1d_simple):
        """Test that RNN produces valid assignment probabilities."""
        data = lgssm_1d_simple
        
        dpf = DifferentiableParticleFilterRNN(
            n_particles=25,
            state_dim=1,
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            use_baseline_resampling=False
        )
        
        y_batch = data['y'][None, :5, :]
        init_mean = np.array([0.0], dtype=np.float32)
        init_cov_chol = np.array([[1.0]], dtype=np.float32)
        
        params = {
            'a': data['a'],
            'sigma_q': data['sigma_q'],
            'sigma_r': data['sigma_r']
        }
        
        particles_seq, logw_seq, assign_mats, _ = dpf.filter(
            y_batch, init_mean, init_cov_chol, params, return_ess=True
        )
        
        # Check that assignment matrix contains valid probabilities
        assign_np = assign_mats.numpy()[0, 0]  # First timestep, (N, N)
        
        # Each row should sum to 1 (valid probability distribution)
        row_sums = assign_np.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"Row sums not close to 1: {row_sums}"
        
        # All values should be non-negative
        assert (assign_np >= 0).all(), "Assignment probabilities should be non-negative"
        
        # Shape should be correct
        assert assign_np.shape == (25, 25)


@pytest.mark.integration
class TestDPFRNNNumericalStability:
    """Test numerical stability."""
    
    def test_stability_with_extreme_observations(self, lgssm_1d_simple):
        """Test that filter remains stable with extreme observations."""
        data = lgssm_1d_simple
        
        dpf = DifferentiableParticleFilterRNN(
            n_particles=40,
            state_dim=1,
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn
        )
        
        # Create observations with extreme values
        y_extreme = data['y'][:10].copy()
        y_extreme[5] = 10.0  # Extreme outlier
        y_batch = y_extreme[None, :, :]
        
        init_mean = np.array([0.0], dtype=np.float32)
        init_cov_chol = np.array([[1.0]], dtype=np.float32)
        
        params = {
            'a': data['a'],
            'sigma_q': data['sigma_q'],
            'sigma_r': data['sigma_r']
        }
        
        # Should not raise exception
        particles_seq, logw_seq, _, _ = dpf.filter(
            y_batch, init_mean, init_cov_chol, params, return_ess=True
        )
        
        # Check no NaNs
        assert not tf.reduce_any(tf.math.is_nan(particles_seq))
        assert not tf.reduce_any(tf.math.is_nan(logw_seq))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
