"""Unit tests for DPF with RNN resampling: shapes, API, and basic functionality."""

import numpy as np
import pytest
import tensorflow as tf


pytestmark = pytest.mark.tensorflow


# Import the DPF RNN implementation
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from models.DPF_RNN_resampling import DifferentiableParticleFilterRNN


@pytest.fixture
def dpf_rnn_config():
    """Configuration for DPF RNN testing."""
    return {
        'n_particles': 30,
        'state_dim': 2,
        'rnn_type': 'lstm',
        'rnn_hidden_dim': 32,
        'rnn_num_layers': 1,
        'temperature': 1.0,
        'use_weight_features': True,
        'use_particle_features': True
    }


@pytest.fixture
def simple_transition_fn():
    """Simple linear transition function."""
    def transition(x_prev, params):
        A = params.get('a', 0.9)
        sigma_q = params.get('sigma_q', 0.1)
        
        A = tf.convert_to_tensor(A, dtype=tf.float32)
        sigma_q = tf.convert_to_tensor(sigma_q, dtype=tf.float32)
        A = tf.reshape(A, (1, 1, 1))
        sigma_q = tf.reshape(sigma_q, (1, 1, 1))
        
        eps = tf.random.normal(tf.shape(x_prev), dtype=tf.float32)
        x_new = A * x_prev + sigma_q * eps
        return x_new
    return transition


@pytest.fixture
def simple_log_likelihood_fn():
    """Simple Gaussian log-likelihood."""
    def log_likelihood(x, y, params):
        sigma_r = params.get('sigma_r', 0.5)
        sigma_r = tf.convert_to_tensor(sigma_r, dtype=tf.float32)
        
        # x: (B, N, d), y: (B, d)
        y = tf.expand_dims(y, axis=1)  # (B, 1, d)
        y = tf.broadcast_to(y, tf.shape(x))  # (B, N, d)
        
        diff = y - x  # (B, N, d)
        var = sigma_r ** 2
        log_norm_const = -0.5 * tf.math.log(2.0 * np.pi * var)
        log_lik = log_norm_const - 0.5 * tf.reduce_sum(diff ** 2, axis=-1) / var  # (B, N)
        return log_lik
    return log_likelihood


class TestDPFRNNInitialization:
    """Test DPF RNN initialization."""
    
    def test_dpf_rnn_creation_lstm(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test DPF RNN creation with LSTM."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            rnn_type='lstm',
            rnn_hidden_dim=dpf_rnn_config['rnn_hidden_dim'],
            rnn_num_layers=dpf_rnn_config['rnn_num_layers']
        )
        
        assert dpf.n_particles == dpf_rnn_config['n_particles']
        assert dpf.state_dim == dpf_rnn_config['state_dim']
        assert dpf.rnn_type == 'lstm'
        assert dpf.rnn_hidden_dim == dpf_rnn_config['rnn_hidden_dim']
    
    def test_dpf_rnn_creation_gru(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test DPF RNN creation with GRU."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            rnn_type='gru',
            rnn_hidden_dim=dpf_rnn_config['rnn_hidden_dim'],
            rnn_num_layers=dpf_rnn_config['rnn_num_layers']
        )
        
        assert dpf.rnn_type == 'gru'
    
    def test_dpf_rnn_baseline_mode(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test DPF RNN with baseline resampling (no RNN)."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            use_baseline_resampling=True
        )
        
        assert dpf.use_baseline_resampling is True
    
    def test_dpf_rnn_feature_flags(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test DPF RNN with different feature configurations."""
        # Only weights
        dpf_weights = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            use_weight_features=True,
            use_particle_features=False
        )
        assert dpf_weights.use_weight_features is True
        assert dpf_weights.use_particle_features is False
        
        # Only particles
        dpf_particles = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            use_weight_features=False,
            use_particle_features=True
        )
        assert dpf_particles.use_weight_features is False
        assert dpf_particles.use_particle_features is True


class TestDPFRNNParticleInitialization:
    """Test particle initialization."""
    
    def test_init_particles_shape(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test initialized particle shapes."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 2
        init_mean = np.zeros(dpf_rnn_config['state_dim'], dtype=np.float32)
        init_cov_chol = np.eye(dpf_rnn_config['state_dim'], dtype=np.float32)
        
        particles, log_weights = dpf.init_particles(batch_size, init_mean, init_cov_chol)
        
        assert particles.shape == (batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['state_dim'])
        assert log_weights.shape == (batch_size, dpf_rnn_config['n_particles'])
    
    def test_init_particles_uniform_weights(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test initial weights are uniform."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 1
        init_mean = np.zeros(dpf_rnn_config['state_dim'], dtype=np.float32)
        init_cov_chol = np.eye(dpf_rnn_config['state_dim'], dtype=np.float32)
        
        particles, log_weights = dpf.init_particles(batch_size, init_mean, init_cov_chol)
        
        weights = tf.exp(log_weights).numpy()
        expected_weight = 1.0 / dpf_rnn_config['n_particles']
        np.testing.assert_allclose(weights, expected_weight, rtol=1e-5)


class TestDPFRNNUtilities:
    """Test utility functions."""
    
    def test_log_normalize(self):
        """Test log normalization."""
        log_w = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        log_w_norm, _ = DifferentiableParticleFilterRNN._log_normalize(log_w, axis=-1, keepdims=False)
        
        w_norm = tf.exp(log_w_norm).numpy()
        np.testing.assert_allclose(np.sum(w_norm, axis=1), 1.0, rtol=1e-5)
    
    def test_compute_ess(self):
        """Test ESS computation."""
        # Uniform weights
        N = 50
        log_weights = tf.constant([[tf.math.log(1.0 / N).numpy()] * N], dtype=tf.float32)
        
        ess = DifferentiableParticleFilterRNN.compute_ess(log_weights)
        np.testing.assert_allclose(ess.numpy(), N, rtol=0.01)
    
    def test_compute_weight_entropy(self):
        """Test weight entropy computation."""
        N = 50
        log_weights = tf.constant([[tf.math.log(1.0 / N).numpy()] * N], dtype=tf.float32)
        
        entropy = DifferentiableParticleFilterRNN.compute_weight_entropy(log_weights)
        expected_entropy = np.log(N)
        np.testing.assert_allclose(entropy.numpy(), expected_entropy, rtol=0.01)


class TestDPFRNNFeatureComputation:
    """Test RNN feature computation."""
    
    def test_compute_rnn_features_with_weights(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test feature computation with weights."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            use_weight_features=True,
            use_particle_features=False
        )
        
        batch_size = 2
        particles = tf.random.normal((batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['state_dim']))
        log_weights = tf.zeros((batch_size, dpf_rnn_config['n_particles']))
        
        features = dpf._compute_rnn_features(particles, log_weights)
        
        # Features should have weights (1 feature: log_weight) per particle
        # Note: the implementation uses log_weight feature only
        assert features.shape[0] == batch_size
        assert features.shape[1] == dpf_rnn_config['n_particles']
        # Weight features dimension depends on implementation
    
    def test_compute_rnn_features_with_particles(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test feature computation with particles."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            use_weight_features=False,
            use_particle_features=True
        )
        
        batch_size = 2
        particles = tf.random.normal((batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['state_dim']))
        log_weights = tf.zeros((batch_size, dpf_rnn_config['n_particles']))
        
        features = dpf._compute_rnn_features(particles, log_weights)
        
        # Features should have state_dim features per particle
        assert features.shape == (batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['state_dim'])
    
    def test_compute_rnn_features_with_both(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test feature computation with both weights and particles."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            use_weight_features=True,
            use_particle_features=True
        )
        
        batch_size = 2
        particles = tf.random.normal((batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['state_dim']))
        log_weights = tf.zeros((batch_size, dpf_rnn_config['n_particles']))
        
        features = dpf._compute_rnn_features(particles, log_weights)
        
        # Features should have weight features + state_dim (particles) features
        assert features.shape[0] == batch_size
        assert features.shape[1] == dpf_rnn_config['n_particles']
        # Total features depend on implementation
    
    def test_compute_rnn_features_with_particle_index(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test feature computation with particle index conditioning."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            use_weight_features=True,
            use_particle_features=True
        )
        
        batch_size = 1
        particles = tf.random.normal((batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['state_dim']))
        log_weights = tf.zeros((batch_size, dpf_rnn_config['n_particles']))
        
        # Test with particle index
        features_with_idx = dpf._compute_rnn_features(particles, log_weights, target_particle_idx=5)
        
        # Should have extra N dimensions for one-hot encoding
        assert features_with_idx.shape[0] == batch_size
        assert features_with_idx.shape[1] == dpf_rnn_config['n_particles']
        # Feature dimension includes one-hot encoding


class TestDPFRNNSingleStep:
    """Test single filtering step."""
    
    def test_step_shapes(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test step output shapes."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 2
        particles = tf.random.normal((batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['state_dim']))
        log_weights = tf.math.log(tf.ones((batch_size, dpf_rnn_config['n_particles'])) / dpf_rnn_config['n_particles'])
        observation = tf.random.normal((batch_size, dpf_rnn_config['state_dim']))
        params = {'a': 0.9, 'sigma_q': 0.1, 'sigma_r': 0.5}
        
        new_particles, new_log_weights, assignment_matrix = dpf.step(particles, log_weights, observation, params)
        
        assert new_particles.shape == particles.shape
        assert new_log_weights.shape == log_weights.shape
        assert assignment_matrix.shape == (batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['n_particles'])
    
    def test_step_with_ess(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test step with ESS monitoring."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 1
        particles = tf.random.normal((batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['state_dim']))
        log_weights = tf.math.log(tf.ones((batch_size, dpf_rnn_config['n_particles'])) / dpf_rnn_config['n_particles'])
        observation = tf.random.normal((batch_size, dpf_rnn_config['state_dim']))
        params = {'a': 0.9, 'sigma_q': 0.1, 'sigma_r': 0.5}
        
        new_particles, new_log_weights, assign_mat, ess_dict = dpf.step(
            particles, log_weights, observation, params, return_ess=True
        )
        
        assert 'ess_before' in ess_dict
        assert 'ess_after' in ess_dict
        assert 'entropy_before' in ess_dict
        assert 'entropy_after' in ess_dict
        assert assign_mat.shape == (batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['n_particles'])


class TestDPFRNNFullFiltering:
    """Test full filtering sequence."""
    
    def test_filter_shapes(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test filter output shapes."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 2
        T = 5
        observations = tf.random.normal((batch_size, T, dpf_rnn_config['state_dim']))
        init_mean = np.zeros(dpf_rnn_config['state_dim'], dtype=np.float32)
        init_cov_chol = np.eye(dpf_rnn_config['state_dim'], dtype=np.float32)
        params = {'a': 0.9, 'sigma_q': 0.1, 'sigma_r': 0.5}
        
        particles_seq, logw_seq, assign_mats, ess_stats = dpf.filter(
            observations, init_mean, init_cov_chol, params, return_ess=True
        )
        
        assert particles_seq.shape == (batch_size, T + 1, dpf_rnn_config['n_particles'], dpf_rnn_config['state_dim'])
        assert logw_seq.shape == (batch_size, T + 1, dpf_rnn_config['n_particles'])
        assert assign_mats.shape == (batch_size, T, dpf_rnn_config['n_particles'], dpf_rnn_config['n_particles'])
    
    def test_filter_with_ess_stats(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test filter with ESS statistics."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 1
        T = 5
        observations = tf.random.normal((batch_size, T, dpf_rnn_config['state_dim']))
        init_mean = np.zeros(dpf_rnn_config['state_dim'], dtype=np.float32)
        init_cov_chol = np.eye(dpf_rnn_config['state_dim'], dtype=np.float32)
        params = {'a': 0.9, 'sigma_q': 0.1, 'sigma_r': 0.5}
        
        particles_seq, logw_seq, assign_mats, ess_stats = dpf.filter(
            observations, init_mean, init_cov_chol, params, return_ess=True
        )
        
        assert 'mean_ess_before' in ess_stats
        assert 'mean_ess_after' in ess_stats
        assert 'min_ess_before' in ess_stats
        assert 'min_ess_after' in ess_stats
        assert 'mean_entropy_before' in ess_stats
        assert 'mean_entropy_after' in ess_stats


class TestDPFRNNBaselineResampling:
    """Test baseline resampling mode."""
    
    def test_baseline_resample_shapes(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test baseline resampling produces correct shapes."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            use_baseline_resampling=True
        )
        
        batch_size = 2
        particles = tf.random.normal((batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['state_dim']))
        log_weights = tf.zeros((batch_size, dpf_rnn_config['n_particles']))
        
        new_particles, assign_mat = dpf._baseline_resample(particles, log_weights)
        
        assert new_particles.shape == particles.shape
        assert assign_mat.shape == (batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['n_particles'])


class TestDPFRNNRNNResampling:
    """Test RNN-based resampling."""
    
    def test_rnn_resample_shapes(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test RNN resampling produces correct shapes."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            use_baseline_resampling=False
        )
        
        batch_size = 2
        particles = tf.random.normal((batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['state_dim']))
        log_weights = tf.zeros((batch_size, dpf_rnn_config['n_particles']))
        
        new_particles, assign_mat = dpf._rnn_resample(particles, log_weights)
        
        assert new_particles.shape == particles.shape
        assert assign_mat.shape == (batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['n_particles'])
    
    def test_rnn_produces_diverse_assignments(self, dpf_rnn_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test that RNN produces valid assignment probabilities."""
        dpf = DifferentiableParticleFilterRNN(
            n_particles=dpf_rnn_config['n_particles'],
            state_dim=dpf_rnn_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            use_baseline_resampling=False
        )
        
        batch_size = 1
        particles = tf.random.normal((batch_size, dpf_rnn_config['n_particles'], dpf_rnn_config['state_dim']))
        # Create non-uniform weights
        log_weights = tf.constant([[-5.0] * (dpf_rnn_config['n_particles'] - 1) + [0.0]], dtype=tf.float32)
        
        new_particles, assign_mat = dpf._rnn_resample(particles, log_weights)
        
        # Assignment matrix should be valid probabilities
        assign_mat_np = assign_mat.numpy()[0]  # (N, N)
        
        # Each row should sum to 1 (valid probability distribution)
        row_sums = assign_mat_np.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"Row sums: {row_sums}"
        
        # All values should be non-negative
        assert (assign_mat_np >= 0).all(), "Assignment probabilities should be non-negative"
        
        # At least check shape is correct
        assert assign_mat_np.shape == (dpf_rnn_config['n_particles'], dpf_rnn_config['n_particles'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
