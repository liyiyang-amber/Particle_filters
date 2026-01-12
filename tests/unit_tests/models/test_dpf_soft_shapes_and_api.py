"""Unit tests for DPF with soft resampling: shapes, API, and basic functionality."""

import numpy as np
import pytest
import tensorflow as tf


pytestmark = pytest.mark.tensorflow


# Import the DPF implementation
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from models.DPF_soft_resampling import DifferentiableParticleFilter


@pytest.fixture
def dpf_simple_config():
    """Simple configuration for DPF testing."""
    return {
        'n_particles': 50,
        'state_dim': 2,
        'soft_alpha': 0.2,
        'gumbel_temperature': 0.3
    }


@pytest.fixture
def simple_transition_fn():
    """Simple linear transition function for testing."""
    def transition(x_prev, params):
        """x_t = 0.9 * x_{t-1} + noise"""
        A = params.get('A', 0.9)
        sigma = params.get('sigma_q', 0.1)
        
        A = tf.convert_to_tensor(A, dtype=tf.float32)
        sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
        
        noise = tf.random.normal(tf.shape(x_prev), dtype=tf.float32) * sigma
        return A * x_prev + noise
    return transition


@pytest.fixture
def simple_log_likelihood_fn():
    """Simple Gaussian log-likelihood function."""
    def log_likelihood(x, y, params):
        """log p(y | x) for y = x + noise"""
        sigma_r = params.get('sigma_r', 0.5)
        sigma_r = tf.convert_to_tensor(sigma_r, dtype=tf.float32)
        
        # x: (B, N, d), y: (B, d)
        # Broadcast y to match x
        y = tf.expand_dims(y, axis=1)  # (B, 1, d)
        y = tf.broadcast_to(y, tf.shape(x))  # (B, N, d)
        
        diff = y - x  # (B, N, d)
        var = sigma_r ** 2
        log_norm = -0.5 * tf.math.log(2.0 * np.pi * var)
        log_lik = log_norm - 0.5 * tf.reduce_sum(diff ** 2, axis=-1) / var  # (B, N)
        return log_lik
    return log_likelihood


class TestDPFInitialization:
    """Test DPF initialization and configuration."""
    
    def test_dpf_creation(self, dpf_simple_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test basic DPF object creation."""
        dpf = DifferentiableParticleFilter(
            n_particles=dpf_simple_config['n_particles'],
            state_dim=dpf_simple_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            soft_alpha=dpf_simple_config['soft_alpha'],
            gumbel_temperature=dpf_simple_config['gumbel_temperature']
        )
        
        assert dpf.n_particles == dpf_simple_config['n_particles']
        assert dpf.state_dim == dpf_simple_config['state_dim']
        assert dpf.soft_alpha == dpf_simple_config['soft_alpha']
        assert dpf.gumbel_temperature == dpf_simple_config['gumbel_temperature']
    
    def test_dpf_with_default_parameters(self, simple_transition_fn, simple_log_likelihood_fn):
        """Test DPF with default parameters."""
        dpf = DifferentiableParticleFilter(
            n_particles=100,
            state_dim=1,
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        assert dpf.n_particles == 100
        assert dpf.state_dim == 1
        assert dpf.soft_alpha == 0.1  # default
        assert dpf.gumbel_temperature == 0.2  # default


class TestDPFParticleInitialization:
    """Test particle initialization."""
    
    def test_init_particles_shape(self, dpf_simple_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test that initialized particles have correct shapes."""
        dpf = DifferentiableParticleFilter(
            n_particles=dpf_simple_config['n_particles'],
            state_dim=dpf_simple_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 3
        init_mean = np.zeros(dpf_simple_config['state_dim'], dtype=np.float32)
        init_cov_chol = np.eye(dpf_simple_config['state_dim'], dtype=np.float32)
        
        particles, log_weights = dpf.init_particles(batch_size, init_mean, init_cov_chol)
        
        assert particles.shape == (batch_size, dpf_simple_config['n_particles'], dpf_simple_config['state_dim'])
        assert log_weights.shape == (batch_size, dpf_simple_config['n_particles'])
    
    def test_init_particles_uniform_weights(self, dpf_simple_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test that initial weights are uniform."""
        dpf = DifferentiableParticleFilter(
            n_particles=dpf_simple_config['n_particles'],
            state_dim=dpf_simple_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 2
        init_mean = np.zeros(dpf_simple_config['state_dim'], dtype=np.float32)
        init_cov_chol = np.eye(dpf_simple_config['state_dim'], dtype=np.float32)
        
        particles, log_weights = dpf.init_particles(batch_size, init_mean, init_cov_chol)
        
        # Convert log weights to weights
        weights = tf.exp(log_weights).numpy()
        
        # Check weights are uniform and sum to 1
        expected_weight = 1.0 / dpf_simple_config['n_particles']
        np.testing.assert_allclose(weights, expected_weight, rtol=1e-5)
        np.testing.assert_allclose(np.sum(weights, axis=1), 1.0, rtol=1e-5)
    
    def test_init_particles_gaussian_distribution(self, dpf_simple_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test that particles follow the specified Gaussian distribution."""
        dpf = DifferentiableParticleFilter(
            n_particles=1000,  # More particles for statistical test
            state_dim=2,
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 1
        init_mean = np.array([1.0, -0.5], dtype=np.float32)
        init_cov_chol = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        
        particles, _ = dpf.init_particles(batch_size, init_mean, init_cov_chol)
        particles_np = particles.numpy()[0]  # Remove batch dimension
        
        # Check empirical mean
        empirical_mean = np.mean(particles_np, axis=0)
        np.testing.assert_allclose(empirical_mean, init_mean, atol=0.15)
        
        # Check empirical covariance (diagonal elements)
        empirical_cov = np.cov(particles_np.T)
        expected_cov = init_cov_chol @ init_cov_chol.T
        np.testing.assert_allclose(np.diag(empirical_cov), np.diag(expected_cov), atol=0.3)


class TestDPFUtilityFunctions:
    """Test utility functions."""
    
    def test_log_normalize(self):
        """Test log normalization function."""
        log_w = tf.constant([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], dtype=tf.float32)
        log_w_norm, _ = DifferentiableParticleFilter._log_normalize(log_w, axis=-1, keepdims=False)
        
        # Check that normalized weights sum to 1
        w_norm = tf.exp(log_w_norm).numpy()
        np.testing.assert_allclose(np.sum(w_norm, axis=1), 1.0, rtol=1e-5)
    
    def test_compute_ess(self):
        """Test Effective Sample Size computation."""
        # Uniform weights: ESS should be N
        N = 100
        log_weights = tf.constant([[tf.math.log(1.0 / N).numpy()] * N], dtype=tf.float32)
        
        ess = DifferentiableParticleFilter.compute_ess(log_weights)
        np.testing.assert_allclose(ess.numpy(), N, rtol=0.01)
        
        # One dominant weight: ESS should be close to 1
        log_weights_degenerate = tf.constant([[-10.0] * (N - 1) + [0.0]], dtype=tf.float32)
        
        ess_degenerate = DifferentiableParticleFilter.compute_ess(log_weights_degenerate)
        assert ess_degenerate.numpy() < 5.0  # Much less than N
    
    def test_compute_weight_entropy(self):
        """Test weight entropy computation."""
        # Uniform weights: entropy should be log(N)
        N = 100
        log_weights = tf.constant([[tf.math.log(1.0 / N).numpy()] * N], dtype=tf.float32)
        
        entropy = DifferentiableParticleFilter.compute_weight_entropy(log_weights)
        expected_entropy = np.log(N)
        np.testing.assert_allclose(entropy.numpy(), expected_entropy, rtol=0.01)
        
        # One dominant weight: entropy should be low
        log_weights_degenerate = tf.constant([[-10.0] * (N - 1) + [0.0]], dtype=tf.float32)
        
        entropy_degenerate = DifferentiableParticleFilter.compute_weight_entropy(log_weights_degenerate)
        assert entropy_degenerate.numpy() < 1.0  # Much less than log(N)


class TestDPFSingleStep:
    """Test single filtering step."""
    
    def test_step_shapes(self, dpf_simple_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test that step produces correct output shapes."""
        dpf = DifferentiableParticleFilter(
            n_particles=dpf_simple_config['n_particles'],
            state_dim=dpf_simple_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 2
        particles = tf.random.normal((batch_size, dpf_simple_config['n_particles'], 
                                     dpf_simple_config['state_dim']))
        log_weights = tf.math.log(tf.ones((batch_size, dpf_simple_config['n_particles'])) 
                                  / dpf_simple_config['n_particles'])
        observation = tf.random.normal((batch_size, dpf_simple_config['state_dim']))
        params = {'A': 0.9, 'sigma_q': 0.1, 'sigma_r': 0.5}
        
        new_particles, new_log_weights = dpf.step(particles, log_weights, observation, params)
        
        assert new_particles.shape == particles.shape
        assert new_log_weights.shape == log_weights.shape
    
    def test_step_with_diagnostics(self, dpf_simple_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test that step returns diagnostics when requested."""
        dpf = DifferentiableParticleFilter(
            n_particles=dpf_simple_config['n_particles'],
            state_dim=dpf_simple_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 1
        particles = tf.random.normal((batch_size, dpf_simple_config['n_particles'], 
                                     dpf_simple_config['state_dim']))
        log_weights = tf.math.log(tf.ones((batch_size, dpf_simple_config['n_particles'])) 
                                  / dpf_simple_config['n_particles'])
        observation = tf.random.normal((batch_size, dpf_simple_config['state_dim']))
        params = {'A': 0.9, 'sigma_q': 0.1, 'sigma_r': 0.5}
        
        new_particles, new_log_weights, diagnostics = dpf.step(
            particles, log_weights, observation, params, return_diagnostics=True
        )
        
        assert 'ess_before' in diagnostics
        assert 'ess_after' in diagnostics
        assert 'entropy_before' in diagnostics
        assert 'entropy_after' in diagnostics
        assert 'diversity_before' in diagnostics
        assert 'diversity_after' in diagnostics


class TestDPFFullFiltering:
    """Test full filtering over sequence."""
    
    def test_filter_shapes(self, dpf_simple_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test that filter produces correct output shapes."""
        dpf = DifferentiableParticleFilter(
            n_particles=dpf_simple_config['n_particles'],
            state_dim=dpf_simple_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 2
        T = 10
        observations = tf.random.normal((batch_size, T, dpf_simple_config['state_dim']))
        init_mean = np.zeros(dpf_simple_config['state_dim'], dtype=np.float32)
        init_cov_chol = np.eye(dpf_simple_config['state_dim'], dtype=np.float32)
        params = {'A': 0.9, 'sigma_q': 0.1, 'sigma_r': 0.5}
        
        particles_seq, logw_seq = dpf.filter(observations, init_mean, init_cov_chol, params)
        
        assert particles_seq.shape == (batch_size, T + 1, dpf_simple_config['n_particles'], 
                                       dpf_simple_config['state_dim'])
        assert logw_seq.shape == (batch_size, T + 1, dpf_simple_config['n_particles'])
    
    def test_filter_with_diagnostics(self, dpf_simple_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test filter with diagnostics."""
        dpf = DifferentiableParticleFilter(
            n_particles=dpf_simple_config['n_particles'],
            state_dim=dpf_simple_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 1
        T = 5
        observations = tf.random.normal((batch_size, T, dpf_simple_config['state_dim']))
        init_mean = np.zeros(dpf_simple_config['state_dim'], dtype=np.float32)
        init_cov_chol = np.eye(dpf_simple_config['state_dim'], dtype=np.float32)
        params = {'A': 0.9, 'sigma_q': 0.1, 'sigma_r': 0.5}
        
        particles_seq, logw_seq, diagnostics = dpf.filter(
            observations, init_mean, init_cov_chol, params, return_diagnostics=True
        )
        
        # Check that diagnostics are returned
        assert isinstance(diagnostics, dict)
        assert len(diagnostics) > 0
        # Check for some aggregated metrics
        assert 'ess_before_mean' in diagnostics or 'mean_ess_before' in diagnostics or 'ess_before' in diagnostics
    
    def test_filter_empty_sequence(self, dpf_simple_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test filter with empty observation sequence."""
        dpf = DifferentiableParticleFilter(
            n_particles=dpf_simple_config['n_particles'],
            state_dim=dpf_simple_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn
        )
        
        batch_size = 1
        T = 0
        observations = tf.random.normal((batch_size, T, dpf_simple_config['state_dim']))
        init_mean = np.zeros(dpf_simple_config['state_dim'], dtype=np.float32)
        init_cov_chol = np.eye(dpf_simple_config['state_dim'], dtype=np.float32)
        params = {'A': 0.9, 'sigma_q': 0.1, 'sigma_r': 0.5}
        
        particles_seq, logw_seq = dpf.filter(observations, init_mean, init_cov_chol, params)
        
        # Should only have initial particles
        assert particles_seq.shape == (batch_size, 1, dpf_simple_config['n_particles'], 
                                       dpf_simple_config['state_dim'])


class TestDPFGumbelSoftmax:
    """Test Gumbel-Softmax sampling."""
    
    def test_gumbel_sample_shape(self):
        """Test Gumbel noise generation."""
        shape = (10, 5)
        gumbel = DifferentiableParticleFilter._sample_gumbel(shape)
        assert gumbel.shape == shape
    
    def test_gumbel_softmax_temperature(self, dpf_simple_config, simple_transition_fn, simple_log_likelihood_fn):
        """Test that temperature affects softmax sharpness."""
        # Make the test deterministic: Gumbel-Softmax includes random Gumbel noise.
        tf.random.set_seed(0)

        dpf = DifferentiableParticleFilter(
            n_particles=dpf_simple_config['n_particles'],
            state_dim=dpf_simple_config['state_dim'],
            transition_fn=simple_transition_fn,
            log_likelihood_fn=simple_log_likelihood_fn,
            gumbel_temperature=0.1  # Low temperature = sharper
        )
        
        log_probs = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        
        # Lower temperature should produce sharper distribution
        soft_samples_low = dpf._gumbel_softmax(log_probs, temperature=0.1)
        soft_samples_high = dpf._gumbel_softmax(log_probs, temperature=1.0)
        
        # Entropy of low-temp should be lower
        entropy_low = -tf.reduce_sum(soft_samples_low * tf.math.log(soft_samples_low + 1e-10))
        entropy_high = -tf.reduce_sum(soft_samples_high * tf.math.log(soft_samples_high + 1e-10))
        
        assert entropy_low < entropy_high


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
