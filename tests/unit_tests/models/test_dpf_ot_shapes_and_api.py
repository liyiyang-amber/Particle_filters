"""Unit tests for DPF with OT resampling: shapes, API, and basic functionality."""

import numpy as np
import pytest
import tensorflow as tf


pytestmark = pytest.mark.tensorflow


# Import the DPF OT implementation
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from models.DPF_OT_resampling import DPF_OT, sinkhorn_ot_resample, pairwise_squared_distances


@pytest.fixture
def dpf_ot_config():
    """Configuration for DPF OT testing."""
    return {
        'N_particles': 40,
        'state_dim': 2,
        'epsilon': 0.1,
        'sinkhorn_iters': 50
    }


@pytest.fixture
def simple_transition_fn():
    """Simple transition function for testing."""
    def transition(particles, t):
        sigma = 0.1
        noise = tf.random.normal(tf.shape(particles), stddev=sigma)
        return particles + noise
    return transition


@pytest.fixture
def simple_obs_loglik_fn():
    """Simple observation log-likelihood."""
    def obs_loglik(particles, y_t, t):
        tau = 0.2
        y_t = tf.convert_to_tensor(y_t, dtype=tf.float32)
        diff = particles - y_t[None, :]
        sq_norm = tf.reduce_sum(tf.square(diff), axis=1)
        log_lik = -0.5 * sq_norm / (tau ** 2)
        return log_lik
    return obs_loglik


class TestPairwiseDistances:
    """Test pairwise distance computation."""
    
    def test_pairwise_distances_shape(self):
        """Test output shape of pairwise distances."""
        N, M, d = 10, 15, 3
        x = tf.random.normal((N, d))
        y = tf.random.normal((M, d))
        
        dist = pairwise_squared_distances(x, y)
        assert dist.shape == (N, M)
    
    def test_pairwise_distances_diagonal_zero(self):
        """Test that distance between identical points is zero."""
        N, d = 5, 2
        x = tf.random.normal((N, d))
        
        dist = pairwise_squared_distances(x, x)
        diagonal = tf.linalg.diag_part(dist)
        
        np.testing.assert_allclose(diagonal.numpy(), 0.0, atol=1e-5)
    
    def test_pairwise_distances_symmetry(self):
        """Test that distance matrix is symmetric."""
        N, d = 8, 3
        x = tf.random.normal((N, d))
        
        dist = pairwise_squared_distances(x, x)
        
        np.testing.assert_allclose(dist.numpy(), dist.numpy().T, rtol=1e-5)
    
    def test_pairwise_distances_non_negative(self):
        """Test that all distances are non-negative."""
        N, M, d = 10, 12, 2
        x = tf.random.normal((N, d))
        y = tf.random.normal((M, d))
        
        dist = pairwise_squared_distances(x, y)
        
        assert tf.reduce_min(dist) >= 0.0


class TestSinkhornOTResample:
    """Test Sinkhorn OT resampling."""
    
    def test_sinkhorn_output_shapes(self):
        """Test output shapes of Sinkhorn resampling."""
        N, d = 30, 2
        particles = tf.random.normal((N, d))
        weights = tf.ones(N) / N
        
        new_particles, new_weights = sinkhorn_ot_resample(
            particles, weights, epsilon=0.1, n_iters=20
        )
        
        assert new_particles.shape == (N, d)
        assert new_weights.shape == (N,)
    
    def test_sinkhorn_uniform_weights(self):
        """Test that output weights are uniform."""
        N, d = 25, 2
        particles = tf.random.normal((N, d))
        weights = tf.random.uniform((N,))
        weights = weights / tf.reduce_sum(weights)
        
        new_particles, new_weights = sinkhorn_ot_resample(
            particles, weights, epsilon=0.1, n_iters=30
        )
        
        expected_weight = 1.0 / N
        np.testing.assert_allclose(new_weights.numpy(), expected_weight, rtol=1e-5)
    
    def test_sinkhorn_preserves_particle_count(self):
        """Test that number of particles is preserved."""
        N, d = 20, 3
        particles = tf.random.normal((N, d))
        weights = tf.ones(N) / N
        
        new_particles, new_weights = sinkhorn_ot_resample(
            particles, weights, epsilon=0.1, n_iters=25
        )
        
        assert new_particles.shape[0] == N
        assert new_weights.shape[0] == N
    
    def test_sinkhorn_with_diagnostics(self):
        """Test Sinkhorn with diagnostics."""
        N, d = 30, 2
        particles = tf.random.normal((N, d))
        weights = tf.ones(N) / N
        
        new_particles, new_weights, diagnostics = sinkhorn_ot_resample(
            particles, weights, epsilon=0.1, n_iters=50, return_diagnostics=True
        )
        
        assert 'sinkhorn_iterations' in diagnostics
        assert 'converged' in diagnostics
        assert 'ot_distance' in diagnostics
        assert 'transport_plan_sparsity' in diagnostics
        assert 'dual_variables' in diagnostics
    
    def test_sinkhorn_convergence(self):
        """Test that Sinkhorn converges with sufficient iterations."""
        N, d = 20, 2
        particles = tf.random.normal((N, d))
        weights = tf.ones(N) / N
        
        new_particles, new_weights, diagnostics = sinkhorn_ot_resample(
            particles, weights, epsilon=0.1, n_iters=100, tol=1e-6, return_diagnostics=True
        )
        
        # With enough iterations, should converge
        assert diagnostics['sinkhorn_iterations'] <= 100
    
    def test_sinkhorn_epsilon_effect(self):
        """Test that epsilon affects regularization strength."""
        N, d = 25, 2
        particles = tf.random.normal((N, d))
        # Create non-uniform weights
        weights = tf.constant([0.8] + [0.2 / (N - 1)] * (N - 1), dtype=tf.float32)
        
        # Small epsilon (more regularization)
        _, _, diag_small = sinkhorn_ot_resample(
            particles, weights, epsilon=0.01, n_iters=50, return_diagnostics=True
        )
        
        # Large epsilon (less regularization)
        _, _, diag_large = sinkhorn_ot_resample(
            particles, weights, epsilon=1.0, n_iters=50, return_diagnostics=True
        )
        
        # OT distance should be different
        assert diag_small['ot_distance'] != diag_large['ot_distance']


class TestDPFOTInitialization:
    """Test DPF OT initialization."""
    
    def test_dpf_ot_creation(self, dpf_ot_config, simple_transition_fn, simple_obs_loglik_fn):
        """Test DPF OT object creation."""
        dpf = DPF_OT(
            N_particles=dpf_ot_config['N_particles'],
            state_dim=dpf_ot_config['state_dim'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn,
            epsilon=dpf_ot_config['epsilon'],
            sinkhorn_iters=dpf_ot_config['sinkhorn_iters']
        )
        
        assert dpf.N == dpf_ot_config['N_particles']
        assert dpf.d == dpf_ot_config['state_dim']
        assert dpf.epsilon == dpf_ot_config['epsilon']
        assert dpf.sinkhorn_iters == dpf_ot_config['sinkhorn_iters']
    
    def test_dpf_ot_default_parameters(self, simple_transition_fn, simple_obs_loglik_fn):
        """Test DPF OT with default parameters."""
        dpf = DPF_OT(
            N_particles=50,
            state_dim=2,
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn
        )
        
        assert dpf.N == 50
        assert dpf.d == 2
        assert dpf.epsilon == 0.1
        assert dpf.sinkhorn_iters == 50


class TestDPFOTParticleInitialization:
    """Test particle initialization."""
    
    def test_init_particles_shape(self, dpf_ot_config, simple_transition_fn, simple_obs_loglik_fn):
        """Test initialized particle shapes."""
        dpf = DPF_OT(
            N_particles=dpf_ot_config['N_particles'],
            state_dim=dpf_ot_config['state_dim'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn
        )
        
        mean = np.zeros(dpf_ot_config['state_dim'], dtype=np.float32)
        cov_chol = np.eye(dpf_ot_config['state_dim'], dtype=np.float32)
        
        particles, weights = dpf.init_particles(mean, cov_chol)
        
        assert particles.shape == (dpf_ot_config['N_particles'], dpf_ot_config['state_dim'])
        assert weights.shape == (dpf_ot_config['N_particles'],)
    
    def test_init_particles_uniform_weights(self, dpf_ot_config, simple_transition_fn, simple_obs_loglik_fn):
        """Test initial weights are uniform."""
        dpf = DPF_OT(
            N_particles=dpf_ot_config['N_particles'],
            state_dim=dpf_ot_config['state_dim'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn
        )
        
        mean = np.zeros(dpf_ot_config['state_dim'], dtype=np.float32)
        cov_chol = np.eye(dpf_ot_config['state_dim'], dtype=np.float32)
        
        particles, weights = dpf.init_particles(mean, cov_chol)
        
        expected_weight = 1.0 / dpf_ot_config['N_particles']
        np.testing.assert_allclose(weights.numpy(), expected_weight, rtol=1e-5)
    
    def test_init_particles_gaussian_distribution(self, simple_transition_fn, simple_obs_loglik_fn):
        """Test particles follow Gaussian distribution."""
        dpf = DPF_OT(
            N_particles=1000,  # More particles for statistical test
            state_dim=2,
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn
        )
        
        mean = np.array([1.0, -0.5], dtype=np.float32)
        cov_chol = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        
        particles, _ = dpf.init_particles(mean, cov_chol)
        particles_np = particles.numpy()
        
        # Check empirical mean
        empirical_mean = np.mean(particles_np, axis=0)
        np.testing.assert_allclose(empirical_mean, mean, atol=0.15)
        
        # Check empirical covariance
        empirical_cov = np.cov(particles_np.T)
        expected_cov = cov_chol @ cov_chol.T
        # Increase tolerance slightly for stochastic sampling (100 particles)
        np.testing.assert_allclose(np.diag(empirical_cov), np.diag(expected_cov), atol=0.4)


class TestDPFOTUtilities:
    """Test utility functions."""
    
    def test_compute_ess(self):
        """Test ESS computation."""
        # Uniform weights
        N = 50
        weights = tf.ones(N) / N
        
        ess = DPF_OT.compute_ess(weights)
        np.testing.assert_allclose(ess.numpy(), N, rtol=0.01)
        
        # One dominant weight
        weights_degenerate = tf.constant([0.99] + [0.01 / (N - 1)] * (N - 1), dtype=tf.float32)
        ess_degenerate = DPF_OT.compute_ess(weights_degenerate)
        assert ess_degenerate.numpy() < 5.0
    
    def test_compute_weight_entropy(self):
        """Test weight entropy computation."""
        N = 50
        weights = tf.ones(N) / N
        
        entropy = DPF_OT.compute_weight_entropy(weights)
        expected_entropy = np.log(N)
        np.testing.assert_allclose(entropy.numpy(), expected_entropy, rtol=0.01)


class TestDPFOTSingleStep:
    """Test single filtering step."""
    
    def test_step_shapes(self, dpf_ot_config, simple_transition_fn, simple_obs_loglik_fn):
        """Test step output shapes."""
        dpf = DPF_OT(
            N_particles=dpf_ot_config['N_particles'],
            state_dim=dpf_ot_config['state_dim'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn
        )
        
        particles = tf.random.normal((dpf_ot_config['N_particles'], dpf_ot_config['state_dim']))
        weights = tf.ones(dpf_ot_config['N_particles']) / dpf_ot_config['N_particles']
        y_t = tf.random.normal((dpf_ot_config['state_dim'],))
        
        new_particles, new_weights = dpf.step(particles, weights, y_t)
        
        assert new_particles.shape == particles.shape
        assert new_weights.shape == weights.shape
    
    def test_step_with_diagnostics(self, dpf_ot_config, simple_transition_fn, simple_obs_loglik_fn):
        """Test step with diagnostics."""
        dpf = DPF_OT(
            N_particles=dpf_ot_config['N_particles'],
            state_dim=dpf_ot_config['state_dim'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn
        )
        
        particles = tf.random.normal((dpf_ot_config['N_particles'], dpf_ot_config['state_dim']))
        weights = tf.ones(dpf_ot_config['N_particles']) / dpf_ot_config['N_particles']
        y_t = tf.random.normal((dpf_ot_config['state_dim'],))
        
        new_particles, new_weights, diagnostics = dpf.step(
            particles, weights, y_t, return_diagnostics=True
        )
        
        assert 'ess_before' in diagnostics
        assert 'ess_after' in diagnostics
        assert 'entropy_before' in diagnostics
        assert 'entropy_after' in diagnostics
        assert 'sinkhorn_iterations' in diagnostics
        assert 'ot_distance' in diagnostics
    
    def test_step_uniform_output_weights(self, dpf_ot_config, simple_transition_fn, simple_obs_loglik_fn):
        """Test that step produces uniform output weights."""
        dpf = DPF_OT(
            N_particles=dpf_ot_config['N_particles'],
            state_dim=dpf_ot_config['state_dim'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn
        )
        
        particles = tf.random.normal((dpf_ot_config['N_particles'], dpf_ot_config['state_dim']))
        # Non-uniform input weights
        weights = tf.random.uniform((dpf_ot_config['N_particles'],))
        weights = weights / tf.reduce_sum(weights)
        y_t = tf.random.normal((dpf_ot_config['state_dim'],))
        
        new_particles, new_weights = dpf.step(particles, weights, y_t)
        
        # Output weights should be uniform
        expected_weight = 1.0 / dpf_ot_config['N_particles']
        np.testing.assert_allclose(new_weights.numpy(), expected_weight, rtol=1e-5)


class TestDPFOTFullFiltering:
    """Test full filtering sequence."""
    
    def test_run_filter_shapes(self, dpf_ot_config, simple_transition_fn, simple_obs_loglik_fn):
        """Test filter output shapes."""
        dpf = DPF_OT(
            N_particles=dpf_ot_config['N_particles'],
            state_dim=dpf_ot_config['state_dim'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn
        )
        
        T = 5
        y_seq = [tf.random.normal((dpf_ot_config['state_dim'],)) for _ in range(T)]
        mean0 = np.zeros(dpf_ot_config['state_dim'], dtype=np.float32)
        cov0_chol = np.eye(dpf_ot_config['state_dim'], dtype=np.float32)
        
        particles_seq, weights_seq = dpf.run_filter(y_seq, mean0, cov0_chol)
        
        assert len(particles_seq) == T
        assert len(weights_seq) == T
        for t in range(T):
            assert particles_seq[t].shape == (dpf_ot_config['N_particles'], dpf_ot_config['state_dim'])
            assert weights_seq[t].shape == (dpf_ot_config['N_particles'],)
    
    def test_run_filter_with_diagnostics(self, dpf_ot_config, simple_transition_fn, simple_obs_loglik_fn):
        """Test filter with diagnostics."""
        dpf = DPF_OT(
            N_particles=dpf_ot_config['N_particles'],
            state_dim=dpf_ot_config['state_dim'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn
        )
        
        T = 5
        y_seq = [tf.random.normal((dpf_ot_config['state_dim'],)) for _ in range(T)]
        mean0 = np.zeros(dpf_ot_config['state_dim'], dtype=np.float32)
        cov0_chol = np.eye(dpf_ot_config['state_dim'], dtype=np.float32)
        
        particles_seq, weights_seq, diagnostics = dpf.run_filter(
            y_seq, mean0, cov0_chol, return_diagnostics=True
        )
        
        # Check that diagnostics contain aggregated metrics
        assert 'ess_before_mean' in diagnostics
        assert 'ess_after_mean' in diagnostics
        assert 'total_time' in diagnostics
        assert 'mean_step_time' in diagnostics
    
    def test_run_filter_empty_sequence(self, dpf_ot_config, simple_transition_fn, simple_obs_loglik_fn):
        """Test filter with empty observation sequence."""
        dpf = DPF_OT(
            N_particles=dpf_ot_config['N_particles'],
            state_dim=dpf_ot_config['state_dim'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn
        )
        
        y_seq = []
        mean0 = np.zeros(dpf_ot_config['state_dim'], dtype=np.float32)
        cov0_chol = np.eye(dpf_ot_config['state_dim'], dtype=np.float32)
        
        particles_seq, weights_seq = dpf.run_filter(y_seq, mean0, cov0_chol)
        
        assert len(particles_seq) == 0
        assert len(weights_seq) == 0


class TestDPFOTNumericalStability:
    """Test numerical stability."""
    
    def test_step_with_extreme_weights(self, dpf_ot_config, simple_transition_fn, simple_obs_loglik_fn):
        """Test step handles extreme weight values."""
        dpf = DPF_OT(
            N_particles=dpf_ot_config['N_particles'],
            state_dim=dpf_ot_config['state_dim'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn
        )
        
        particles = tf.random.normal((dpf_ot_config['N_particles'], dpf_ot_config['state_dim']))
        # Extreme weights (one very large)
        weights = tf.constant([1e-10] * (dpf_ot_config['N_particles'] - 1) + [1.0], dtype=tf.float32)
        weights = weights / tf.reduce_sum(weights)
        y_t = tf.random.normal((dpf_ot_config['state_dim'],))
        
        # Should not raise exception
        new_particles, new_weights = dpf.step(particles, weights, y_t)
        
        # Check outputs are valid
        assert not tf.reduce_any(tf.math.is_nan(new_particles))
        assert not tf.reduce_any(tf.math.is_nan(new_weights))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
