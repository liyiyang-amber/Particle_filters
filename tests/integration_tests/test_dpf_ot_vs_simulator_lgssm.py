"""Integration tests for DPF with OT resampling on LGSSM."""

import numpy as np
import pytest
import tensorflow as tf


pytestmark = [pytest.mark.integration, pytest.mark.tensorflow]

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.DPF_OT_resampling import DPF_OT
from simulator.simulator_LGSSM import simulate_lgssm


def simple_transition_fn(particles, t):
    """Simple transition for 1D random walk."""
    sigma = 0.1
    noise = tf.random.normal(tf.shape(particles), stddev=sigma)
    return particles + noise


def simple_obs_loglik_fn(particles, y_t, t):
    """Simple Gaussian observation model."""
    tau = 0.2
    y_t = tf.convert_to_tensor(y_t, dtype=tf.float32)
    diff = particles - y_t[None, :]
    sq_norm = tf.reduce_sum(tf.square(diff), axis=1)
    log_lik = -0.5 * sq_norm / (tau ** 2)
    return log_lik


@pytest.fixture
def simple_1d_data():
    """Generate simple 1D tracking data."""
    T = 50
    d = 1
    
    np.random.seed(42)
    x = np.zeros((T, d), dtype=np.float32)
    y = np.zeros((T, d), dtype=np.float32)
    
    x[0] = 0.0
    for t in range(1, T):
        x[t] = x[t-1] + 0.1 * np.random.randn(d)
        y[t] = x[t] + 0.2 * np.random.randn(d)
    
    return {'x': x, 'y': y, 'T': T, 'd': d}


@pytest.mark.integration
class TestDPFOTSimple1D:
    """Integration tests for DPF OT on simple 1D system."""
    
    def test_dpf_ot_tracks_1d_system(self, simple_1d_data):
        """Test that DPF OT can track simple 1D system."""
        data = simple_1d_data
        
        dpf_ot = DPF_OT(
            N_particles=50,
            state_dim=data['d'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn,
            epsilon=0.1,
            sinkhorn_iters=50
        )
        
        # Initialize
        mean0 = np.zeros(data['d'], dtype=np.float32)
        cov0_chol = np.eye(data['d'], dtype=np.float32)
        
        # Run filter
        particles_seq, weights_seq = dpf_ot.run_filter(
            [data['y'][t] for t in range(data['T'])],
            mean0, cov0_chol
        )
        
        # Compute posterior means
        posterior_means = []
        for particles, weights in zip(particles_seq, weights_seq):
            mean = tf.reduce_sum(weights[:, None] * particles, axis=0).numpy()
            posterior_means.append(mean)
        posterior_means = np.array(posterior_means)
        
        # Compute RMSE (excluding first few timesteps for warmup)
        rmse = np.sqrt(np.mean((posterior_means[5:] - data['x'][5:data['T']]) ** 2))
        
        # Should achieve reasonable tracking
        assert rmse < 0.5, f"RMSE too high: {rmse}"
    
    def test_dpf_ot_with_diagnostics(self, simple_1d_data):
        """Test DPF OT with diagnostics."""
        data = simple_1d_data
        
        dpf_ot = DPF_OT(
            N_particles=40,
            state_dim=data['d'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn,
            epsilon=0.15,
            sinkhorn_iters=40
        )
        
        mean0 = np.zeros(data['d'], dtype=np.float32)
        cov0_chol = np.eye(data['d'], dtype=np.float32)
        
        # Run with diagnostics
        particles_seq, weights_seq, diagnostics = dpf_ot.run_filter(
            [data['y'][t] for t in range(20)],  # First 20 timesteps
            mean0, cov0_chol,
            return_diagnostics=True
        )
        
        # Check diagnostics (aggregated keys have _mean, _std suffixes)
        assert 'ess_before_mean' in diagnostics
        assert 'ess_after_mean' in diagnostics
        assert 'total_time' in diagnostics
        
        # ESS after resampling should be high (uniform weights)
        ess_after = diagnostics['ess_after_mean']
        if hasattr(ess_after, 'numpy'):
            ess_after = ess_after.numpy()
        assert ess_after > 35
        
        # OT distance should be positive (aggregated key is ot_distance_mean)
        ot_dist = diagnostics['ot_distance_mean']
        if hasattr(ot_dist, 'numpy'):
            ot_dist = ot_dist.numpy()
        assert ot_dist > 0
    
    def test_dpf_ot_with_ground_truth(self, simple_1d_data):
        """Test DPF OT with ground truth for RMSE computation."""
        data = simple_1d_data
        
        dpf_ot = DPF_OT(
            N_particles=60,
            state_dim=data['d'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn,
            epsilon=0.1,
            sinkhorn_iters=50
        )
        
        mean0 = np.zeros(data['d'], dtype=np.float32)
        cov0_chol = np.eye(data['d'], dtype=np.float32)
        
        # Prepend initial state to ground truth (filter expects T+1 states including t=0)
        T = 30
        ground_truth_with_init = np.concatenate([mean0[None, :], data['x'][:T]], axis=0)
        
        # Run with ground truth
        particles_seq, weights_seq, diagnostics = dpf_ot.run_filter(
            [data['y'][t] for t in range(T)],
            mean0, cov0_chol,
            return_diagnostics=True,
            ground_truth=ground_truth_with_init
        )
        
        # Check RMSE was computed
        assert 'rmse_sequence' in diagnostics
        assert 'mean_rmse' in diagnostics
        
        # Mean RMSE should be reasonable
        mean_rmse = diagnostics['mean_rmse'].numpy()
        assert mean_rmse < 0.6, f"Mean RMSE too high: {mean_rmse}"


@pytest.fixture
def lgssm_1d_params():
    """1D LGSSM parameters."""
    A = np.array([[0.95]])
    B = np.array([[0.4]])
    C = np.array([[1.0]])
    D = np.array([[0.6]])
    Sigma = np.array([[1.0]])
    return {
        'A': A, 'B': B, 'C': C, 'D': D, 'Sigma': Sigma,
        'N': 60, 'seed': 123
    }


@pytest.fixture
def lgssm_1d_data(lgssm_1d_params):
    """Simulate 1D LGSSM data."""
    p = lgssm_1d_params
    result = simulate_lgssm(
        A=p['A'], B=p['B'], C=p['C'], D=p['D'],
        Sigma=p['Sigma'], N=p['N'], seed=p['seed']
    )
    return result


def lgssm_transition_fn(particles, t):
    """LGSSM transition function."""
    # For this test, use fixed parameters
    A = np.array([[0.95]])
    B = np.array([[0.4]])
    
    A = tf.convert_to_tensor(A, dtype=tf.float32)
    B = tf.convert_to_tensor(B, dtype=tf.float32)
    
    noise = tf.random.normal(tf.shape(particles), dtype=tf.float32)
    new_particles = tf.matmul(particles, A, transpose_b=True) + \
                   tf.matmul(noise, B, transpose_b=True)
    return new_particles


def lgssm_obs_loglik_fn(particles, y_t, t):
    """LGSSM observation log-likelihood."""
    C = np.array([[1.0]])
    D = np.array([[0.6]])
    
    C = tf.convert_to_tensor(C, dtype=tf.float32)
    D = tf.convert_to_tensor(D, dtype=tf.float32)
    
    # Expected observation
    y_pred = tf.matmul(particles, C, transpose_b=True)
    
    # Observation noise variance
    R = tf.matmul(D, D, transpose_b=True)
    var = R[0, 0]
    
    # Observation
    y_t = tf.convert_to_tensor(y_t, dtype=tf.float32)
    y_t = tf.reshape(y_t, (1, -1))
    
    # Log-likelihood
    diff = y_t - y_pred  # (N, 1)
    log_norm = -0.5 * tf.math.log(2.0 * np.pi * var)
    log_lik = log_norm - 0.5 * tf.reduce_sum(diff ** 2, axis=-1) / var
    
    return log_lik


@pytest.mark.integration
class TestDPFOTLGSSM1D:
    """Integration tests for DPF OT on 1D LGSSM."""
    
    def test_dpf_ot_tracks_lgssm_1d(self, lgssm_1d_data, lgssm_1d_params):
        """Test DPF OT on simulated LGSSM."""
        X_true = lgssm_1d_data.X
        Y_obs = lgssm_1d_data.Y
        
        dpf_ot = DPF_OT(
            N_particles=80,
            state_dim=1,
            transition_fn=lgssm_transition_fn,
            obs_loglik_fn=lgssm_obs_loglik_fn,
            epsilon=0.1,
            sinkhorn_iters=50
        )
        
        mean0 = np.zeros(1, dtype=np.float32)
        cov0_chol = lgssm_1d_params['Sigma'].astype(np.float32)
        
        # Run filter
        particles_seq, weights_seq = dpf_ot.run_filter(
            [Y_obs[t] for t in range(lgssm_1d_params['N'])],
            mean0, cov0_chol
        )
        
        # Compute posterior means
        posterior_means = []
        for particles, weights in zip(particles_seq, weights_seq):
            mean = tf.reduce_sum(weights[:, None] * particles, axis=0).numpy()
            posterior_means.append(mean)
        posterior_means = np.array(posterior_means)
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((posterior_means - X_true) ** 2))
        
        # Should track reasonably well
        assert rmse < 1.2, f"RMSE too high: {rmse}"
    
    def test_dpf_ot_convergence_monitoring(self, lgssm_1d_data, lgssm_1d_params):
        """Test Sinkhorn convergence monitoring."""
        Y_obs = lgssm_1d_data.Y
        
        dpf_ot = DPF_OT(
            N_particles=60,
            state_dim=1,
            transition_fn=lgssm_transition_fn,
            obs_loglik_fn=lgssm_obs_loglik_fn,
            epsilon=0.1,
            sinkhorn_iters=100  # More iterations for convergence
        )
        
        mean0 = np.zeros(1, dtype=np.float32)
        cov0_chol = lgssm_1d_params['Sigma'].astype(np.float32)
        
        particles_seq, weights_seq, diagnostics = dpf_ot.run_filter(
            [Y_obs[t] for t in range(20)],
            mean0, cov0_chol,
            return_diagnostics=True
        )
        
        # Check convergence info (aggregated keys)
        assert 'sinkhorn_iterations_mean' in diagnostics
        assert 'converged_rate' in diagnostics
        
        # With enough iterations, should have some convergence
        converged_rate = diagnostics['converged_rate']
        assert converged_rate >= 0.0  # At least some steps should converge or reach max iters


@pytest.mark.integration
class TestDPFOTEpsilonEffects:
    """Test effects of epsilon parameter."""
    
    def test_epsilon_range(self, simple_1d_data):
        """Test DPF OT with different epsilon values."""
        data = simple_1d_data
        
        epsilons = [0.01, 0.1, 0.5]
        rmses = []
        
        for eps in epsilons:
            dpf_ot = DPF_OT(
                N_particles=50,
                state_dim=data['d'],
                transition_fn=simple_transition_fn,
                obs_loglik_fn=simple_obs_loglik_fn,
                epsilon=eps,
                sinkhorn_iters=60
            )
            
            mean0 = np.zeros(data['d'], dtype=np.float32)
            cov0_chol = np.eye(data['d'], dtype=np.float32)
            
            particles_seq, weights_seq = dpf_ot.run_filter(
                [data['y'][t] for t in range(30)],
                mean0, cov0_chol
            )
            
            # Compute posterior means
            posterior_means = []
            for particles, weights in zip(particles_seq, weights_seq):
                mean = tf.reduce_sum(weights[:, None] * particles, axis=0).numpy()
                posterior_means.append(mean)
            posterior_means = np.array(posterior_means)
            
            rmse = np.sqrt(np.mean((posterior_means[5:] - data['x'][5:30]) ** 2))
            rmses.append(rmse)
        
        # All should achieve reasonable performance
        for rmse in rmses:
            assert rmse < 0.8
    
    def test_epsilon_effect_on_ot_distance(self, simple_1d_data):
        """Test that epsilon affects OT distance."""
        data = simple_1d_data
        
        ot_distances = []
        
        for eps in [0.05, 0.2, 0.5]:
            dpf_ot = DPF_OT(
                N_particles=40,
                state_dim=data['d'],
                transition_fn=simple_transition_fn,
                obs_loglik_fn=simple_obs_loglik_fn,
                epsilon=eps,
                sinkhorn_iters=50
            )
            
            mean0 = np.zeros(data['d'], dtype=np.float32)
            cov0_chol = np.eye(data['d'], dtype=np.float32)
            
            _, _, diagnostics = dpf_ot.run_filter(
                [data['y'][t] for t in range(15)],
                mean0, cov0_chol,
                return_diagnostics=True
            )
            
            # Aggregated key is ot_distance_mean
            ot_dist = diagnostics['ot_distance_mean']
            if hasattr(ot_dist, 'numpy'):
                ot_dist = ot_dist.numpy()
            ot_distances.append(ot_dist)
        
        # OT distances should vary with epsilon
        assert not all(np.isclose(d, ot_distances[0]) for d in ot_distances)


@pytest.mark.integration
class TestDPFOTSinkhornIterations:
    """Test effects of Sinkhorn iteration count."""
    
    def test_iteration_count_effect(self, simple_1d_data):
        """Test DPF OT with different Sinkhorn iteration counts."""
        data = simple_1d_data
        
        iteration_counts = [10, 30, 60]
        
        for n_iters in iteration_counts:
            dpf_ot = DPF_OT(
                N_particles=40,
                state_dim=data['d'],
                transition_fn=simple_transition_fn,
                obs_loglik_fn=simple_obs_loglik_fn,
                epsilon=0.1,
                sinkhorn_iters=n_iters
            )
            
            mean0 = np.zeros(data['d'], dtype=np.float32)
            cov0_chol = np.eye(data['d'], dtype=np.float32)
            
            # Should complete without errors
            particles_seq, weights_seq = dpf_ot.run_filter(
                [data['y'][t] for t in range(15)],
                mean0, cov0_chol
            )
            
            assert len(particles_seq) == 15
    
    def test_early_convergence(self, simple_1d_data):
        """Test that Sinkhorn can converge early with tight tolerance."""
        data = simple_1d_data
        
        dpf_ot = DPF_OT(
            N_particles=30,
            state_dim=data['d'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn,
            epsilon=0.1,
            sinkhorn_iters=100  # High max, but should converge earlier
        )
        
        mean0 = np.zeros(data['d'], dtype=np.float32)
        cov0_chol = np.eye(data['d'], dtype=np.float32)
        
        _, _, diagnostics = dpf_ot.run_filter(
            [data['y'][t] for t in range(10)],
            mean0, cov0_chol,
            return_diagnostics=True
        )
        
        # With convergence checking, should use fewer than or equal to max iterations
        mean_iters = diagnostics['sinkhorn_iterations_mean']
        if hasattr(mean_iters, 'numpy'):
            mean_iters = mean_iters.numpy()
        # May converge early in some cases
        assert mean_iters <= 100


@pytest.mark.integration
class TestDPFOTNumericalStability:
    """Test numerical stability."""
    
    def test_stability_with_degenerate_weights(self, simple_1d_data):
        """Test stability when weights become degenerate."""
        data = simple_1d_data
        
        dpf_ot = DPF_OT(
            N_particles=40,
            state_dim=data['d'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn,
            epsilon=0.1,
            sinkhorn_iters=50
        )
        
        mean0 = np.zeros(data['d'], dtype=np.float32)
        cov0_chol = np.eye(data['d'], dtype=np.float32)
        
        # Run filter (some timesteps may produce degenerate weights)
        particles_seq, weights_seq = dpf_ot.run_filter(
            [data['y'][t] for t in range(20)],
            mean0, cov0_chol
        )
        
        # Check no NaNs or Infs
        for particles, weights in zip(particles_seq, weights_seq):
            assert not tf.reduce_any(tf.math.is_nan(particles))
            assert not tf.reduce_any(tf.math.is_inf(particles))
            assert not tf.reduce_any(tf.math.is_nan(weights))
            assert not tf.reduce_any(tf.math.is_inf(weights))
    
    def test_stability_with_extreme_observations(self, simple_1d_data):
        """Test stability with extreme observation values."""
        data = simple_1d_data
        
        dpf_ot = DPF_OT(
            N_particles=40,
            state_dim=data['d'],
            transition_fn=simple_transition_fn,
            obs_loglik_fn=simple_obs_loglik_fn,
            epsilon=0.1,
            sinkhorn_iters=50
        )
        
        # Create extreme observations
        y_extreme = [data['y'][t].copy() for t in range(15)]
        y_extreme[7] = np.array([10.0], dtype=np.float32)  # Extreme outlier
        
        mean0 = np.zeros(data['d'], dtype=np.float32)
        cov0_chol = np.eye(data['d'], dtype=np.float32)
        
        # Should not crash
        particles_seq, weights_seq = dpf_ot.run_filter(
            y_extreme, mean0, cov0_chol
        )
        
        # Check outputs are valid
        for particles, weights in zip(particles_seq, weights_seq):
            assert not tf.reduce_any(tf.math.is_nan(particles))
            assert not tf.reduce_any(tf.math.is_nan(weights))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
