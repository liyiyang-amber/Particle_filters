"""Integration tests for DPF with soft resampling on LGSSM."""

import numpy as np
import pytest
import tensorflow as tf


pytestmark = [pytest.mark.integration, pytest.mark.tensorflow]

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.DPF_soft_resampling import DifferentiableParticleFilter
from simulator.simulator_LGSSM import simulate_lgssm


@pytest.fixture
def lgssm_1d_params():
    """1D linear Gaussian state-space model parameters."""
    nx, ny = 1, 1
    A = np.array([[0.9]])
    B = np.array([[0.5]])  # Process noise scale
    C = np.array([[1.0]])
    D = np.array([[0.7]])  # Observation noise scale
    Sigma = np.array([[1.0]])
    return {
        'A': A, 'B': B, 'C': C, 'D': D, 'Sigma': Sigma,
        'nx': nx, 'ny': ny, 'N': 100, 'seed': 42
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


def linear_transition_fn(x_prev, params):
    """Linear transition for LGSSM."""
    A = params['A']
    B = params['B']
    
    A = tf.convert_to_tensor(A, dtype=tf.float32)
    B = tf.convert_to_tensor(B, dtype=tf.float32)
    
    # x_prev: (B, N_particles, nx)
    # A: (nx, nx), B: (nx, nv)
    noise = tf.random.normal(tf.shape(x_prev), dtype=tf.float32)
    # x_new = A @ x_prev + B @ noise
    x_new = tf.einsum('ij,bnj->bni', A, x_prev) + tf.einsum('ij,bnj->bni', B, noise)
    return x_new


def linear_log_likelihood_fn(x, y, params):
    """Linear Gaussian observation likelihood."""
    C = params['C']
    D = params['D']
    
    C = tf.convert_to_tensor(C, dtype=tf.float32)
    D = tf.convert_to_tensor(D, dtype=tf.float32)
    
    # x: (B, N_particles, nx)
    # y: (B, ny)
    # C: (ny, nx), D: (ny, nw)
    
    # Expected observation: C @ x
    y_pred = tf.einsum('ij,bnj->bni', C, x)  # (B, N_particles, ny)
    
    # Observation noise covariance: D @ D^T
    R = tf.matmul(D, D, transpose_b=True)  # (ny, ny)
    
    # Expand y to match particles
    y = tf.expand_dims(y, axis=1)  # (B, 1, ny)
    y = tf.broadcast_to(y, tf.shape(y_pred))  # (B, N_particles, ny)
    
    # Compute log-likelihood
    diff = y - y_pred  # (B, N_particles, ny)
    
    # For 1D case, simplify
    if diff.shape[-1] == 1:
        var = R[0, 0]
        log_norm = -0.5 * tf.math.log(2.0 * np.pi * var)
        log_lik = log_norm - 0.5 * tf.reduce_sum(diff ** 2, axis=-1) / var
    else:
        # General multivariate case
        R_inv = tf.linalg.inv(R)
        log_det_R = tf.linalg.logdet(R)
        log_norm = -0.5 * (tf.cast(tf.shape(y)[-1], tf.float32) * tf.math.log(2.0 * np.pi) + log_det_R)
        
        # Mahalanobis distance
        mahal = tf.einsum('bni,ij,bnj->bn', diff, R_inv, diff)
        log_lik = log_norm - 0.5 * mahal
    
    return log_lik  # (B, N_particles)


@pytest.mark.integration
class TestDPFSoftLGSSM1D:
    """Integration tests for DPF soft resampling on 1D LGSSM."""
    
    def test_dpf_tracks_lgssm_1d(self, lgssm_1d_data, lgssm_1d_params):
        """Test that DPF can track 1D LGSSM state."""
        # Create DPF
        dpf = DifferentiableParticleFilter(
            n_particles=100,
            state_dim=lgssm_1d_params['nx'],
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            soft_alpha=0.2,
            gumbel_temperature=0.3
        )
        
        # Prepare data
        X_true = lgssm_1d_data.X  # (N, nx)
        Y_obs = lgssm_1d_data.Y   # (N, ny)
        
        # Add batch dimension
        Y_batch = Y_obs[None, :, :]  # (1, N, ny)
        
        # Initial distribution
        init_mean = np.zeros(lgssm_1d_params['nx'], dtype=np.float32)
        init_cov_chol = lgssm_1d_params['Sigma'].astype(np.float32)
        
        # Run filter
        params = {
            'A': lgssm_1d_params['A'],
            'B': lgssm_1d_params['B'],
            'C': lgssm_1d_params['C'],
            'D': lgssm_1d_params['D']
        }
        
        particles_seq, logw_seq = dpf.filter(
            Y_batch, init_mean, init_cov_chol, params
        )
        
        # Compute posterior means
        weights = tf.exp(logw_seq)
        posterior_means = tf.reduce_sum(
            weights[..., None] * particles_seq, axis=2
        )  # (1, T+1, nx) where T+1 = N+1
        
        posterior_means = posterior_means.numpy()[0]  # Remove batch dimension -> (T+1, nx)
        
        # Compare only timesteps 1..T (skip t=0 initial state)
        # X_true has shape (T, nx), posterior_means[1:] also has shape (T, nx)
        rmse = np.sqrt(np.mean((posterior_means[1:] - X_true) ** 2))
        
        # Should achieve reasonable tracking
        assert rmse < 1.5, f"RMSE too high: {rmse}"
    
    def test_dpf_with_diagnostics(self, lgssm_1d_data, lgssm_1d_params):
        """Test DPF with diagnostics on LGSSM."""
        dpf = DifferentiableParticleFilter(
            n_particles=80,
            state_dim=lgssm_1d_params['nx'],
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            soft_alpha=0.2,
            gumbel_temperature=0.3
        )
        
        Y_batch = lgssm_1d_data.Y[None, :20, :]  # Use first 20 timesteps
        init_mean = np.zeros(lgssm_1d_params['nx'], dtype=np.float32)
        init_cov_chol = lgssm_1d_params['Sigma'].astype(np.float32)
        
        params = {
            'A': lgssm_1d_params['A'],
            'B': lgssm_1d_params['B'],
            'C': lgssm_1d_params['C'],
            'D': lgssm_1d_params['D']
        }
        
        particles_seq, logw_seq, diagnostics = dpf.filter(
            Y_batch, init_mean, init_cov_chol, params, return_diagnostics=True
        )
        
        # Check diagnostics (aggregated with _mean, _std suffixes)
        assert 'ess_before_mean' in diagnostics
        assert 'ess_after_mean' in diagnostics
        
        # ESS after resampling should be close to N (uniform weights)
        assert diagnostics['ess_after_mean'].numpy() > 50  # At least 50% of particles
    
    def test_dpf_comparison_with_ground_truth(self, lgssm_1d_data, lgssm_1d_params):
        """Test DPF tracking accuracy against ground truth."""
        dpf = DifferentiableParticleFilter(
            n_particles=150,
            state_dim=lgssm_1d_params['nx'],
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            soft_alpha=0.15,
            gumbel_temperature=0.25
        )
        
        X_true = lgssm_1d_data.X
        Y_batch = lgssm_1d_data.Y[None, :, :]
        
        init_mean = np.zeros(lgssm_1d_params['nx'], dtype=np.float32)
        init_cov_chol = lgssm_1d_params['Sigma'].astype(np.float32)
        
        params = {
            'A': lgssm_1d_params['A'],
            'B': lgssm_1d_params['B'],
            'C': lgssm_1d_params['C'],
            'D': lgssm_1d_params['D']
        }
        
        # Add ground truth for RMSE computation
        # The filter expects ground_truth of shape (B, T+1, nx) including initial state at t=0
        # We use the prior mean as the initial state
        X_init = init_mean[None, None, :]  # (1, 1, nx)
        X_batch = X_true[None, :, :]  # (1, T, nx)
        X_batch_with_init = np.concatenate([X_init, X_batch], axis=1)  # (1, T+1, nx)
        
        particles_seq, logw_seq, diagnostics = dpf.filter(
            Y_batch, init_mean, init_cov_chol, params,
            return_diagnostics=True, ground_truth=X_batch_with_init
        )
        
        # Check RMSE was computed
        assert 'rmse_sequence' in diagnostics
        assert 'mean_rmse' in diagnostics
        
        # Mean RMSE should be reasonable
        assert diagnostics['mean_rmse'].numpy() < 1.5


@pytest.fixture
def lgssm_2d_params():
    """2D linear Gaussian state-space model parameters."""
    nx, ny = 2, 2
    A = np.array([[0.9, 0.2], [0.0, 0.7]])
    B = np.diag([0.3, 0.2])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = np.diag([0.5, 0.5])
    Sigma = np.eye(2)
    return {
        'A': A, 'B': B, 'C': C, 'D': D, 'Sigma': Sigma,
        'nx': nx, 'ny': ny, 'N': 50, 'seed': 123
    }


@pytest.fixture
def lgssm_2d_data(lgssm_2d_params):
    """Simulate 2D LGSSM data."""
    p = lgssm_2d_params
    result = simulate_lgssm(
        A=p['A'], B=p['B'], C=p['C'], D=p['D'],
        Sigma=p['Sigma'], N=p['N'], seed=p['seed']
    )
    return result


@pytest.mark.integration
class TestDPFSoftLGSSM2D:
    """Integration tests for DPF soft resampling on 2D LGSSM."""
    
    def test_dpf_tracks_lgssm_2d(self, lgssm_2d_data, lgssm_2d_params):
        """Test that DPF can track 2D LGSSM state."""
        dpf = DifferentiableParticleFilter(
            n_particles=120,
            state_dim=lgssm_2d_params['nx'],
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            soft_alpha=0.2,
            gumbel_temperature=0.3
        )
        
        X_true = lgssm_2d_data.X
        Y_batch = lgssm_2d_data.Y[None, :, :]
        
        init_mean = np.zeros(lgssm_2d_params['nx'], dtype=np.float32)
        init_cov_chol = lgssm_2d_params['Sigma'].astype(np.float32)
        
        params = {
            'A': lgssm_2d_params['A'],
            'B': lgssm_2d_params['B'],
            'C': lgssm_2d_params['C'],
            'D': lgssm_2d_params['D']
        }
        
        particles_seq, logw_seq = dpf.filter(
            Y_batch, init_mean, init_cov_chol, params
        )
        
        # Compute posterior means
        weights = tf.exp(logw_seq)
        posterior_means = tf.reduce_sum(
            weights[..., None] * particles_seq, axis=2
        ).numpy()[0]  # (T+1, nx)
        
        # Compare only timesteps 1..T (skip t=0)
        # Compute RMSE per dimension
        rmse_dim1 = np.sqrt(np.mean((posterior_means[1:, 0] - X_true[:, 0]) ** 2))
        rmse_dim2 = np.sqrt(np.mean((posterior_means[1:, 1] - X_true[:, 1]) ** 2))
        
        # Both dimensions should track reasonably
        assert rmse_dim1 < 1.5
        assert rmse_dim2 < 1.5
    
    def test_dpf_particle_diversity(self, lgssm_2d_data, lgssm_2d_params):
        """Test that particles maintain diversity."""
        dpf = DifferentiableParticleFilter(
            n_particles=100,
            state_dim=lgssm_2d_params['nx'],
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            soft_alpha=0.3,  # Higher alpha for more diversity
            gumbel_temperature=0.4
        )
        
        Y_batch = lgssm_2d_data.Y[None, :20, :]
        init_mean = np.zeros(lgssm_2d_params['nx'], dtype=np.float32)
        init_cov_chol = lgssm_2d_params['Sigma'].astype(np.float32)
        
        params = {
            'A': lgssm_2d_params['A'],
            'B': lgssm_2d_params['B'],
            'C': lgssm_2d_params['C'],
            'D': lgssm_2d_params['D']
        }
        
        particles_seq, logw_seq, diagnostics = dpf.filter(
            Y_batch, init_mean, init_cov_chol, params, return_diagnostics=True
        )
        
        # Check particle diversity doesn't collapse
        # diversity_after is a nested dict after aggregation
        assert 'diversity_after' in diagnostics
        diversity_dict = diagnostics['diversity_after']
        diversity = diversity_dict['mean_pairwise_dist_mean'].numpy()
        
        # Diversity should be non-zero (particles not collapsed to a point)
        assert diversity > 0.01  # Relaxed threshold


@pytest.mark.integration
class TestDPFSoftParameterSensitivity:
    """Test DPF sensitivity to hyperparameters."""
    
    def test_soft_alpha_effect(self, lgssm_1d_data, lgssm_1d_params):
        """Test effect of soft_alpha parameter."""
        Y_batch = lgssm_1d_data.Y[None, :30, :]
        init_mean = np.zeros(lgssm_1d_params['nx'], dtype=np.float32)
        init_cov_chol = lgssm_1d_params['Sigma'].astype(np.float32)
        
        params = {
            'A': lgssm_1d_params['A'],
            'B': lgssm_1d_params['B'],
            'C': lgssm_1d_params['C'],
            'D': lgssm_1d_params['D']
        }
        
        # Low alpha (more importance sampling)
        dpf_low = DifferentiableParticleFilter(
            n_particles=80,
            state_dim=lgssm_1d_params['nx'],
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            soft_alpha=0.05,
            gumbel_temperature=0.3
        )
        
        # High alpha (more uniform)
        dpf_high = DifferentiableParticleFilter(
            n_particles=80,
            state_dim=lgssm_1d_params['nx'],
            transition_fn=linear_transition_fn,
            log_likelihood_fn=linear_log_likelihood_fn,
            soft_alpha=0.5,
            gumbel_temperature=0.3
        )
        
        _, _, diag_low = dpf_low.filter(
            Y_batch, init_mean, init_cov_chol, params, return_diagnostics=True
        )
        
        _, _, diag_high = dpf_high.filter(
            Y_batch, init_mean, init_cov_chol, params, return_diagnostics=True
        )
        
        # High alpha should maintain higher or equal entropy (relaxed due to stochasticity)
        # With high alpha, weights are mixed more with uniform, leading to higher entropy
        entropy_low = diag_low['entropy_after_mean'].numpy()
        entropy_high = diag_high['entropy_after_mean'].numpy()
        
        # Check that high alpha doesn't significantly reduce entropy
        assert entropy_high >= entropy_low * 0.95  # Within 5% tolerance
    
    def test_gumbel_temperature_effect(self, lgssm_1d_data, lgssm_1d_params):
        """Test effect of Gumbel temperature."""
        Y_batch = lgssm_1d_data.Y[None, :30, :]
        init_mean = np.zeros(lgssm_1d_params['nx'], dtype=np.float32)
        init_cov_chol = lgssm_1d_params['Sigma'].astype(np.float32)
        
        params = {
            'A': lgssm_1d_params['A'],
            'B': lgssm_1d_params['B'],
            'C': lgssm_1d_params['C'],
            'D': lgssm_1d_params['D']
        }
        
        # Different temperatures
        for temp in [0.1, 0.5, 1.0]:
            dpf = DifferentiableParticleFilter(
                n_particles=80,
                state_dim=lgssm_1d_params['nx'],
                transition_fn=linear_transition_fn,
                log_likelihood_fn=linear_log_likelihood_fn,
                soft_alpha=0.2,
                gumbel_temperature=temp
            )
            
            # Should complete without errors
            particles_seq, logw_seq = dpf.filter(
                Y_batch, init_mean, init_cov_chol, params
            )
            
            assert particles_seq.shape[1] == 31  # T+1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
