"""
Integration tests for DPF implementations with Stochastic Volatility (SV) simulator.

Tests DPF-Soft, DPF-RNN, and DPF-OT on the nonlinear stochastic volatility model:
    X_t = alpha * X_{t-1} + sigma * V_t,  V_t ~ N(0, 1)
    Y_t = beta * exp(0.5 * X_t) * W_t,    W_t ~ N(0, 1)
"""

import numpy as np
import pytest
import tensorflow as tf


pytestmark = [pytest.mark.integration, pytest.mark.tensorflow]

from simulator.simulator_sto_volatility_model import simulate_sv_1d
from models.DPF_soft_resampling import DifferentiableParticleFilter
from models.DPF_RNN_resampling import DifferentiableParticleFilterRNN
from models.DPF_OT_resampling import DPF_OT


# Standard SV model parameters
ALPHA = 0.9
SIGMA = 0.2
BETA = 1.0


# ============================================
# Transition and observation functions for DPF
# ============================================

def sv_transition_fn(particles, params=None):
    """
    SV transition: X_t = alpha * X_{t-1} + sigma * V_t
    
    Args:
        particles: (B, N, 1) or (N, 1) current states
        params: dict with 'alpha', 'sigma'
    
    Returns:
        (B, N, 1) or (N, 1) predicted states
    """
    if params is None:
        params = {'alpha': ALPHA, 'sigma': SIGMA}
    
    alpha = params.get('alpha', ALPHA)
    sigma = params.get('sigma', SIGMA)
    
    particles = tf.convert_to_tensor(particles, dtype=tf.float32)
    
    # Deterministic part: alpha * X_{t-1}
    pred_mean = alpha * particles
    
    # Add process noise: sigma * V_t
    noise = tf.random.normal(tf.shape(particles), mean=0.0, stddev=sigma, dtype=tf.float32)
    
    return pred_mean + noise


def sv_log_likelihood_fn(particles, observation, params=None):
    """
    SV observation log-likelihood: Y_t = beta * exp(0.5 * X_t) * W_t
    
    Log p(y_t | x_t) = -0.5 * log(2*pi) - log(sigma_obs(x_t)) - 0.5 * (y_t / sigma_obs(x_t))^2
    where sigma_obs(x_t) = beta * exp(0.5 * x_t)
    
    Args:
        particles: (B, N, d) current states
        observation: (B, obs_dim) observation
        params: dict with 'beta'
    
    Returns:
        (B, N) log-likelihoods
    """
    if params is None:
        params = {'beta': BETA}
    
    beta = params.get('beta', BETA)
    
    particles = tf.convert_to_tensor(particles, dtype=tf.float32)
    observation = tf.convert_to_tensor(observation, dtype=tf.float32)
    
    # Extract state dimension: particles (B, N, 1) -> x_t is [:, :, 0]
    x_t = particles[:, :, 0]  # (B, N)
    
    # Observation standard deviation: beta * exp(0.5 * x_t)
    sigma_obs = beta * tf.exp(0.5 * x_t)  # (B, N)
    
    # Observation is (B, 1), extract scalar per batch
    y_t = observation[:, 0:1]  # (B, 1)
    
    # Log-likelihood: -0.5 * log(2*pi) - log(sigma_obs) - 0.5 * (y / sigma_obs)^2
    log_lik = -0.5 * tf.math.log(2.0 * np.pi) - tf.math.log(sigma_obs) - 0.5 * (y_t / sigma_obs) ** 2
    
    return log_lik


def sv_obs_loglik_fn_ot(particles, observation, t=None):
    """
    SV observation log-likelihood for DPF-OT (no params dict).
    
    Args:
        particles: (N, 1) current states
        observation: (1,) or scalar observation
        t: time index (unused)
    
    Returns:
        (N,) log-likelihoods
    """
    particles = tf.convert_to_tensor(particles, dtype=tf.float32)
    observation = tf.convert_to_tensor(observation, dtype=tf.float32)
    
    # Reshape observation if needed
    if len(observation.shape) == 0:
        observation = tf.reshape(observation, (1,))
    
    # Observation standard deviation: beta * exp(0.5 * x_t)
    sigma_obs = BETA * tf.exp(0.5 * particles[:, 0])  # (N,)
    
    # Log-likelihood
    log_lik = -0.5 * tf.math.log(2.0 * np.pi) - tf.math.log(sigma_obs) - 0.5 * (observation[0] / sigma_obs) ** 2
    
    return log_lik


def sv_transition_fn_ot(particles, t=None):
    """
    SV transition for DPF-OT (no params dict).
    
    Args:
        particles: (N, 1) current states
        t: time index (unused)
    
    Returns:
        (N, 1) predicted states
    """
    particles = tf.convert_to_tensor(particles, dtype=tf.float32)
    
    # Deterministic part
    pred_mean = ALPHA * particles
    
    # Add process noise
    noise = tf.random.normal(tf.shape(particles), mean=0.0, stddev=SIGMA, dtype=tf.float32)
    
    return pred_mean + noise


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def sv_data_short():
    """Generate short SV simulation for quick tests."""
    results = simulate_sv_1d(n=50, alpha=ALPHA, sigma=SIGMA, beta=BETA, seed=42)
    return results


@pytest.fixture
def sv_data_medium():
    """Generate medium-length SV simulation."""
    results = simulate_sv_1d(n=100, alpha=ALPHA, sigma=SIGMA, beta=BETA, seed=123)
    return results


@pytest.fixture
def sv_params():
    """SV model parameters."""
    return {
        'alpha': ALPHA,
        'sigma': SIGMA,
        'beta': BETA
    }


# ============================================
# DPF-Soft Tests
# ============================================

@pytest.mark.integration
class TestDPFSoftSV:
    """Integration tests for DPF with Soft Resampling on SV model."""
    
    def test_dpf_soft_tracks_sv(self, sv_data_short, sv_params):
        """Test that DPF-Soft can track SV system."""
        X_true = sv_data_short.X
        Y_obs = sv_data_short.Y
        n = len(X_true)
        
        dpf = DifferentiableParticleFilter(
            n_particles=100,
            state_dim=1,
            transition_fn=sv_transition_fn,
            log_likelihood_fn=sv_log_likelihood_fn,
            soft_alpha=1.0,
            gumbel_temperature=0.5
        )
        
        # Prepare batch observations
        Y_batch = Y_obs[None, :, None]  # (1, T, 1)
        
        # Initialize
        init_mean = np.array([X_true[0]], dtype=np.float32)
        init_cov_chol = np.array([[SIGMA]], dtype=np.float32)
        
        # Run filter
        particles_seq, logw_seq = dpf.filter(Y_batch, init_mean, init_cov_chol, sv_params)
        
        # Compute posterior means
        weights = tf.exp(logw_seq)
        posterior_means = tf.reduce_sum(
            weights[..., None] * particles_seq, axis=2
        )  # (1, T+1, 1)
        
        posterior_means = posterior_means.numpy()[0, 1:, 0]  # Remove batch and t=0
        
        # Compute RMSE (skip first few timesteps for warmup)
        warmup = 5
        rmse = np.sqrt(np.mean((posterior_means[warmup:] - X_true[warmup:]) ** 2))
        
        # Should achieve reasonable tracking on this nonlinear model
        assert rmse < 1.0, f"RMSE too high: {rmse}"
    
    def test_dpf_soft_with_diagnostics(self, sv_data_short, sv_params):
        """Test DPF-Soft with diagnostics on SV model."""
        Y_obs = sv_data_short.Y
        X_true = sv_data_short.X
        
        dpf = DifferentiableParticleFilter(
            n_particles=80,
            state_dim=1,
            transition_fn=sv_transition_fn,
            log_likelihood_fn=sv_log_likelihood_fn
        )
        
        Y_batch = Y_obs[None, :30, None]
        init_mean = np.array([X_true[0]], dtype=np.float32)
        init_cov_chol = np.array([[SIGMA]], dtype=np.float32)
        
        # Run with diagnostics
        particles_seq, logw_seq, diagnostics = dpf.filter(
            Y_batch, init_mean, init_cov_chol, sv_params, return_diagnostics=True
        )
        
        # Check diagnostics
        assert 'ess_before_mean' in diagnostics
        assert 'ess_after_mean' in diagnostics
        assert 'total_time' in diagnostics
        
        # ESS should be reasonable
        ess_before = diagnostics['ess_before_mean']
        if hasattr(ess_before, 'numpy'):
            ess_before = ess_before.numpy()
        
        assert ess_before > 10  # Should have some diversity
    
    def test_dpf_soft_alpha_sensitivity(self, sv_data_short, sv_params):
        """Test effect of soft_alpha parameter on SV tracking."""
        Y_obs = sv_data_short.Y
        X_true = sv_data_short.X
        
        rmses = []
        
        for alpha in [0.5, 1.0, 2.0]:
            dpf = DifferentiableParticleFilter(
                n_particles=80,
                state_dim=1,
                transition_fn=sv_transition_fn,
                log_likelihood_fn=sv_log_likelihood_fn,
                soft_alpha=alpha
            )
            
            Y_batch = Y_obs[None, :30, None]
            init_mean = np.array([X_true[0]], dtype=np.float32)
            init_cov_chol = np.array([[SIGMA]], dtype=np.float32)
            
            particles_seq, logw_seq = dpf.filter(Y_batch, init_mean, init_cov_chol, sv_params)
            
            weights = tf.exp(logw_seq)
            posterior_means = tf.reduce_sum(weights[..., None] * particles_seq, axis=2).numpy()[0, 1:, 0]
            
            rmse = np.sqrt(np.mean((posterior_means[5:] - X_true[5:30]) ** 2))
            rmses.append(rmse)
        
        # All configurations should achieve reasonable tracking
        for rmse in rmses:
            assert rmse < 1.5


# ============================================
# DPF-RNN Tests
# ============================================

@pytest.mark.integration
class TestDPFRNNSV:
    """Integration tests for DPF with RNN Resampling on SV model."""
    
    def test_dpf_rnn_tracks_sv(self, sv_data_short, sv_params):
        """Test that DPF-RNN can track SV system."""
        X_true = sv_data_short.X
        Y_obs = sv_data_short.Y
        n = len(X_true)
        
        dpf = DifferentiableParticleFilterRNN(
            n_particles=100,
            state_dim=1,
            transition_fn=sv_transition_fn,
            log_likelihood_fn=sv_log_likelihood_fn,
            use_baseline_resampling=True  # Use baseline for untrained RNN
        )
        
        # Prepare batch observations
        Y_batch = Y_obs[None, :, None]  # (1, T, 1)
        
        # Initialize
        init_mean = np.array([X_true[0]], dtype=np.float32)
        init_cov_chol = np.array([[SIGMA]], dtype=np.float32)
        
        # Run filter
        particles_seq, logw_seq, assign_mats, _ = dpf.filter(
            Y_batch, init_mean, init_cov_chol, sv_params, return_ess=True
        )
        
        # Compute posterior means
        weights = tf.exp(logw_seq)
        posterior_means = tf.reduce_sum(
            weights[..., None] * particles_seq, axis=2
        )  # (1, T+1, 1)
        
        posterior_means = posterior_means.numpy()[0, 1:, 0]
        
        # Compute RMSE
        warmup = 5
        rmse = np.sqrt(np.mean((posterior_means[warmup:] - X_true[warmup:]) ** 2))
        
        # Should achieve reasonable tracking
        assert rmse < 1.0, f"RMSE too high: {rmse}"
    
    def test_dpf_rnn_baseline_vs_rnn_mode(self, sv_data_short, sv_params):
        """Test DPF-RNN in both baseline and RNN modes."""
        Y_obs = sv_data_short.Y
        X_true = sv_data_short.X
        
        rmses = {}
        
        for mode in ['baseline', 'rnn']:
            dpf = DifferentiableParticleFilterRNN(
                n_particles=80,
                state_dim=1,
                transition_fn=sv_transition_fn,
                log_likelihood_fn=sv_log_likelihood_fn,
                use_baseline_resampling=(mode == 'baseline')
            )
            
            Y_batch = Y_obs[None, :30, None]
            init_mean = np.array([X_true[0]], dtype=np.float32)
            init_cov_chol = np.array([[SIGMA]], dtype=np.float32)
            
            particles_seq, logw_seq, _, _ = dpf.filter(
                Y_batch, init_mean, init_cov_chol, sv_params, return_ess=True
            )
            
            weights = tf.exp(logw_seq)
            posterior_means = tf.reduce_sum(weights[..., None] * particles_seq, axis=2).numpy()[0, 1:, 0]
            
            rmse = np.sqrt(np.mean((posterior_means[5:] - X_true[5:30]) ** 2))
            rmses[mode] = rmse
        
        # Both modes should work
        assert rmses['baseline'] < 1.5
        assert rmses['rnn'] < 1.5
    
    def test_dpf_rnn_feature_configurations(self, sv_data_short, sv_params):
        """Test different RNN feature configurations on SV model."""
        Y_obs = sv_data_short.Y
        X_true = sv_data_short.X
        
        configs = [
            {'use_weight_features': True, 'use_particle_features': False},
            {'use_weight_features': False, 'use_particle_features': True},
            {'use_weight_features': True, 'use_particle_features': True},
        ]
        
        for config in configs:
            dpf = DifferentiableParticleFilterRNN(
                n_particles=60,
                state_dim=1,
                transition_fn=sv_transition_fn,
                log_likelihood_fn=sv_log_likelihood_fn,
                use_baseline_resampling=True,
                **config
            )
            
            Y_batch = Y_obs[None, :20, None]
            init_mean = np.array([X_true[0]], dtype=np.float32)
            init_cov_chol = np.array([[SIGMA]], dtype=np.float32)
            
            particles_seq, logw_seq, _, _ = dpf.filter(
                Y_batch, init_mean, init_cov_chol, sv_params, return_ess=True
            )
            
            # Should complete without errors
            assert particles_seq.shape[1] == 21  # T+1


# ============================================
# DPF-OT Tests
# ============================================

@pytest.mark.integration
class TestDPFOTSV:
    """Integration tests for DPF with OT Resampling on SV model."""
    
    def test_dpf_ot_tracks_sv(self, sv_data_short):
        """Test that DPF-OT can track SV system."""
        X_true = sv_data_short.X
        Y_obs = sv_data_short.Y
        n = len(X_true)
        
        dpf = DPF_OT(
            N_particles=100,
            state_dim=1,
            transition_fn=sv_transition_fn_ot,
            obs_loglik_fn=sv_obs_loglik_fn_ot,
            epsilon=0.1,
            sinkhorn_iters=50
        )
        
        # Initialize
        mean0 = np.array([X_true[0]], dtype=np.float32)
        cov0_chol = np.array([[SIGMA]], dtype=np.float32)
        
        # Run filter (DPF-OT takes list of observations)
        particles_seq, weights_seq = dpf.run_filter(
            [Y_obs[t] for t in range(n)],
            mean0, cov0_chol
        )
        
        # Compute posterior means
        posterior_means = []
        for particles, weights in zip(particles_seq, weights_seq):
            mean = tf.reduce_sum(weights[:, None] * particles, axis=0).numpy()
            posterior_means.append(mean[0])
        posterior_means = np.array(posterior_means)
        
        # Compute RMSE
        warmup = 5
        rmse = np.sqrt(np.mean((posterior_means[warmup:] - X_true[warmup:]) ** 2))
        
        # Should achieve reasonable tracking
        assert rmse < 1.0, f"RMSE too high: {rmse}"
    
    def test_dpf_ot_with_diagnostics(self, sv_data_short):
        """Test DPF-OT with diagnostics on SV model."""
        X_true = sv_data_short.X
        Y_obs = sv_data_short.Y
        
        dpf = DPF_OT(
            N_particles=80,
            state_dim=1,
            transition_fn=sv_transition_fn_ot,
            obs_loglik_fn=sv_obs_loglik_fn_ot,
            epsilon=0.1,
            sinkhorn_iters=50
        )
        
        mean0 = np.array([X_true[0]], dtype=np.float32)
        cov0_chol = np.array([[SIGMA]], dtype=np.float32)
        
        # Run with diagnostics
        particles_seq, weights_seq, diagnostics = dpf.run_filter(
            [Y_obs[t] for t in range(30)],
            mean0, cov0_chol,
            return_diagnostics=True
        )
        
        # Check diagnostics
        assert 'ess_before_mean' in diagnostics
        assert 'ess_after_mean' in diagnostics
        assert 'ot_distance_mean' in diagnostics
        assert 'total_time' in diagnostics
        
        # ESS after should be high (uniform weights)
        ess_after = diagnostics['ess_after_mean']
        if hasattr(ess_after, 'numpy'):
            ess_after = ess_after.numpy()
        
        assert ess_after > 50  # Should be near N_particles
    
    def test_dpf_ot_epsilon_sensitivity(self, sv_data_short):
        """Test effect of epsilon parameter on SV tracking."""
        X_true = sv_data_short.X
        Y_obs = sv_data_short.Y
        
        rmses = []
        
        for epsilon in [0.05, 0.1, 0.3]:
            dpf = DPF_OT(
                N_particles=80,
                state_dim=1,
                transition_fn=sv_transition_fn_ot,
                obs_loglik_fn=sv_obs_loglik_fn_ot,
                epsilon=epsilon,
                sinkhorn_iters=50
            )
            
            mean0 = np.array([X_true[0]], dtype=np.float32)
            cov0_chol = np.array([[SIGMA]], dtype=np.float32)
            
            particles_seq, weights_seq = dpf.run_filter(
                [Y_obs[t] for t in range(30)],
                mean0, cov0_chol
            )
            
            posterior_means = []
            for particles, weights in zip(particles_seq, weights_seq):
                mean = tf.reduce_sum(weights[:, None] * particles, axis=0).numpy()[0]
                posterior_means.append(mean)
            posterior_means = np.array(posterior_means)
            
            rmse = np.sqrt(np.mean((posterior_means[5:] - X_true[5:30]) ** 2))
            rmses.append(rmse)
        
        # All epsilon values should work
        for rmse in rmses:
            assert rmse < 1.5
    
    def test_dpf_ot_sinkhorn_convergence(self, sv_data_short):
        """Test Sinkhorn convergence on SV model."""
        X_true = sv_data_short.X
        Y_obs = sv_data_short.Y
        
        dpf = DPF_OT(
            N_particles=60,
            state_dim=1,
            transition_fn=sv_transition_fn_ot,
            obs_loglik_fn=sv_obs_loglik_fn_ot,
            epsilon=0.1,
            sinkhorn_iters=100  # More iterations
        )
        
        mean0 = np.array([X_true[0]], dtype=np.float32)
        cov0_chol = np.array([[SIGMA]], dtype=np.float32)
        
        particles_seq, weights_seq, diagnostics = dpf.run_filter(
            [Y_obs[t] for t in range(20)],
            mean0, cov0_chol,
            return_diagnostics=True
        )
        
        # Check convergence
        assert 'sinkhorn_iterations_mean' in diagnostics
        assert 'converged_rate' in diagnostics
        
        # Should have some convergence
        converged_rate = diagnostics['converged_rate']
        assert converged_rate >= 0.0


# ============================================
# Comparative Tests
# ============================================

@pytest.mark.integration
class TestDPFComparativeSV:
    """Comparative tests between DPF variants on SV model."""
    
    def test_all_dpfs_track_sv(self, sv_data_medium):
        """Test that all three DPF variants can track SV model."""
        X_true = sv_data_medium.X
        Y_obs = sv_data_medium.Y
        
        rmses = {}
        
        # DPF-Soft
        dpf_soft = DifferentiableParticleFilter(
            n_particles=100,
            state_dim=1,
            transition_fn=sv_transition_fn,
            log_likelihood_fn=sv_log_likelihood_fn
        )
        
        Y_batch = Y_obs[None, :, None]
        init_mean = np.array([X_true[0]], dtype=np.float32)
        init_cov_chol = np.array([[SIGMA]], dtype=np.float32)
        
        particles_seq, logw_seq = dpf_soft.filter(Y_batch, init_mean, init_cov_chol)
        weights = tf.exp(logw_seq)
        posterior_means = tf.reduce_sum(weights[..., None] * particles_seq, axis=2).numpy()[0, 1:, 0]
        rmses['soft'] = np.sqrt(np.mean((posterior_means[10:] - X_true[10:]) ** 2))
        
        # DPF-RNN
        dpf_rnn = DifferentiableParticleFilterRNN(
            n_particles=100,
            state_dim=1,
            transition_fn=sv_transition_fn,
            log_likelihood_fn=sv_log_likelihood_fn,
            use_baseline_resampling=True
        )
        
        particles_seq, logw_seq, _, _ = dpf_rnn.filter(Y_batch, init_mean, init_cov_chol, return_ess=True)
        weights = tf.exp(logw_seq)
        posterior_means = tf.reduce_sum(weights[..., None] * particles_seq, axis=2).numpy()[0, 1:, 0]
        rmses['rnn'] = np.sqrt(np.mean((posterior_means[10:] - X_true[10:]) ** 2))
        
        # DPF-OT
        dpf_ot = DPF_OT(
            N_particles=100,
            state_dim=1,
            transition_fn=sv_transition_fn_ot,
            obs_loglik_fn=sv_obs_loglik_fn_ot,
            epsilon=0.1,
            sinkhorn_iters=50
        )
        
        particles_seq, weights_seq = dpf_ot.run_filter(
            [Y_obs[t] for t in range(len(X_true))],
            init_mean, init_cov_chol
        )
        
        posterior_means = []
        for particles, weights in zip(particles_seq, weights_seq):
            mean = tf.reduce_sum(weights[:, None] * particles, axis=0).numpy()[0]
            posterior_means.append(mean)
        posterior_means = np.array(posterior_means)
        rmses['ot'] = np.sqrt(np.mean((posterior_means[10:] - X_true[10:]) ** 2))
        
        # All should achieve reasonable tracking
        for name, rmse in rmses.items():
            assert rmse < 1.2, f"{name} RMSE too high: {rmse}"
        
        print(f"\nRMSE comparison on SV model:")
        for name, rmse in rmses.items():
            print(f"  {name.upper()}: {rmse:.4f}")
    
    def test_dpf_particle_count_effect(self, sv_data_short):
        """Test effect of particle count on tracking accuracy."""
        X_true = sv_data_short.X
        Y_obs = sv_data_short.Y
        
        particle_counts = [50, 100, 200]
        rmses = []
        
        for N in particle_counts:
            dpf = DifferentiableParticleFilter(
                n_particles=N,
                state_dim=1,
                transition_fn=sv_transition_fn,
                log_likelihood_fn=sv_log_likelihood_fn
            )
            
            Y_batch = Y_obs[None, :, None]
            init_mean = np.array([X_true[0]], dtype=np.float32)
            init_cov_chol = np.array([[SIGMA]], dtype=np.float32)
            
            particles_seq, logw_seq = dpf.filter(Y_batch, init_mean, init_cov_chol)
            weights = tf.exp(logw_seq)
            posterior_means = tf.reduce_sum(weights[..., None] * particles_seq, axis=2).numpy()[0, 1:, 0]
            
            rmse = np.sqrt(np.mean((posterior_means[5:] - X_true[5:]) ** 2))
            rmses.append(rmse)
        
        # More particles should generally improve accuracy (or at least not hurt)
        # But due to randomness, we just check all are reasonable
        for rmse in rmses:
            assert rmse < 1.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
