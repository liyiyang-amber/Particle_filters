import numpy as np
import tensorflow as tf


class DifferentiableParticleFilter(tf.Module):
    """Differentiable Particle Filter with soft-resampling.

    Implements a differentiable particle filter with soft-resampling using
    a mixture with uniform distribution via Gumbel-Softmax reparameterization.
    """

    def __init__(
        self,
        n_particles,
        state_dim,
        transition_fn,
        log_likelihood_fn,
        soft_alpha=0.1,
        gumbel_temperature=0.2,
        name=None,
    ):
        """Initialize the Differentiable Particle Filter.

        Args:
            n_particles (int): Number of particles N.
            state_dim (int): Dimension of latent state.
            transition_fn (callable): State transition function (x_prev, params) -> x_pred.
                x_prev has shape (B, N, state_dim), returns shape (B, N, state_dim).
            log_likelihood_fn (callable): Log-likelihood function (x, y, params) -> log p(y | x).
                x has shape (B, N, state_dim), y has shape (B, obs_dim), returns shape (B, N).
            soft_alpha (float, optional): Mixing parameter in [0,1] for mixture with uniform,
                q = (1 - alpha) * w + alpha * 1/N. Defaults to 0.1.
            gumbel_temperature (float, optional): Temperature for Gumbel-Softmax.
                Smaller values create harder assignments. Defaults to 0.2.
            name (str, optional): Name for the module. Defaults to None.
        """
        super().__init__(name=name)
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.transition_fn = transition_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.soft_alpha = soft_alpha
        self.gumbel_temperature = gumbel_temperature

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _log_normalize(log_w, axis=-1, keepdims=False):
        """Normalize log-weights along given axis.

        Args:
            log_w (Tensor): Log-weights to normalize.
            axis (int, optional): Axis along which to normalize. Defaults to -1.
            keepdims (bool, optional): Whether to keep dimensions. Defaults to False.

        Returns:
            Tuple[Tensor, Tensor]: Normalized log-weights and log normalization constant.
        """
        log_z = tf.reduce_logsumexp(log_w, axis=axis, keepdims=True)
        log_w_norm = log_w - log_z
        if not keepdims:
            log_z = tf.squeeze(log_z, axis=axis)
        return log_w_norm, log_z

    @staticmethod
    def compute_ess(log_weights):
        """Compute Effective Sample Size (ESS) from log-weights.

        ESS measures the quality of particle approximation: ESS = 1 / sum(w_i^2),
        where w_i are normalized weights.

        Args:
            log_weights (Tensor): Log-weights of shape (B, N).

        Returns:
            Tensor: Effective Sample Size for each batch, shape (B,).
                Range: [1, N] where N is number of particles.
                ESS ≈ N indicates equal weights (good),
                ESS ≈ 1 indicates one particle dominates (bad, degeneracy).
        """
        # Normalize weights
        weights = tf.exp(log_weights)  # (B, N)
        weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
        
        # ESS = 1 / sum(w^2)
        ess = 1.0 / tf.reduce_sum(weights ** 2, axis=-1)
        return ess
    
    @staticmethod
    def compute_weight_entropy(log_weights):
        """Compute entropy of weight distribution.

        Computes H(w) = -sum(w_i * log(w_i)).
        Higher entropy indicates more uniform weights (better particle diversity).

        Args:
            log_weights (Tensor): Log-weights of shape (B, N).

        Returns:
            Tensor: Entropy for each batch, shape (B,).
        """
        weights = tf.exp(log_weights)  # (B, N)
        weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
        
        # Avoid log(0) by adding small epsilon
        entropy = -tf.reduce_sum(weights * tf.math.log(weights + 1e-10), axis=-1)
        return entropy
    
    @staticmethod
    def compute_particle_diversity(particles):
        """Compute particle diversity metrics.

        Args:
            particles (Tensor): Particle states of shape (B, N, d).

        Returns:
            dict: Dictionary containing diversity metrics:
                - 'mean_pairwise_dist': Mean distance between particles.
                - 'std_pairwise_dist': Standard deviation of pairwise distances.
                - 'particle_spread': Trace of covariance matrix.
        """
        B = tf.shape(particles)[0]
        N = tf.shape(particles)[1]
        d = tf.shape(particles)[2]
        
        # Compute mean pairwise distances
        # particles: (B, N, d)
        p1 = tf.expand_dims(particles, axis=2)  # (B, N, 1, d)
        p2 = tf.expand_dims(particles, axis=1)  # (B, 1, N, d)
        diff = p1 - p2  # (B, N, N, d)
        distances = tf.norm(diff, axis=-1)  # (B, N, N)
        
        # Mean over all pairs (excluding diagonal)
        mask = 1.0 - tf.eye(N, dtype=tf.float32)  # Exclude self-distances
        masked_distances = distances * mask
        mean_dist = tf.reduce_sum(masked_distances, axis=[1, 2]) / (tf.cast(N * (N - 1), tf.float32))
        
        # Std of pairwise distances
        flat_distances = tf.reshape(masked_distances, [B, N * N])
        std_dist = tf.math.reduce_std(flat_distances, axis=-1)
        
        # Particle spread (trace of covariance)
        particle_mean = tf.reduce_mean(particles, axis=1, keepdims=True)  # (B, 1, d)
        centered = particles - particle_mean  # (B, N, d)
        cov = tf.reduce_mean(centered[:, :, :, None] * centered[:, :, None, :], axis=1)  # (B, d, d)
        spread = tf.linalg.trace(cov)  # (B,)
        
        return {
            'mean_pairwise_dist': mean_dist,
            'std_pairwise_dist': std_dist,
            'particle_spread': spread
        }

    @staticmethod
    def _sample_gumbel(shape, eps=1e-20):
        """Sample i.i.d. Gumbel(0,1) variables.

        Args:
            shape (tuple): Shape of the output tensor.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-20.

        Returns:
            Tensor: Gumbel(0,1) samples of specified shape.
        """
        u = tf.random.uniform(shape, minval=eps, maxval=1.0 - eps)
        return -tf.math.log(-tf.math.log(u))

    def _gumbel_softmax(self, log_probs, temperature):
        """Gumbel-Softmax sample given log-probabilities.

        Args:
            log_probs (Tensor): Log probabilities of shape (..., K), unnormalized is OK.
            temperature (float): Scalar temperature > 0.

        Returns:
            Tensor: Soft one-hot vectors of shape (..., K).
        """
        g = self._sample_gumbel(tf.shape(log_probs))
        y = tf.nn.softmax((log_probs + g) / temperature, axis=-1)
        return y

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def init_particles(self, batch_size, init_mean, init_cov_chol):
        """Initialize particles from a Gaussian prior N(init_mean, init_cov).

        Args:
            batch_size (int): Number of parallel sequences B.
            init_mean (Tensor): Prior mean of shape (state_dim,) or (B, state_dim).
            init_cov_chol (Tensor): Lower Cholesky factor of prior covariance,
                shape (state_dim, state_dim) or (B, state_dim, state_dim).

        Returns:
            Tuple[Tensor, Tensor]: Particles of shape (B, N, state_dim) and
                log-weights of shape (B, N).
        """
        N, d = self.n_particles, self.state_dim

        init_mean = tf.convert_to_tensor(init_mean, dtype=tf.float32)
        init_cov_chol = tf.convert_to_tensor(init_cov_chol, dtype=tf.float32)

        # Broadcast mean and chol to batch dimension
        if init_mean.shape.rank == 1:
            init_mean = tf.broadcast_to(init_mean[None, :], (batch_size, d))
        if init_cov_chol.shape.rank == 2:
            init_cov_chol = tf.broadcast_to(
                init_cov_chol[None, :, :], (batch_size, d, d)
            )

        # Sample standard normal noise (B, N, d)
        eps = tf.random.normal((batch_size, N, d), dtype=tf.float32)
        # Apply Cholesky: eps @ L^T
        L = init_cov_chol  # (B, d, d)
        noise = tf.einsum("bnd,bdk->bnk", eps, L)

        # Broadcast mean to (B, N, d)
        mean_expanded = tf.expand_dims(init_mean, axis=1)  # (B,1,d)
        mean_expanded = tf.broadcast_to(mean_expanded, (batch_size, N, d))

        particles = mean_expanded + noise
        log_weights = tf.math.log(
            tf.fill((batch_size, self.n_particles), 1.0 / self.n_particles)
        )
        return particles, log_weights

    # ------------------------------------------------------------------
    # Single filtering step
    # ------------------------------------------------------------------
    def step(self, particles, log_weights, observation, params=None, return_diagnostics=False):
        """Perform one step of the differentiable particle filter.

        Args:
            particles (Tensor): Previous particle states of shape (B, N, d).
            log_weights (Tensor): Previous log weights of shape (B, N).
            observation (Tensor): Current observation of shape (B, obs_dim).
            params (dict, optional): Parameters passed to transition and likelihood functions.
                Defaults to None.
            return_diagnostics (bool, optional): If True, return diagnostic metrics.
                Defaults to False.

        Returns:
            Tuple: If return_diagnostics is False, returns (new_particles, new_log_weights).
                new_particles has shape (B, N, d), new_log_weights has shape (B, N).
                If return_diagnostics is True, also returns diagnostics dict as third element.
        """
        if params is None:
            params = {}

        # Compute diagnostics before resampling (if requested)
        if return_diagnostics:
            ess_before = self.compute_ess(log_weights)
            entropy_before = self.compute_weight_entropy(log_weights)
            diversity_before = self.compute_particle_diversity(particles)

        # 1) Propagation: x_t^{(i)} ~ p(x_t | x_{t-1}^{(i)})
        pred_particles = self.transition_fn(particles, params)  # (B, N, d)

        # 2) Importance weights update: w_t^{(i)} ∝ w_{t-1}^{(i)} * p(y_t | x_t^{(i)})
        log_lik = self.log_likelihood_fn(pred_particles, observation, params)  # (B, N)
        log_weights = log_weights + log_lik
        log_weights, _ = self._log_normalize(log_weights, axis=-1, keepdims=False)
        weights = tf.exp(log_weights)  # (B, N)

        # 3) Soft-resampling with mixture with uniform
        # q = (1 - alpha) * w + alpha * 1/N
        # This mixture prevents complete weight degeneracy while maintaining differentiability
        alpha = self.soft_alpha
        N = self.n_particles
        uniform = tf.fill(tf.shape(weights), 1.0 / N)
        probs = (1.0 - alpha) * weights + alpha * uniform  # (B, N)

        # Convert to log-probs (no need to be exactly normalized)
        log_probs = tf.math.log(probs + 1e-20)  # (B, N)

        # For each new particle i = 1..N, we want a (soft) ancestor distribution
        # We let every new particle use the same base distribution log_probs,
        # but add *independent* Gumbel noise per i.
        #
        # Shape: (B, N, N) where last dim indexes 'ancestor j'
        log_probs_expanded = tf.expand_dims(log_probs, axis=1)  # (B,1,N)
        log_probs_tiled = tf.tile(log_probs_expanded, [1, N, 1])  # (B,N,N)

        # Gumbel-Softmax along last axis
        assign_matrix = self._gumbel_softmax(
            log_probs_tiled, temperature=self.gumbel_temperature
        )  # (B,N,N)

        # 4) Construct new particles as barycenters: x'_i = sum_j a_{ij} x_j
        # pred_particles: (B, N, d)
        new_particles = tf.einsum("bij,bjd->bid", assign_matrix, pred_particles)

        # After resampling, reset weights to uniform
        new_log_weights = tf.math.log(
            tf.fill((tf.shape(particles)[0], N), 1.0 / N)
        )
        
        # Compute diagnostics after resampling (if requested)
        if return_diagnostics:
            ess_after = self.compute_ess(new_log_weights)
            entropy_after = self.compute_weight_entropy(new_log_weights)
            diversity_after = self.compute_particle_diversity(new_particles)
            
            # Compute assignment matrix statistics
            assign_entropy = -tf.reduce_sum(
                assign_matrix * tf.math.log(assign_matrix + 1e-10), axis=-1
            )  # (B, N)
            
            diagnostics = {
                'ess_before': ess_before,
                'ess_after': ess_after,
                'entropy_before': entropy_before,
                'entropy_after': entropy_after,
                'diversity_before': diversity_before,
                'diversity_after': diversity_after,
                'assignment_entropy_mean': tf.reduce_mean(assign_entropy),
                'assignment_entropy_std': tf.math.reduce_std(assign_entropy),
                'max_weight_before': tf.reduce_max(weights, axis=-1),
                'soft_alpha': self.soft_alpha,
                'gumbel_temperature': self.gumbel_temperature,
            }
            return new_particles, new_log_weights, diagnostics

        return new_particles, new_log_weights

    # ------------------------------------------------------------------
    # Full filtering over a sequence
    # ------------------------------------------------------------------
    def filter(self, observations, init_mean, init_cov_chol, params=None, 
               return_diagnostics=False, ground_truth=None):
        """Run differentiable particle filter over a sequence of observations.

        Args:
            observations (Tensor): Observation sequence of shape (B, T, obs_dim).
            init_mean (Tensor): Initial prior mean, see init_particles.
            init_cov_chol (Tensor): Initial prior covariance Cholesky factor, see init_particles.
            params (dict, optional): Parameters for transition and likelihood functions.
                Can be static or contain time-dependent tensors. Defaults to None.
            return_diagnostics (bool, optional): If True, return detailed diagnostic metrics.
                Defaults to False.
            ground_truth (Tensor, optional): True states of shape (B, T+1, state_dim)
                for computing RMSE. Defaults to None.

        Returns:
            Tuple: If return_diagnostics is False, returns (particles_seq, logw_seq).
                particles_seq has shape (B, T+1, N, d), logw_seq has shape (B, T+1, N).
                If return_diagnostics is True, also returns diagnostics dict as third element.
        """
        if params is None:
            params = {}

        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        B = tf.shape(observations)[0]
        T = tf.shape(observations)[1]

        # Initialize particles at t=0
        particles0, logw0 = self.init_particles(
            batch_size=B,
            init_mean=init_mean,
            init_cov_chol=init_cov_chol,
        )

        # Storage
        particles_list = [particles0]
        logw_list = [logw0]
        
        if return_diagnostics:
            diagnostics_list = []
            times = []

        particles = particles0
        logw = logw0

        # Loop in Python (eager) for clarity; can wrap in tf.function if needed
        for t in range(T.numpy() if isinstance(T, tf.Tensor) else T):
            y_t = observations[:, t, :]  # (B, obs_dim)
            
            if return_diagnostics:
                import time
                start_time = time.time()
                particles, logw, step_diag = self.step(particles, logw, y_t, params=params, 
                                                      return_diagnostics=True)
                end_time = time.time()
                step_diag['step_time'] = end_time - start_time
                times.append(end_time - start_time)
                diagnostics_list.append(step_diag)
            else:
                particles, logw = self.step(particles, logw, y_t, params=params, 
                                          return_diagnostics=False)
            
            particles_list.append(particles)
            logw_list.append(logw)

        particles_seq = tf.stack(particles_list, axis=1)  # (B, T+1, N, d)
        logw_seq = tf.stack(logw_list, axis=1)  # (B, T+1, N)

        if return_diagnostics:
            # Aggregate diagnostics over time
            diagnostics = self._aggregate_diagnostics(diagnostics_list)
            diagnostics['total_time'] = sum(times)
            diagnostics['mean_step_time'] = sum(times) / len(times) if times else 0.0
            
            # Compute RMSE if ground truth provided
            if ground_truth is not None:
                ground_truth = tf.convert_to_tensor(ground_truth, dtype=tf.float32)
                rmse_seq = self._compute_rmse_sequence(particles_seq, logw_seq, ground_truth)
                diagnostics['rmse_sequence'] = rmse_seq
                diagnostics['mean_rmse'] = tf.reduce_mean(rmse_seq)
                diagnostics['final_rmse'] = rmse_seq[-1]
            
            return particles_seq, logw_seq, diagnostics

        return particles_seq, logw_seq
    
    def _aggregate_diagnostics(self, diagnostics_list):
        """Aggregate per-timestep diagnostics into summary statistics.

        Args:
            diagnostics_list (list): List of diagnostic dictionaries from each timestep.

        Returns:
            dict: Aggregated statistics with mean, std, min, and max values.
        """
        if not diagnostics_list:
            return {}
        
        # Extract time series
        aggregated = {}
        keys = diagnostics_list[0].keys()
        
        for key in keys:
            values = [d[key] for d in diagnostics_list]
            
            # Convert to tensor if not already
            if isinstance(values[0], dict):
                # Handle nested dictionaries (like diversity metrics)
                aggregated[key] = {}
                for subkey in values[0].keys():
                    subvalues = tf.stack([v[subkey] for v in values])
                    aggregated[key][subkey + '_mean'] = tf.reduce_mean(subvalues)
                    aggregated[key][subkey + '_std'] = tf.math.reduce_std(subvalues)
                    aggregated[key][subkey + '_min'] = tf.reduce_min(subvalues)
                    aggregated[key][subkey + '_max'] = tf.reduce_max(subvalues)
            elif isinstance(values[0], (int, float)):
                # Scalar values (like step_time)
                aggregated[key + '_mean'] = sum(values) / len(values)
                aggregated[key + '_std'] = tf.math.reduce_std(tf.constant(values))
            else:
                # Tensor values
                values_tensor = tf.stack(values)
                aggregated[key + '_mean'] = tf.reduce_mean(values_tensor)
                aggregated[key + '_std'] = tf.math.reduce_std(values_tensor)
                aggregated[key + '_min'] = tf.reduce_min(values_tensor)
                aggregated[key + '_max'] = tf.reduce_max(values_tensor)
        
        return aggregated
    
    def _compute_rmse_sequence(self, particles_seq, logw_seq, ground_truth):
        """Compute RMSE at each timestep between weighted particle mean and ground truth.

        Args:
            particles_seq (Tensor): Particle sequence of shape (B, T+1, N, d).
            logw_seq (Tensor): Log-weight sequence of shape (B, T+1, N).
            ground_truth (Tensor): True state sequence of shape (B, T+1, d).

        Returns:
            Tensor: RMSE at each timestep (averaged over batch), shape (T+1,).
        """
        # Compute weighted mean: sum_i w_i * x_i
        weights = tf.exp(logw_seq)  # (B, T+1, N)
        weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
        
        # Weighted average of particles
        particle_means = tf.reduce_sum(
            weights[..., None] * particles_seq, axis=2
        )  # (B, T+1, d)
        
        # Compute squared errors
        squared_errors = tf.reduce_sum(
            (particle_means - ground_truth) ** 2, axis=-1
        )  # (B, T+1)
        
        # RMSE averaged over batch
        rmse = tf.sqrt(tf.reduce_mean(squared_errors, axis=0))  # (T+1,)
        
        return rmse


# ----------------------------------------------------------------------
# Example: 1D linear-Gaussian state-space model
# x_t = a * x_{t-1} + q_noise,      q_noise ~ N(0, sigma_q^2)
# y_t = x_t + r_noise,              r_noise ~ N(0, sigma_r^2)
# ----------------------------------------------------------------------


def linear_gaussian_transition(x_prev, params):
    """Linear-Gaussian state transition function.

    Args:
        x_prev (Tensor): Previous states of shape (B, N, 1).
        params (dict): Parameters containing:
            - 'a': Transition coefficient, scalar or shape (B, 1, 1).
            - 'sigma_q': Process noise standard deviation, scalar.

    Returns:
        Tensor: Next states of shape (B, N, 1).
    """
    a = params.get("a", 0.9)
    sigma_q = params.get("sigma_q", 0.5)

    a = tf.convert_to_tensor(a, dtype=tf.float32)
    sigma_q = tf.convert_to_tensor(sigma_q, dtype=tf.float32)

    # Broadcast a and sigma_q
    a = tf.reshape(a, (1, 1, 1))
    sigma_q = tf.reshape(sigma_q, (1, 1, 1))

    eps = tf.random.normal(tf.shape(x_prev), dtype=tf.float32)
    x_new = a * x_prev + sigma_q * eps
    return x_new


def linear_gaussian_log_likelihood(x, y, params):
    """Compute log-likelihood log p(y | x) for linear-Gaussian observation model.

    Assumes y = x + N(0, sigma_r^2).

    Args:
        x (Tensor): States of shape (B, N, 1).
        y (Tensor): Observations of shape (B, 1).
        params (dict): Parameters containing 'sigma_r' (observation noise std dev).

    Returns:
        Tensor: Log-likelihoods of shape (B, N).
    """
    sigma_r = params.get("sigma_r", 0.5)
    sigma_r = tf.convert_to_tensor(sigma_r, dtype=tf.float32)

    # Broadcast y to (B, N, 1)
    y = tf.expand_dims(y, axis=1)  # (B,1,1)
    y = tf.broadcast_to(y, tf.shape(x))  # (B,N,1)

    diff = y - x
    var = sigma_r ** 2
    log_norm_const = -0.5 * tf.math.log(2.0 * np.pi * var)
    log_lik = log_norm_const - 0.5 * (diff ** 2) / var  # (B,N,1)
    return tf.squeeze(log_lik, axis=-1)  # (B,N)


