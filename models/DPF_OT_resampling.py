"""
Differentiable Particle Filter with OT-based Resampling
"""

import tensorflow as tf

# Utility: pairwise squared distances
def pairwise_squared_distances(x, y):
    """Compute pairwise squared Euclidean distances between two sets of points.

    Parameters
    ----------
    x : Tensor
        Points of shape [N, d].
    y : Tensor
        Points of shape [M, d].

    Returns
    -------
    Tensor
        Distance matrix of shape [N, M] where dist[i, j] = ||x[i] - y[j]||^2.
    """
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_sq = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)  # [N, 1]
    y_sq = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)  # [M, 1]
    # Use (x - y)^2 = x^2 + y^2 - 2 x.y
    # x_sq: [N, 1], y_sq^T: [1, M]
    xy = tf.linalg.matmul(x, y, transpose_b=True)  # [N, M]
    dist = x_sq - 2.0 * xy + tf.transpose(y_sq)    # [N, M]
    return tf.maximum(dist, 0.0)



# Sinkhorn OT resampling (entropy-regularized, dual formulation)
def tau_epsilon(a, f, C_vec, epsilon, min_val=1e-12):
    """Compute the tau operator for dual Sinkhorn.

    Computes: Tau_eps(a, f, C_vec) = -epsilon * log(sum_k a_k * exp((f_k - C_vec_k) / epsilon))

    This is the c-transform used in dual Sinkhorn iterations.

    Parameters
    ----------
    a : Tensor
        Source/target mass distribution of shape [N].
    f : Tensor
        Dual variable of shape [N].
    C_vec : Tensor
        Row or column of cost matrix of shape [N].
    epsilon : float
        Regularization parameter.
    min_val : float, optional
        Small constant for numerical stability. Default is 1e-12.

    Returns
    -------
    Tensor
        Scalar tau value.
    """
    # Compute: -eps * log(sum_k a_k * exp((f_k - C_vec_k) / eps))
    # Use log-sum-exp trick for numerical stability
    exponent = (f - C_vec) / epsilon
    max_exp = tf.reduce_max(exponent)
    exp_shifted = tf.exp(exponent - max_exp)
    weighted_sum = tf.reduce_sum(a * exp_shifted)
    result = -epsilon * (tf.math.log(tf.maximum(weighted_sum, min_val)) + max_exp)
    return result


def sinkhorn_ot_resample(particles,
                         weights,
                         epsilon=0.1,
                         n_iters=50,
                         min_val=1e-12,
                         tol=1e-6,
                         return_diagnostics=False):
    """Entropy-regularized OT resampling using Sinkhorn algorithm in dual formulation.

    Implements entropy-regularized optimal transport with barycentric projection
    for differentiable resampling.

    Primal problem: min_P sum_ij P_ij * (C_ij + eps * log(P_ij / (a_i * b_j)))
    Dual problem: max_{f,g} a^T f + b^T g - eps * sum_ij a_i * b_j * exp((f_i + g_j - C_ij) / eps)

    The dual variables f and g are updated via:
        f_i = Tau_eps(b, g, C[i, :])  (with optional damping: 0.5 * (f_i + Tau_eps(...)))
        g_j = Tau_eps(a, f, C[:, j])  (with optional damping: 0.5 * (g_j + Tau_eps(...)))

    Transport matrix recovery:
        - Standard OT: P_ij = a_i * b_j * exp((f_i + g_j - C_ij) / eps)
        - DPF formulation: P_ij = (w_i / N) * exp((f_i + g_j - C_ij) / eps)
        These are mathematically IDENTICAL since a_i = w_i and b_j = 1/N

    Parameters
    ----------
    particles : Tensor
        Current particles of shape [N, d].
    weights : Tensor
        Normalized weights of shape [N] (sum = 1).
    epsilon : float, optional
        Regularization strength for entropic OT. Default is 0.1.
    n_iters : int, optional
        Number of Sinkhorn iterations. Default is 50.
    min_val : float, optional
        Small constant for numerical stability. Default is 1e-12.
    tol : float, optional
        Convergence tolerance for stopping criterion. Default is 1e-6.
    return_diagnostics : bool, optional
        If True, return diagnostic information. Default is False.

    Returns
    -------
    tuple
        If return_diagnostics is False, returns (new_particles, new_weights).
        new_particles has shape [N, d], new_weights has shape [N] (uniform, 1/N each).
        If return_diagnostics is True, returns (new_particles, new_weights, diagnostics dict).
    """
    particles = tf.convert_to_tensor(particles, dtype=tf.float32)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)

    N = tf.shape(particles)[0]
    d = tf.shape(particles)[1]
    N_float = tf.cast(N, tf.float32)

    # Ensure weights are normalized
    weights = tf.maximum(weights, min_val)
    weights = weights / (tf.reduce_sum(weights) + min_val)
    a = weights  # source mass [N]

    # Target mass: uniform over N particles
    b = tf.fill([N], 1.0 / N_float)  # [N]

    # Cost matrix C_ij = ||x_i - x_j||^2
    C = pairwise_squared_distances(particles, particles)  # [N, N]

    # Initialize dual variables f and g to zero
    f = tf.zeros([N], dtype=tf.float32)
    g = tf.zeros([N], dtype=tf.float32)

    # Track convergence if diagnostics requested
    actual_iterations = 0
    convergence_history = [] if return_diagnostics else None

    # Sinkhorn iterations in dual formulation
    for iteration in range(n_iters):
        f_old = f
        g_old = g
        
        # Update f: f_i = 0.5 * (f_i + Tau_eps(b, g, C[i, :]))
        f_new = tf.TensorArray(dtype=tf.float32, size=N, clear_after_read=False)
        for i in tf.range(N):
            C_row_i = C[i, :]  # C[i, :] - costs from particle i to all others
            tau_val = tau_epsilon(b, g, C_row_i, epsilon, min_val)
            f_new = f_new.write(i, 0.5 * (f[i] + tau_val))
        f = f_new.stack()
        
        # Update g: g_j = 0.5 * (g_j + Tau_eps(a, f, C[:, j]))
        g_new = tf.TensorArray(dtype=tf.float32, size=N, clear_after_read=False)
        for j in tf.range(N):
            C_col_j = C[:, j]  # C[:, j] - costs from all particles to particle j
            tau_val = tau_epsilon(a, f, C_col_j, epsilon, min_val)
            g_new = g_new.write(j, 0.5 * (g[j] + tau_val))
        g = g_new.stack()
        
        actual_iterations = iteration + 1
        
        # Check convergence based on change in dual variables
        if iteration > 0:
            f_change = tf.reduce_max(tf.abs(f - f_old))
            g_change = tf.reduce_max(tf.abs(g - g_old))
            
            if return_diagnostics:
                convergence_history.append({
                    'iteration': iteration,
                    'f_change': f_change.numpy(),
                    'g_change': g_change.numpy()
                })
            
            if f_change < tol and g_change < tol:
                break

    
    f_expanded = f[:, None]  # [N, 1]
    g_expanded = g[None, :]  # [1, N]
    exponent = (f_expanded + g_expanded - C) / epsilon
    
    # Compute P using DPF resampling formulation:
    # P_ij = (w_i / N) * exp((f_i + g_j - C_ij) / eps)
    # Equivalently: P_ij = a_i * b_j * exp((f_i + g_j - C_ij) / eps)
    P = a[:, None] * b[None, :] * tf.exp(exponent)  # [N, N]
    
    # Numerical stability: ensure P is non-negative
    P = tf.maximum(P, min_val)

    # Barycentric projection for DPF resampling:
    # new_X_j = (1 / b_j) * sum_i P_ij * X_i
    # Since b_j = 1/N, this gives: new_X_j = N * sum_i P_ij * X_i
    # In matrix form: new_X = diag(1/b) @ P^T @ X = (P^T @ X) / b[:, None]
    PtX = tf.linalg.matmul(P, particles, transpose_a=True)  # P^T @ X, shape [N, d]
    new_particles = PtX / b[:, None]  # divide each row j by b_j = 1/N

    # New weights are uniform (1/N each)
    new_weights = b

    if return_diagnostics:
        # Compute OT distance (Wasserstein distance)
        ot_distance = tf.reduce_sum(P * C)
        
        # Transport plan sparsity
        threshold = 1e-6
        sparsity = tf.reduce_sum(tf.cast(P > threshold, tf.float32)) / tf.cast(N * N, tf.float32)
        
        # Dual variables statistics
        dual_stats = {
            'f_mean': tf.reduce_mean(f),
            'f_std': tf.math.reduce_std(f),
            'g_mean': tf.reduce_mean(g),
            'g_std': tf.math.reduce_std(g),
        }
        
        diagnostics = {
            'sinkhorn_iterations': actual_iterations,
            'converged': actual_iterations < n_iters,
            'ot_distance': ot_distance,
            'transport_plan_sparsity': sparsity,
            'dual_variables': dual_stats,
            'convergence_history': convergence_history,
            'epsilon': epsilon,
        }
        
        return new_particles, new_weights, diagnostics

    return new_particles, new_weights


# Differentiable Particle Filter with OT resampling
class DPF_OT(tf.Module):
    """Differentiable Particle Filter with Optimal Transport resampling.

    Uses entropy-regularized optimal transport (Sinkhorn algorithm) for
    differentiable particle resampling with barycentric projection.
    """

    def __init__(self,
                 N_particles,
                 state_dim,
                 transition_fn,
                 obs_loglik_fn,
                 epsilon=0.1,
                 sinkhorn_iters=50,
                 name=None):
        """Initialize the Differentiable Particle Filter with OT resampling.

        Parameters
        ----------
        N_particles : int
            Number of particles N.
        state_dim : int
            Dimensionality of state space d.
        transition_fn : callable
            State transition function (particles, t) -> new_particles.
        obs_loglik_fn : callable
            Observation log-likelihood function (particles, y_t, t) -> log_liks of shape [N].
        epsilon : float, optional
            Entropy regularization for OT. Default is 0.1.
        sinkhorn_iters : int, optional
            Number of Sinkhorn iterations. Default is 50.
        name : str, optional
            Name for the module. Default is None.
        """
        super().__init__(name=name)
        self.N = N_particles
        self.d = state_dim
        self.transition_fn = transition_fn
        self.obs_loglik_fn = obs_loglik_fn
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters
    

    # Utility functions for diagnostics
    @staticmethod
    def compute_ess(weights):
        """Compute Effective Sample Size (ESS) from weights.

        ESS = 1 / sum(w_i^2)

        Parameters
        ----------
        weights : Tensor
            Normalized weights of shape (N,) or (B, N).

        Returns
        -------
        Tensor
            Scalar or (B,) Effective Sample Size.
        """
        # Ensure weights are normalized
        if weights.shape.rank == 1:
            weights = weights / (tf.reduce_sum(weights) + 1e-12)
            ess = 1.0 / tf.reduce_sum(weights ** 2)
        else:  # (B, N)
            weights = weights / (tf.reduce_sum(weights, axis=-1, keepdims=True) + 1e-12)
            ess = 1.0 / tf.reduce_sum(weights ** 2, axis=-1)
        return ess
    
    @staticmethod
    def compute_weight_entropy(weights):
        """Compute entropy of weight distribution.

        H(w) = -sum(w_i * log(w_i))

        Parameters
        ----------
        weights : Tensor
            Normalized weights of shape (N,) or (B, N).

        Returns
        -------
        Tensor
            Scalar or (B,) entropy.
        """
        weights = weights / (tf.reduce_sum(weights, axis=-1, keepdims=True) + 1e-12)
        entropy = -tf.reduce_sum(weights * tf.math.log(weights + 1e-10), axis=-1)
        return entropy
    
    @staticmethod
    def compute_particle_diversity(particles):
        """Compute particle diversity metrics.

        Parameters
        ----------
        particles : Tensor
            Particle states of shape (N, d) or (B, N, d).

        Returns
        -------
        dict
            Dictionary with diversity metrics including 'mean_pairwise_dist'
            and 'particle_spread'.
        """
        if particles.shape.rank == 2:
            # Add batch dimension
            particles = particles[None, :, :]
            squeeze_output = True
        else:
            squeeze_output = False
        
        B = tf.shape(particles)[0]
        N = tf.shape(particles)[1]
        
        # Pairwise distances
        p1 = tf.expand_dims(particles, axis=2)  # (B, N, 1, d)
        p2 = tf.expand_dims(particles, axis=1)  # (B, 1, N, d)
        diff = p1 - p2
        distances = tf.norm(diff, axis=-1)  # (B, N, N)
        
        mask = 1.0 - tf.eye(N, dtype=tf.float32)
        masked_distances = distances * mask
        mean_dist = tf.reduce_sum(masked_distances, axis=[1, 2]) / tf.cast(N * (N - 1), tf.float32)
        
        # Particle spread
        particle_mean = tf.reduce_mean(particles, axis=1, keepdims=True)
        centered = particles - particle_mean
        cov = tf.reduce_mean(centered[:, :, :, None] * centered[:, :, None, :], axis=1)
        spread = tf.linalg.trace(cov)
        
        result = {
            'mean_pairwise_dist': mean_dist[0] if squeeze_output else mean_dist,
            'particle_spread': spread[0] if squeeze_output else spread
        }
        return result

    def init_particles(self, mean, cov_chol):
        """Initialize particles from a Gaussian N(mean, cov).

        Parameters
        ----------
        mean : Tensor
            Mean vector of shape [d].
        cov_chol : Tensor
            Cholesky factor of covariance of shape [d, d].

        Returns
        -------
        tuple of Tensor
            Particles of shape [N, d] and uniform weights of shape [N].
        """
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        cov_chol = tf.convert_to_tensor(cov_chol, dtype=tf.float32)
        N = self.N
        d = self.d

        eps = tf.random.normal(shape=(N, d), dtype=tf.float32)
        particles = mean[None, :] + tf.linalg.matmul(eps, cov_chol, transpose_b=True)

        weights = tf.fill([N], 1.0 / tf.cast(N, tf.float32))
        return particles, weights

    def step(self, particles, weights, y_t, t=None, return_diagnostics=False):
        """Perform one step of the DPF with OT resampling.

        Parameters
        ----------
        particles : Tensor
            Particles of shape [N, d].
        weights : Tensor
            Weights of shape [N].
        y_t : Tensor
            Observation at time t (shape user-defined).
        t : int, optional
            Time index. Default is None.
        return_diagnostics : bool, optional
            If True, return diagnostic metrics. Default is False.

        Returns
        -------
        tuple
            If return_diagnostics is False, returns (new_particles, new_weights).
            new_particles has shape [N, d], new_weights has shape [N].
            If return_diagnostics is True, also returns diagnostics dict as third element.
        """
        # Compute diagnostics before resampling (if requested)
        if return_diagnostics:
            import time
            start_time = time.time()
            ess_before = self.compute_ess(weights)
            entropy_before = self.compute_weight_entropy(weights)
            diversity_before = self.compute_particle_diversity(particles)

        # 1) Propagate particles through transition model
        pred_particles = self.transition_fn(particles, t)  # [N, d]

        # 2) Update weights using observation likelihood
        log_liks = self.obs_loglik_fn(pred_particles, y_t, t)  # [N]
        log_liks = tf.convert_to_tensor(log_liks, dtype=tf.float32)

        # unnormalized weights: w_prev * exp(log_lik)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        unnorm_w = weights * tf.exp(log_liks)

        # normalize
        unnorm_w = tf.maximum(unnorm_w, 0.0)
        norm_const = tf.reduce_sum(unnorm_w) + 1e-12
        new_weights = unnorm_w / norm_const

        # 3) OT-Sinkhorn resampling (differentiable)
        if return_diagnostics:
            res_particles, res_weights, resample_diag = sinkhorn_ot_resample(
                pred_particles,
                new_weights,
                epsilon=self.epsilon,
                n_iters=self.sinkhorn_iters,
                return_diagnostics=True
            )
            
            # Compute diagnostics after resampling
            ess_after = self.compute_ess(res_weights)
            entropy_after = self.compute_weight_entropy(res_weights)
            diversity_after = self.compute_particle_diversity(res_particles)
            
            end_time = time.time()
            
            diagnostics = {
                'ess_before': ess_before,
                'ess_after': ess_after,
                'entropy_before': entropy_before,
                'entropy_after': entropy_after,
                'diversity_before': diversity_before,
                'diversity_after': diversity_after,
                'max_weight_before': tf.reduce_max(new_weights),
                'step_time': end_time - start_time,
                **resample_diag
            }
            
            return res_particles, res_weights, diagnostics
        else:
            res_particles, res_weights = sinkhorn_ot_resample(
                pred_particles,
                new_weights,
                epsilon=self.epsilon,
                n_iters=self.sinkhorn_iters,
                return_diagnostics=False
            )

        # res_weights are uniform; we return them
        return res_particles, res_weights

    def run_filter(self, y_seq, mean0, cov0_chol, return_diagnostics=False, ground_truth=None):
        """Run the DPF over a sequence of observations.

        Parameters
        ----------
        y_seq : list or Tensor
            Sequence of observations [T, ...].
        mean0 : Tensor
            Initial mean of shape [d].
        cov0_chol : Tensor
            Initial covariance Cholesky factor of shape [d, d].
        return_diagnostics : bool, optional
            If True, return detailed diagnostic metrics. Default is False.
        ground_truth : Tensor, optional
            True states of shape [T+1, d] for computing RMSE. Default is None.

        Returns
        -------
        tuple
            If return_diagnostics is False, returns (particles_seq, weights_seq).
            particles_seq is a list of length T with tensors [N, d],
            weights_seq is a list of length T with tensors [N].
            If return_diagnostics is True, also returns diagnostics dict as third element.
        """
        particles, weights = self.init_particles(mean0, cov0_chol)
        particles_seq = []
        weights_seq = []
        
        if return_diagnostics:
            diagnostics_list = []
            times = []

        T = len(y_seq)
        for t in range(T):
            y_t = y_seq[t]
            
            if return_diagnostics:
                import time
                start_time = time.time()
                particles, weights, step_diag = self.step(particles, weights, y_t, t=t, 
                                                          return_diagnostics=True)
                end_time = time.time()
                times.append(end_time - start_time)
                diagnostics_list.append(step_diag)
            else:
                particles, weights = self.step(particles, weights, y_t, t=t, 
                                              return_diagnostics=False)
            
            particles_seq.append(particles)
            weights_seq.append(weights)

        if return_diagnostics:
            # Aggregate diagnostics over time
            diagnostics = self._aggregate_diagnostics(diagnostics_list)
            diagnostics['total_time'] = sum(times)
            diagnostics['mean_step_time'] = sum(times) / len(times) if times else 0.0
            
            # Compute RMSE if ground truth provided
            if ground_truth is not None:
                ground_truth = tf.convert_to_tensor(ground_truth, dtype=tf.float32)
                rmse_seq = self._compute_rmse_sequence(particles_seq, weights_seq, ground_truth)
                diagnostics['rmse_sequence'] = rmse_seq
                diagnostics['mean_rmse'] = tf.reduce_mean(rmse_seq)
                diagnostics['final_rmse'] = rmse_seq[-1]
            
            return particles_seq, weights_seq, diagnostics

        return particles_seq, weights_seq
    
    def _aggregate_diagnostics(self, diagnostics_list):
        """Aggregate per-timestep diagnostics into summary statistics.

        Parameters
        ----------
        diagnostics_list : list
            List of diagnostic dictionaries from each timestep.

        Returns
        -------
        dict
            Aggregated statistics with mean, std, min, and max values.
        """
        if not diagnostics_list:
            return {}
        
        aggregated = {}
        keys = diagnostics_list[0].keys()
        
        for key in keys:
            values = [d[key] for d in diagnostics_list]
            
            if key == 'convergence_history':
                # Skip convergence history in aggregation
                continue
            elif isinstance(values[0], dict):
                # Handle nested dictionaries
                aggregated[key] = {}
                for subkey in values[0].keys():
                    subvalues = tf.stack([v[subkey] for v in values])
                    aggregated[key][subkey + '_mean'] = tf.reduce_mean(subvalues)
                    aggregated[key][subkey + '_std'] = tf.math.reduce_std(subvalues)
            elif isinstance(values[0], bool):
                # Boolean values (like converged)
                aggregated[key + '_rate'] = sum(values) / len(values)
            elif isinstance(values[0], (int, float)) or (hasattr(values[0], 'numpy') and values[0].shape.rank == 0):
                # Scalar values
                values_list = [float(v) if hasattr(v, 'numpy') else v for v in values]
                aggregated[key + '_mean'] = sum(values_list) / len(values_list)
                aggregated[key + '_std'] = tf.math.reduce_std(tf.constant(values_list, dtype=tf.float32))
                aggregated[key + '_min'] = min(values_list)
                aggregated[key + '_max'] = max(values_list)
        
        return aggregated
    
    def _compute_rmse_sequence(self, particles_seq, weights_seq, ground_truth):
        """Compute RMSE at each timestep between weighted particle mean and ground truth.

        Parameters
        ----------
        particles_seq: list 
            List of particle tensors of shape [N, d].
        weights_seq: list
            List of weight tensors of shape [N].
        ground_truth: Tensor
            True state sequence of shape [T+1, d] (includes t=0).

        Returns
        ----------
        Tensor
            RMSE at each timestep, shape [T].
        """
        rmse_list = []
        
        for t, (particles, weights) in enumerate(zip(particles_seq, weights_seq)):
            # Weighted mean
            weights = weights / (tf.reduce_sum(weights) + 1e-12)
            particle_mean = tf.reduce_sum(weights[:, None] * particles, axis=0)  # [d]
            
            # Compare to ground truth at t+1 (since particles_seq starts after first step)
            true_state = ground_truth[t + 1]  # [d]
            
            # Squared error
            squared_error = tf.reduce_sum((particle_mean - true_state) ** 2)
            rmse_list.append(tf.sqrt(squared_error))
        
        return tf.stack(rmse_list)


# # ============================================
# # Example transition and observation models
# # ============================================
# def example_transition_fn(particles, t):
#     """Simple linear Gaussian dynamics transition function.

#     Implements: x_{t+1} = x_t + noise, noise ~ N(0, I * sigma^2)

#     Args:
#         particles (Tensor): Current particles of shape [N, d].
#         t (int): Time index (unused in this simple model).

#     Returns:
#         Tensor: Next particles of shape [N, d].
#     """
#     sigma = 0.1
#     noise = tf.random.normal(tf.shape(particles), stddev=sigma)
#     return particles + noise


# def example_obs_loglik_fn(particles, y_t, t):
#     """Simple Gaussian observation model log-likelihood.

#     Implements observation model: y_t = x_t + noise, noise ~ N(0, I * tau^2)
#     Log-likelihood: log p(y_t | x_t) = -0.5 * ||y_t - x_t||^2 / tau^2 + const
#     Constants can be dropped for weighting purposes.

#     Args:
#         particles (Tensor): Particles of shape [N, d].
#         y_t (Tensor): Observation at time t of shape [d].
#         t (int): Time index (unused in this simple model).

#     Returns:
#         Tensor: Log-likelihoods of shape [N].
#     """
#     tau = 0.2
#     y_t = tf.convert_to_tensor(y_t, dtype=tf.float32)
#     # broadcast y_t over particles if needed
#     diff = particles - y_t[None, :]
#     sq_norm = tf.reduce_sum(tf.square(diff), axis=1)  # [N]
#     log_lik = -0.5 * sq_norm / (tau ** 2)
#     return log_lik
