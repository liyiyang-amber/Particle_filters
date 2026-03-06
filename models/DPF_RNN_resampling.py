"""
Differentiable Particle Filter with RNN-based resampling.

Implements a differentiable particle filter where the resampling step is
replaced by an RNN that learns a differentiable assignment distribution over
ancestor particles.  An optional baseline mode substitutes a simple
soft weight-based resampling (no network required) for ablation or comparison.

Classes
-------
DifferentiableParticleFilterRNN – Main filter class (``tf.Module`` subclass).
"""

import numpy as np
import tensorflow as tf


class DifferentiableParticleFilterRNN(tf.Module):
    """Differentiable Particle Filter with RNN-based resampling.

    An RNN network processes the current particle ensemble and importance
    weights to produce a learned assignment distribution over ancestor
    particles.  The new particles are generated as barycentres of the old
    particles under this distribution, making the resampling step fully
    differentiable.

    When ``use_baseline_resampling=True`` the RNN is not built and a simple
    soft weight-based resampling (Gumbel-Softmax) is used instead, providing
    a fair comparison without requiring training.
    """

    def __init__(
        self,
        n_particles,
        state_dim,
        transition_fn,
        log_likelihood_fn,
        rnn_type='lstm',
        rnn_hidden_dim=64,
        rnn_num_layers=1,
        use_weight_features=True,
        use_particle_features=True,
        temperature=1.0,
        use_baseline_resampling=False,
        name=None,
    ):
        """Initialize the Differentiable Particle Filter with RNN resampling.

        Parameters
        ----------
        n_particles : int
            Number of particles N.
        state_dim : int
            Dimension of latent state.
        transition_fn : callable
            State transition function (x_prev, params) -> x_pred.
            x_prev has shape (B, N, state_dim), returns shape (B, N, state_dim).
        log_likelihood_fn : callable
            Log-likelihood function (x, y, params) -> log p(y | x).
            x has shape (B, N, state_dim), y has shape (B, obs_dim), returns shape (B, N).
        rnn_type : str, optional
            Type of RNN cell, 'lstm' or 'gru'. Default is 'lstm'.
        rnn_hidden_dim : int, optional
            Hidden dimension of RNN. Default is 64.
        rnn_num_layers : int, optional
            Number of RNN layers. Default is 1.
        use_weight_features : bool, optional
            Whether to include weights in RNN input. Default is True.
        use_particle_features : bool, optional
            Whether to include particle states in RNN input. Default is True.
        temperature : float, optional
            Softmax temperature for assignment probabilities. Default is 1.0.
        use_baseline_resampling : bool, optional
            If True, use simple weight-based resampling instead of RNN
            (for fair comparison without training). Default is False.
        name : str, optional
            Name for the module. Default is None.
        """
        super().__init__(name=name)
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.transition_fn = transition_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.rnn_type = rnn_type.lower()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.use_weight_features = use_weight_features
        self.use_particle_features = use_particle_features
        self.temperature = temperature
        self.use_baseline_resampling = use_baseline_resampling

        if not use_baseline_resampling:
            # Build the RNN resampling network
            self._build_rnn_resampler()
            
            # Initialize with sensible weights (importance sampling approximation)
            self._initialize_sensible_weights()
        else:
            print(f"✓ Using baseline resampling (weight-based soft resampling)")
            print(f"  This provides a fair comparison without requiring RNN training")


    # RNN Resampler Network
    def _build_rnn_resampler(self):
        """Build the RNN network for learning resampling strategy.

        Constructs RNN layers (LSTM or GRU) and an output layer that maps
        the final hidden state to assignment logits over old particles.
        """
        
        # Compute input dimension
        input_dim = 0
        if self.use_weight_features:
            input_dim += 1  # log weight or normalized weight
        if self.use_particle_features:
            input_dim += self.state_dim  # particle state
        
        if input_dim == 0:
            raise ValueError("Must use at least one of weight_features or particle_features")
        
        # Create RNN layers
        self.rnn_cells = []
        for layer_idx in range(self.rnn_num_layers):
            layer_input_dim = input_dim if layer_idx == 0 else self.rnn_hidden_dim
            
            if self.rnn_type == 'lstm':
                cell = tf.keras.layers.LSTMCell(
                    self.rnn_hidden_dim,
                    name=f'lstm_cell_{layer_idx}'
                )
            elif self.rnn_type == 'gru':
                cell = tf.keras.layers.GRUCell(
                    self.rnn_hidden_dim,
                    name=f'gru_cell_{layer_idx}'
                )
            else:
                raise ValueError(f"Unknown RNN type: {self.rnn_type}. Use 'lstm' or 'gru'")
            
            self.rnn_cells.append(cell)
        
        # Output layer: maps hidden state to resampling logits
        # For each new particle, output N logits (probabilities over ancestors)
        self.output_layer = tf.keras.layers.Dense(
            self.n_particles,
            activation=None,  # Will apply softmax later
            name='resampling_output'
        )
    
    def _initialize_sensible_weights(self):
        """Initialize RNN with sensible weights to approximate importance sampling.

        Strategy: Initialize the output layer with very small random weights to create
        a safe baseline behavior. With small weights, the RNN initially produces nearly
        uniform assignments (high entropy). This is a conservative default that doesn't
        strongly favor any particles. Through training, the network learns to adjust
        based on weights and states.

        """
        # Build the network by doing a dummy forward pass
        dummy_particles = tf.zeros((1, self.n_particles, self.state_dim))
        dummy_log_weights = tf.zeros((1, self.n_particles))
        
        # Create a test case with known weights to verify behavior
        test_weights = tf.constant([[0.5, 0.3, 0.2]], dtype=tf.float32)  # Example for N=3
        test_log_weights = tf.math.log(test_weights)
        
        # Do a forward pass to build all layers
        if self.n_particles == 3:
            # Use small test case if available
            test_particles = tf.zeros((1, 3, self.state_dim))
            _ = self._rnn_resample(test_particles, test_log_weights)
        else:
            # Use full size
            _ = self._rnn_resample(dummy_particles, dummy_log_weights)
        
 
        output_weights = self.output_layer.get_weights()
        if len(output_weights) == 2:  # kernel and bias
            kernel, bias = output_weights
            # Very small random kernel (near-zero for minimal initial influence)
            new_kernel = np.random.randn(*kernel.shape).astype(np.float32) * 0.001
            # Zero bias (uniform initial assignments via softmax)
            new_bias = np.zeros_like(bias).astype(np.float32)
            self.output_layer.set_weights([new_kernel, new_bias])
        
        
        print(f"✓ RNN initialized with small random weights (safe baseline)")
        print(f"  Initial behavior: Nearly uniform assignments (high entropy)")
        print(f"  Note: Training is still recommended for optimal performance")

    def _compute_rnn_features(self, particles, log_weights, target_particle_idx=None):
        """Compute input features for the RNN from particles and weights.

        Parameters
        ----------
        particles : Tensor, shape (B, N, state_dim)
            Particle states.
        log_weights : Tensor, shape (B, N)
            Log-weights (need not be normalised).
        target_particle_idx : int, optional
            Index of the new particle being generated.  When provided, a one-hot
            encoding of this index is appended to the features so the RNN can
            distinguish which output particle it is computing.  Default is
            ``None``.

        Returns
        -------
        Tensor, shape (B, N, feature_dim)
            Concatenated feature vectors for each old particle.
        """
        batch_size = tf.shape(particles)[0]
        N = self.n_particles
        
        features = []
        
        if self.use_weight_features:
            # Normalized weights (not log-weights)
            weights = tf.exp(log_weights)  # (B, N)
            weights_expanded = tf.expand_dims(weights, axis=-1)  # (B, N, 1)
            features.append(weights_expanded)
        
        if self.use_particle_features:
            # Particle states
            features.append(particles)  # (B, N, state_dim)
        
        # Add target particle index as one-hot encoding
        # This allows the RNN to differentiate which new particle it's generating
        if target_particle_idx is not None:
            # Create one-hot encoding: (N,)
            idx_onehot = tf.one_hot(target_particle_idx, depth=N, dtype=tf.float32)
            # Broadcast to (B, N, N) - repeat for each old particle in the sequence
            idx_onehot = tf.tile(idx_onehot[None, None, :], [batch_size, N, 1])
            features.append(idx_onehot)
        
        # Concatenate all features
        combined_features = tf.concat(features, axis=-1)  # (B, N, feature_dim)
        
        return combined_features
    
    def _baseline_resample(self, particles, log_weights):
        """Simple baseline resampling using soft weight-based assignments.

        Provides a drop-in replacement for the RNN resampler that does not
        require training.  Assignment probabilities are obtained via a
        Gumbel-Softmax applied to the normalised importance weights.

        Parameters
        ----------
        particles : Tensor, shape (B, N, state_dim)
            Current particle states.
        log_weights : Tensor, shape (B, N)
            Current log-weights.

        Returns
        -------
        new_particles : Tensor, shape (B, N, state_dim)
            Resampled particles via barycentric projection.
        assignment_probs : Tensor, shape (B, N, N)
            Assignment probability matrix where entry ``[b, i, j]`` is the
            probability that new particle ``i`` originates from old particle
            ``j``.
        """
        batch_size = tf.shape(particles)[0]
        N = self.n_particles
        
        # Normalize log weights to probabilities
        weights = tf.exp(log_weights)  # (B, N)
        weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)  # (B, N)
        
        # Create assignment matrix where each row is proportional to weights
        # Shape: (B, N, N) where assignment_matrix[b, i, j] = probability that
        # new particle i comes from old particle j
        
        # For baseline: use weights directly (with temperature)
        log_probs = tf.math.log(weights + 1e-10) / self.temperature  # (B, N)
        log_probs_expanded = tf.expand_dims(log_probs, axis=1)  # (B, 1, N)
        log_probs_tiled = tf.tile(log_probs_expanded, [1, N, 1])  # (B, N, N)
        
        # Add small Gumbel noise for diversity (similar to Gumbel-Softmax)
        gumbel_noise = -tf.math.log(-tf.math.log(
            tf.random.uniform(tf.shape(log_probs_tiled), minval=1e-10, maxval=1.0)
        ))
        log_probs_noisy = log_probs_tiled + gumbel_noise * 0.1  # Small noise
        
        # Softmax to get assignment probabilities
        assignment_probs = tf.nn.softmax(log_probs_noisy, axis=-1)  # (B, N, N)
        
        # Barycentric projection
        new_particles = tf.einsum("bij,bjd->bid", assignment_probs, particles)
        
        return new_particles, assignment_probs

    def _rnn_resample(self, particles, log_weights):
        """Perform RNN-based differentiable resampling.

        For each new particle ``i``, the RNN processes the sequence of old
        particles (with features) and produces logits over ancestor indices.
        The assignment probabilities are obtained by a softmax with
        ``self.temperature``, and new particles are the resulting barycentres.

        Parameters
        ----------
        particles : Tensor, shape (B, N, state_dim)
            Current particle states.
        log_weights : Tensor, shape (B, N)
            Current log-weights.

        Returns
        -------
        new_particles : Tensor, shape (B, N, state_dim)
            Resampled particles via barycentric projection.
        assignment_probs : Tensor, shape (B, N, N)
            Assignment probability matrix where entry ``[b, i, j]`` is the
            probability that new particle ``i`` originates from old particle
            ``j``.
        """
        # Use baseline resampling if flag is set
        if self.use_baseline_resampling:
            return self._baseline_resample(particles, log_weights)
        
        batch_size = tf.shape(particles)[0]
        N = self.n_particles
        
        # Process each particle through RNN to get assignment logits
        assignment_logits_list = []
        
        for i in range(N):
            # Compute input features for RNN, including target particle index
            features = self._compute_rnn_features(particles, log_weights, target_particle_idx=i)  # (B, N, feature_dim)
            
            # For i-th new particle, process all old particles through RNN
            # Input: features of all old particles + one-hot encoding of target index i
            # Output: logits over old particles (B, N)
            
            # Initialize fresh RNN states for this new particle
            states = []
            for cell in self.rnn_cells:
                if self.rnn_type == 'lstm':
                    # LSTM has (h, c) state
                    h = tf.zeros((batch_size, self.rnn_hidden_dim))
                    c = tf.zeros((batch_size, self.rnn_hidden_dim))
                    states.append([h, c])
                else:  # GRU
                    # GRU has single h state
                    h = tf.zeros((batch_size, self.rnn_hidden_dim))
                    states.append([h])
            
            # Run through RNN layers
            rnn_input = features  # (B, N, feature_dim)
            
            # Process sequence through all RNN layers
            for layer_idx, cell in enumerate(self.rnn_cells):
                # For TensorFlow RNN cells, we need to process time steps manually
                outputs = []
                layer_states = states[layer_idx]
                
                # Process each particle (time step) through the cell
                for t in range(N):
                    particle_input = rnn_input[:, t, :]  # (B, feature_dim or hidden_dim)
                    output, layer_states = cell(particle_input, layer_states)
                    outputs.append(output)
                
                # Stack outputs: (B, N, hidden_dim)
                rnn_input = tf.stack(outputs, axis=1)
                states[layer_idx] = layer_states
            
            # Final RNN output: (B, N, hidden_dim)
            rnn_output = rnn_input
            
            # Use the final time step's hidden state to generate assignment logits
            # The final hidden state encodes information about all particles processed
            final_output = rnn_output[:, -1, :]  # (B, hidden_dim)
            
            # Generate logits for this new particle
            logits_i = self.output_layer(final_output)  # (B, N)
            assignment_logits_list.append(logits_i)
        
        # Stack all logits: (N, B, N) -> (B, N, N)
        assignment_logits = tf.stack(assignment_logits_list, axis=1)  # (B, N, N)
        
        # Apply softmax with temperature to get assignment probabilities
        assignment_probs = tf.nn.softmax(assignment_logits / self.temperature, axis=-1)
        
        # Barycentric projection: new_particles[i] = sum_j assignment_probs[i,j] * particles[j]
        new_particles = tf.einsum("bij,bjd->bid", assignment_probs, particles)
        
        return new_particles, assignment_probs

    
    # Utilities
    @staticmethod
    def _log_normalize(log_w, axis=-1, keepdims=False):
        """Normalise log-weights along a given axis.

        Parameters
        ----------
        log_w : Tensor
            Log-weights to normalise.
        axis : int, optional
            Axis along which to normalise.  Default is ``-1``.
        keepdims : bool, optional
            If ``True``, keep the reduced axis as a size-1 dimension.
            Default is ``False``.

        Returns
        -------
        log_w_norm : Tensor
            Normalised log-weights (same shape as input).
        log_z : Tensor
            Log normalisation constant (shape with ``axis`` removed or kept).
        """
        log_z = tf.reduce_logsumexp(log_w, axis=axis, keepdims=True)
        log_w_norm = log_w - log_z
        if not keepdims:
            log_z = tf.squeeze(log_z, axis=axis)
        return log_w_norm, log_z

    @staticmethod
    def compute_ess(log_weights):
        """Compute Effective Sample Size (ESS) from log-weights.

        Parameters
        ----------
        log_weights : Tensor, shape (B, N)
            Log-weights (need not be normalised).

        Returns
        -------
        Tensor, shape (B,)
            Effective Sample Size per batch element.
        """
        # Normalize weights
        weights = tf.exp(log_weights)  # (B, N)
        weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
        
        # ESS = 1 / sum(w^2)
        ess = 1.0 / tf.reduce_sum(weights ** 2, axis=-1)
        return ess
    
    @staticmethod
    def compute_weight_entropy(log_weights):
        """Compute the Shannon entropy of the weight distribution.

        Parameters
        ----------
        log_weights : Tensor, shape (B, N)
            Log-weights (need not be normalised).

        Returns
        -------
        Tensor, shape (B,)
            Entropy per batch element.
        """
        # Normalize weights
        weights = tf.exp(log_weights)  # (B, N)
        weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        weights = tf.clip_by_value(weights, eps, 1.0)
        
        # Entropy: H = -sum(w * log(w))
        entropy = -tf.reduce_sum(weights * tf.math.log(weights), axis=-1)
        return entropy

    
    # Initialization
    
    def init_particles(self, batch_size, init_mean, init_cov_chol):
        """Initialise particles from a Gaussian prior N(init_mean, init_cov).

        Parameters
        ----------
        batch_size : int
            Number of parallel sequences B.
        init_mean : Tensor, shape (state_dim,) or (B, state_dim)
            Prior mean vector.
        init_cov_chol : Tensor, shape (state_dim, state_dim) or (B, state_dim, state_dim)
            Lower Cholesky factor of the prior covariance.

        Returns
        -------
        particles : Tensor, shape (B, N, state_dim)
            Initial particle ensemble.
        log_weights : Tensor, shape (B, N)
            Uniform initial log-weights ``log(1/N)``.
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

    
    # Single filtering step
    
    def step(self, particles, log_weights, observation, params=None, return_ess=False):
        """Perform one step of the differentiable particle filter with RNN resampling.

        Parameters
        ----------
        particles : Tensor, shape (B, N, d)
            Previous particle states.
        log_weights : Tensor, shape (B, N)
            Previous log-weights (need not be normalised).
        observation : Tensor, shape (B, obs_dim)
            Current observation.
        params : dict, optional
            Parameters forwarded to the transition and likelihood functions.
            Default is ``None`` (empty dict).
        return_ess : bool, optional
            If ``True``, return ESS and entropy before and after resampling as
            a fourth element.  Default is ``False``.

        Returns
        -------
        new_particles : Tensor, shape (B, N, d)
            Updated particle states.
        new_log_weights : Tensor, shape (B, N)
            Uniform log-weights after resampling.
        assignment_matrix : Tensor, shape (B, N, N)
            Assignment probability matrix from the RNN resampler.
        ess_dict : dict
            Keys ``'ess_before'``, ``'ess_after'``, ``'entropy_before'``,
            ``'entropy_after'`` (only returned when ``return_ess=True``).
        """
        if params is None:
            params = {}

        # 1) Propagation: x_t^{(i)} ~ p(x_t | x_{t-1}^{(i)})
        pred_particles = self.transition_fn(particles, params)  # (B, N, d)

        # 2) Importance weights update: w_t^{(i)} ∝ w_{t-1}^{(i)} * p(y_t | x_t^{(i)})
        log_lik = self.log_likelihood_fn(pred_particles, observation, params)  # (B, N)
        log_weights = log_weights + log_lik
        log_weights, _ = self._log_normalize(log_weights, axis=-1, keepdims=False)

        # Compute ESS and entropy before resampling (if requested)
        if return_ess:
            ess_before = self.compute_ess(log_weights)
            entropy_before = self.compute_weight_entropy(log_weights)
        
        # 3) RNN-based resampling
        new_particles, assignment_matrix = self._rnn_resample(pred_particles, log_weights)

        # 4) Reset weights to uniform after resampling
        new_log_weights = tf.math.log(
            tf.fill((tf.shape(particles)[0], self.n_particles), 1.0 / self.n_particles)
        )
        
        # Compute ESS and entropy after resampling (if requested)
        if return_ess:
            ess_after = self.compute_ess(new_log_weights)
            entropy_after = self.compute_weight_entropy(new_log_weights)
            ess_dict = {
                'ess_before': ess_before, 
                'ess_after': ess_after,
                'entropy_before': entropy_before,
                'entropy_after': entropy_after
            }
            return new_particles, new_log_weights, assignment_matrix, ess_dict

        return new_particles, new_log_weights, assignment_matrix

    
    # Full filtering over a sequence
    def filter(self, observations, init_mean, init_cov_chol, params=None, return_ess=False):
        """Run the differentiable RNN particle filter over a sequence of observations.

        Parameters
        ----------
        observations : Tensor, shape (B, T, obs_dim)
            Full observation sequence.
        init_mean : Tensor, shape (state_dim,) or (B, state_dim)
            Initial prior mean; see ``init_particles``.
        init_cov_chol : Tensor, shape (state_dim, state_dim) or (B, state_dim, state_dim)
            Lower Cholesky factor of the initial prior covariance; see
            ``init_particles``.
        params : dict, optional
            Parameters forwarded to transition and likelihood functions.
            Default is ``None`` (empty dict).
        return_ess : bool, optional
            If ``True``, also return ESS and entropy statistics over time.
            Default is ``False``.

        Returns
        -------
        particles_seq : Tensor, shape (B, T+1, N, d)
            Particle ensemble at every time step (t = 0 … T).
        logw_seq : Tensor, shape (B, T+1, N)
            Log-weights at every time step.
        assignment_matrices : Tensor, shape (B, T, N, N)
            Assignment matrices produced by the RNN at each step.
        ess_stats : dict
            ESS and entropy statistics over time (only returned when
            ``return_ess=True``).
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
        assignment_matrices = []
        
        if return_ess:
            ess_before_list = []
            ess_after_list = []
            entropy_before_list = []
            entropy_after_list = []

        particles = particles0
        logw = logw0

        # Loop over time steps
        for t in range(T.numpy() if isinstance(T, tf.Tensor) else T):
            y_t = observations[:, t, :]  # (B, obs_dim)
            
            if return_ess:
                particles, logw, assignment, ess_dict = self.step(
                    particles, logw, y_t, params=params, return_ess=True
                )
                ess_before_list.append(ess_dict['ess_before'])
                ess_after_list.append(ess_dict['ess_after'])
                entropy_before_list.append(ess_dict['entropy_before'])
                entropy_after_list.append(ess_dict['entropy_after'])
            else:
                particles, logw, assignment = self.step(particles, logw, y_t, params=params)
            
            particles_list.append(particles)
            logw_list.append(logw)
            assignment_matrices.append(assignment)

        particles_seq = tf.stack(particles_list, axis=1)  # (B, T+1, N, d)
        logw_seq = tf.stack(logw_list, axis=1)  # (B, T+1, N)
        assignment_matrices = tf.stack(assignment_matrices, axis=1)  # (B, T, N, N)

        if return_ess:
            ess_before = tf.stack(ess_before_list, axis=1)  # (B, T)
            ess_after = tf.stack(ess_after_list, axis=1)  # (B, T)
            entropy_before = tf.stack(entropy_before_list, axis=1)  # (B, T)
            entropy_after = tf.stack(entropy_after_list, axis=1)  # (B, T)
            ess_stats = {
                'ess_before_resampling': ess_before,
                'ess_after_resampling': ess_after,
                'entropy_before_resampling': entropy_before,
                'entropy_after_resampling': entropy_after,
                'mean_ess_before': tf.reduce_mean(ess_before),
                'mean_ess_after': tf.reduce_mean(ess_after),
                'mean_entropy_before': tf.reduce_mean(entropy_before),
                'mean_entropy_after': tf.reduce_mean(entropy_after),
                'min_ess_before': tf.reduce_min(ess_before),
                'min_ess_after': tf.reduce_min(ess_after),
            }
            return particles_seq, logw_seq, assignment_matrices, ess_stats

        return particles_seq, logw_seq, assignment_matrices


# ----------------------------------------------------------------------
# Example: 1D linear-Gaussian state-space model
# x_t = a * x_{t-1} + q_noise,      q_noise ~ N(0, sigma_q^2)
# y_t = x_t + r_noise,              r_noise ~ N(0, sigma_r^2)
# ----------------------------------------------------------------------


# def linear_gaussian_transition(x_prev, params):
#     """Linear-Gaussian state transition function.

#     Args:
#         x_prev (Tensor): Previous states of shape (B, N, 1).
#         params (dict): Parameters containing 'a' (transition coefficient) and
#             'sigma_q' (process noise standard deviation).

#     Returns:
#         Tensor: Next states of shape (B, N, 1).
#     """
#     a = params.get("a", 0.9)
#     sigma_q = params.get("sigma_q", 0.5)

#     a = tf.convert_to_tensor(a, dtype=tf.float32)
#     sigma_q = tf.convert_to_tensor(sigma_q, dtype=tf.float32)

#     a = tf.reshape(a, (1, 1, 1))
#     sigma_q = tf.reshape(sigma_q, (1, 1, 1))

#     eps = tf.random.normal(tf.shape(x_prev), dtype=tf.float32)
#     x_new = a * x_prev + sigma_q * eps
#     return x_new


# def linear_gaussian_log_likelihood(x, y, params):
#     """Compute log-likelihood log p(y | x) for linear-Gaussian observation model.

#     Assumes y = x + N(0, sigma_r^2).

#     Args:
#         x (Tensor): States of shape (B, N, 1).
#         y (Tensor): Observations of shape (B, 1).
#         params (dict): Parameters containing 'sigma_r' (observation noise std dev).

#     Returns:
#         Tensor: Log-likelihoods of shape (B, N).
#     """
#     sigma_r = params.get("sigma_r", 0.5)
#     sigma_r = tf.convert_to_tensor(sigma_r, dtype=tf.float32)

#     # Broadcast y to (B, N, 1)
#     y = tf.expand_dims(y, axis=1)  # (B,1,1)
#     y = tf.broadcast_to(y, tf.shape(x))  # (B,N,1)

#     diff = y - x
#     var = sigma_r ** 2
#     log_norm_const = -0.5 * tf.math.log(2.0 * np.pi * var)
#     log_lik = log_norm_const - 0.5 * (diff ** 2) / var  # (B,N,1)
#     return tf.squeeze(log_lik, axis=-1)  # (B,N)


# # ----------------------------------------------------------------------
# # Demo usage
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     tf.random.set_seed(42)
#     np.random.seed(42)

#     print("=" * 70)
#     print("Differentiable Particle Filter with RNN Resampling - Demo")
#     print("=" * 70)

#     # Model parameters
#     a_true = 0.9
#     sigma_q_true = 0.5
#     sigma_r_true = 0.7

#     # Generate a synthetic trajectory
#     T = 20
#     x = np.zeros((T + 1, 1), dtype=np.float32)
#     y = np.zeros((T, 1), dtype=np.float32)

#     x[0] = 0.0
#     for t in range(T):
#         x[t + 1] = a_true * x[t] + sigma_q_true * np.random.randn()
#         y[t] = x[t + 1] + sigma_r_true * np.random.randn()

#     # Wrap in batch dimension B=1
#     y_batch = y[None, :, :]  # (1,T,1)

#     # Build DPF with RNN resampling
#     n_particles = 50
#     state_dim = 1

#     dpf_rnn = DifferentiableParticleFilterRNN(
#         n_particles=n_particles,
#         state_dim=state_dim,
#         transition_fn=linear_gaussian_transition,
#         log_likelihood_fn=linear_gaussian_log_likelihood,
#         rnn_type='lstm',           # or 'gru'
#         rnn_hidden_dim=32,
#         rnn_num_layers=1,
#         use_weight_features=True,
#         use_particle_features=True,
#         temperature=1.0,
#     )

#     # Prior N(0, 1)
#     init_mean = np.array([0.0], dtype=np.float32)
#     init_cov_chol = np.array([[1.0]], dtype=np.float32)

#     params = {
#         "a": a_true,
#         "sigma_q": sigma_q_true,
#         "sigma_r": sigma_r_true,
#     }

#     print("\nRunning RNN-based particle filter with ESS monitoring...")
#     particles_seq, logw_seq, assignment_matrices, ess_stats = dpf_rnn.filter(
#         observations=y_batch,
#         init_mean=init_mean,
#         init_cov_chol=init_cov_chol,
#         params=params,
#         return_ess=True,
#     )

#     # Compute posterior means
#     w = tf.exp(logw_seq)  # (1, T+1, N)
#     x_particles = particles_seq  # (1, T+1, N, 1)
#     posterior_means = tf.reduce_sum(w[..., None] * x_particles, axis=2)  # (1,T+1,1)

#     print("\nResults:")
#     print(f"  Posterior means shape: {posterior_means.shape}")
#     print(f"  Assignment matrices shape: {assignment_matrices.shape}")
    
#     # Compute RMSE
#     rmse = np.sqrt(np.mean((posterior_means.numpy().squeeze() - x.squeeze()) ** 2))
#     print(f"  RMSE: {rmse:.4f}")
    
#     # Display ESS statistics
#     print("\nESS Statistics:")
#     print(f"  Mean ESS before resampling: {ess_stats['mean_ess_before'].numpy():.2f}")
#     print(f"  Mean ESS after resampling:  {ess_stats['mean_ess_after'].numpy():.2f}")
#     print(f"  Min ESS before resampling:  {ess_stats['min_ess_before'].numpy():.2f}")
#     print(f"  Min ESS after resampling:   {ess_stats['min_ess_after'].numpy():.2f}")
#     print(f"  ESS as % of N particles:    {ess_stats['mean_ess_before'].numpy() / n_particles * 100:.1f}%")
    
#     # Display entropy statistics
#     print("\nWeight Entropy Statistics:")
#     print(f"  Mean entropy before resampling: {ess_stats['mean_entropy_before'].numpy():.3f}")
#     print(f"  Mean entropy after resampling:  {ess_stats['mean_entropy_after'].numpy():.3f}")
#     print(f"  Maximum possible entropy:       {np.log(n_particles):.3f} (log(N))")
#     print(f"  Entropy ratio before:           {ess_stats['mean_entropy_before'].numpy() / np.log(n_particles) * 100:.1f}%")