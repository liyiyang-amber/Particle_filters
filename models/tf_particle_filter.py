"""
TF/TFP Particle Filter (SIR).

Vectorised, @tf.function-compiled Sequential Importance Resampling (SIR)
particle filter for additive-noise nonlinear SSMs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf

try:
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    _HAS_TFP = True
except ImportError:
    _HAS_TFP = False

try:
    from models.DPF_OT_resampling import sinkhorn_ot_resample
except ModuleNotFoundError:
    from DPF_OT_resampling import sinkhorn_ot_resample

F32 = tf.float32
Tensor = tf.Tensor

# Helpers
def _to_f32(x) -> Tensor:
    """Cast *x* to a ``tf.float32`` tensor."""
    return tf.cast(tf.convert_to_tensor(x), F32)

# Systematic resampling (fully vectorised, @tf.function-compatible)
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=F32),
    tf.TensorSpec(shape=[], dtype=tf.int32),
])
def systematic_resample_tf(weights: Tensor, seed: Tensor) -> Tensor:
    """Systematic resampling; returns integer ancestor indices.

    Parameters
    ----------
    weights : Tensor, shape (N,)
        Non-negative particle weights (need not be normalised).
    seed : Tensor, scalar int32
        Seed for the single uniform draw.

    Returns
    -------
    Tensor, shape (N,)
        Integer ancestor indices in ``[0, N-1]``.
    """
    N      = tf.shape(weights)[0]
    N_f    = tf.cast(N, F32)
    w      = weights / tf.reduce_sum(weights)
    # Draw single uniform; generate N stratified positions
    u      = tf.random.stateless_uniform([], seed=[seed, 0], dtype=F32)
    pos    = (u + tf.cast(tf.range(N), F32)) / N_f
    cdf    = tf.cumsum(w)
    idx    = tf.searchsorted(cdf, pos, side="right")
    return tf.minimum(idx, N - 1)


# Weight update kernel
@tf.function
def _log_gaussian_likelihood(z: Tensor, z_pred: Tensor, L_R: Tensor) -> Tensor:
    """Compute batch log-likelihoods log N(z | z_pred_i, R) for all particles.

    Parameters
    ----------
    z : Tensor, shape (nz,)
        Observation vector.
    z_pred : Tensor, shape (N, nz)
        Predicted observations for each of the N particles.
    L_R : Tensor, shape (nz, nz)
        Lower Cholesky factor of the measurement-noise covariance ``R``.

    Returns
    -------
    Tensor, shape (N,)
        Per-particle log-likelihood values.
    """
    nz_f  = tf.cast(tf.shape(z)[0], F32)
    diff  = z[None, :] - z_pred                          # (N, nz)
    # Solve L_R * alpha = diff^T   -> (nz, N)
    alpha = tf.linalg.triangular_solve(L_R, tf.linalg.matrix_transpose(diff), lower=True)
    quad  = tf.reduce_sum(alpha * alpha, axis=0)         # (N,)
    # log|R| = 2 sum log diag(L_R)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_R)))
    log2pi = tf.constant(math.log(2.0 * math.pi), dtype=F32)
    return -0.5 * (quad + logdet + nz_f * log2pi)


# Particle filter state
@dataclass
@dataclass
class PFStateTF:
    """State container for ``ParticleFilterTF``.

    Attributes
    ----------
    particles : Tensor, shape (N, nx)
        Current particle ensemble.
    weights : Tensor, shape (N,)
        Normalised particle weights.
    mean : Tensor, shape (nx,)
        Weighted mean of the ensemble.
    cov : Tensor, shape (nx, nx)
        Weighted sample covariance of the ensemble.
    t : int
        Current time step index.
    """

    particles: tf.Tensor
    weights:   tf.Tensor
    mean:      tf.Tensor
    cov:       tf.Tensor
    t:         int


# SIR Particle Filter
class ParticleFilterTF:
    """Vectorised SIR Particle Filter in TensorFlow.

    Implements Sequential Importance Resampling (SIR) with optional
    systematic or Sinkhorn-OT resampling, compiled with ``@tf.function``
    for efficiency.

    Parameters
    ----------
    g : callable
        Process function.  If ``vectorised_g=True``, signature is
        ``(particles, u) -> particles`` where ``particles`` has shape
        ``(N, nx)``; otherwise signature is ``(x, u) -> x_next`` and
        ``tf.map_fn`` is applied.
    h_vectorised : callable
        Observation function with vectorised signature
        ``(N, nx) -> (N, nz)``.
    Q : ndarray, shape (nx, nx)
        Process-noise covariance.
    R : ndarray, shape (nz, nz)
        Measurement-noise covariance.
    n_particles : int, optional
        Number of particles.  Default is ``1000``.
    resample_thresh : float, optional
        Resample when ``ESS < thresh * N``.  Set to ``0`` to disable.
        Default is ``0.5``.
    use_ot : bool, optional
        Use Sinkhorn-OT resampling instead of systematic.  Default is
        ``False``.
    ot_epsilon : float, optional
        Sinkhorn regularisation parameter.  Default is ``0.1``.
    ot_iters : int, optional
        Number of Sinkhorn iterations.  Default is ``50``.
    random_seed : int, optional
        Base seed for reproducibility.  Default is ``0``.
    vectorised_g : bool, optional
        If ``True``, ``g`` is assumed to be fully vectorised (faster).
        Default is ``False``.
    """

    def __init__(
        self,
        g:              Callable,
        h_vectorised:   Callable,
        Q:              np.ndarray,
        R:              np.ndarray,
        *,
        n_particles:    int   = 1000,
        resample_thresh: float = 0.5,
        use_ot:         bool  = False,
        ot_epsilon:     float = 0.1,
        ot_iters:       int   = 50,
        random_seed:    int   = 0,
        vectorised_g:   bool  = False,
    ) -> None:
        self.g              = g
        self.h_vec          = h_vectorised
        self.Q              = _to_f32(Q)
        self.R              = _to_f32(R)
        self.N              = n_particles
        self.thresh         = resample_thresh
        self.use_ot         = use_ot
        self.ot_eps         = ot_epsilon
        self.ot_iters       = ot_iters
        self.seed           = random_seed
        self.vec_g          = vectorised_g

        nx = Q.shape[0]
        nz = R.shape[0]
        self.nx = nx
        self.nz = nz

        # Precompute Cholesky factors
        self.L_Q = tf.linalg.cholesky(_to_f32(Q) + 1e-12 * tf.eye(nx, dtype=F32))
        self.L_R = tf.linalg.cholesky(_to_f32(R) + 1e-12 * tf.eye(nz, dtype=F32))

        self._state: Optional[PFStateTF] = None
        self._step_counter = 0

    # ---- initialisation ----

    def initialize(self, mean0: np.ndarray, cov0: np.ndarray) -> PFStateTF:
        """Initialise the particle ensemble from a Gaussian prior.

        Parameters
        ----------
        mean0 : ndarray, shape (nx,)
            Prior mean.
        cov0 : ndarray, shape (nx, nx)
            Prior covariance.

        Returns
        -------
        PFStateTF
            Initial filter state with ``t = 0``.
        """
        m0 = _to_f32(mean0)
        P0 = _to_f32(cov0)
        L0 = tf.linalg.cholesky(P0 + 1e-12 * tf.eye(self.nx, dtype=F32))
        eps = tf.random.normal([self.N, self.nx], dtype=F32,
                               seed=self.seed)
        particles = m0[None, :] + eps @ tf.linalg.matrix_transpose(L0)
        weights   = tf.ones(self.N, dtype=F32) / tf.cast(self.N, F32)
        mean, cov = self._weighted_stats(particles, weights)
        self._state = PFStateTF(particles=particles, weights=weights,
                                mean=mean, cov=cov, t=0)
        self._step_counter = 0
        return self._state

    # ---- core step ----

    def step(self, z: np.ndarray, u: Optional[np.ndarray] = None) -> PFStateTF:
        """Run one SIR step: propagate → reweight → (optional) resample.

        Parameters
        ----------
        z : ndarray, shape (nz,)
            Current observation.
        u : ndarray, optional
            Control input passed to the process function.  Default is ``None``.

        Returns
        -------
        PFStateTF
            Updated filter state.
        """
        assert self._state is not None, "Call initialize() first."
        particles = self._state.particles
        weights   = self._state.weights
        self._step_counter += 1
        seed = tf.constant(self._step_counter, dtype=tf.int32)

        # --- 1. Propagate: x_k^i = g(x_{k-1}^i, u) + w^i ---
        noise = tf.random.normal([self.N, self.nx], dtype=F32) @ tf.linalg.matrix_transpose(self.L_Q)
        if self.vec_g:
            u_tf = _to_f32(u) if u is not None else None
            prop = self.g(particles, u_tf)
        else:
            nx = self.nx
            u_tf = _to_f32(u) if u is not None else None
            g_fn = self.g
            prop = tf.map_fn(
                lambda x: g_fn(x, u_tf),
                particles,
                fn_output_signature=tf.TensorSpec([nx], F32),
            )
        particles = prop + noise                               # (N, nx)

        # --- 2. Reweight: log w^i += log p(z | x^i) ---
        z_pred = self.h_vec(particles)                        # (N, nz)
        z_tf   = _to_f32(z)
        logw   = tf.math.log(weights + 1e-300) + _log_gaussian_likelihood(z_tf, z_pred, self.L_R)
        logw   = logw - tf.reduce_max(logw)
        w      = tf.exp(logw)
        w      = w / tf.reduce_sum(w)

        # --- 3. Resample ---
        ess = 1.0 / tf.reduce_sum(w * w)
        if self.thresh > 0.0 and float(ess) < self.thresh * self.N:
            if self.use_ot:
                particles, w, _ = sinkhorn_ot_resample(
                    particles, w,
                    epsilon=self.ot_eps,
                    n_iters=self.ot_iters,
                    return_diagnostics=True,
                )
            else:
                idx       = systematic_resample_tf(w, seed)
                particles = tf.gather(particles, idx)
                w         = tf.ones(self.N, dtype=F32) / tf.cast(self.N, F32)

        mean, cov = self._weighted_stats(particles, w)
        self._state = PFStateTF(particles=particles, weights=w,
                                mean=mean, cov=cov,
                                t=self._step_counter)
        return self._state

    # ---- helpers ----

    @staticmethod
    def _weighted_stats(x: Tensor, w: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute weighted mean and covariance of a particle ensemble.

        Parameters
        ----------
        x : Tensor, shape (N, nx)
            Particle positions.
        w : Tensor, shape (N,)
            Non-negative weights (normalised internally).

        Returns
        -------
        mean : Tensor, shape (nx,)
            Weighted mean.
        cov : Tensor, shape (nx, nx)
            Weighted sample covariance (symmetrised).
        """
        w    = w / tf.reduce_sum(w)
        mean = tf.reduce_sum(x * w[:, None], axis=0)
        xc   = x - mean[None, :]
        wxc  = xc * tf.sqrt(w)[:, None]
        cov  = tf.linalg.matmul(wxc, wxc, transpose_a=True)
        cov  = 0.5 * (cov + tf.linalg.matrix_transpose(cov))
        return mean, cov


# Convenience: run full sequence
def run_particle_filter_tf(
    pf:   ParticleFilterTF,
    Y:    np.ndarray,
    mean0: np.ndarray,
    cov0:  np.ndarray,
    U:    Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the particle filter over a full observation sequence.

    Parameters
    ----------
    pf : ParticleFilterTF
        Initialised particle filter object.
    Y : ndarray, shape (T, nz)
        Observation sequence.
    mean0 : ndarray, shape (nx,)
        Prior mean.
    cov0 : ndarray, shape (nx, nx)
        Prior covariance.
    U : ndarray, shape (T, nu), optional
        Control inputs.  Pass ``None`` if the model has no controls.

    Returns
    -------
    means : ndarray, shape (T, nx)
        Filtered state means.
    covs : ndarray, shape (T, nx, nx)
        Filtered state covariances.
    ess : ndarray, shape (T,)
        Effective sample size at each time step.
    """
    pf.initialize(mean0, cov0)
    T     = Y.shape[0]
    means = np.zeros((T, pf.nx), dtype=np.float32)
    covs  = np.zeros((T, pf.nx, pf.nx), dtype=np.float32)
    ess   = np.zeros(T, dtype=np.float32)

    for t in range(T):
        u_t   = U[t] if U is not None else None
        state = pf.step(Y[t], u=u_t)
        means[t] = state.mean.numpy()
        covs[t]  = state.cov.numpy()
        w        = state.weights.numpy()
        ess[t]   = 1.0 / np.sum(w ** 2)

    return means, covs, ess
