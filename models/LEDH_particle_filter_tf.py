"""
TensorFlow implementation of the Local-Exact Daum–Huang (LEDH) particle-flow filter.

The LEDH filter is a particle-flow method that transports particles along a
pseudo-time path from the prior to the posterior using a per-particle
linearisation of the observation model.  An EKF or UKF provides the global
covariance estimate :math:`P` used in the flow equations.  Optional
entropy-regularised optimal-transport (OT) resampling is available to combat
particle degeneracy.

Classes
-------
GaussianTracker
    Protocol for EKF/UKF wrappers consumed by :class:`LEDHFlowPF_TF`.
LEDHConfig
    Hyper-parameter container for the LEDH particle-flow filter.
PFState
    Immutable snapshot of particle-filter state at a single time step.
LEDHFlowPF_TF
    Main filter class; implements initialisation, one-step update, and
    weighted-statistics helpers.

Functions
---------
systematic_resample_tf
    TensorFlow systematic resampling returning ancestor indices.
effective_sample_size
    Compute the effective sample size from normalised weights.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple
import tensorflow as tf
import numpy as np

# Import OT resampling
from DPF_OT_resampling import sinkhorn_ot_resample


class GaussianTracker(Protocol):
    """Protocol for EKF/UKF wrappers used by :class:`LEDHFlowPF_TF`.

    Any object that implements ``predict``, ``update``, and
    ``get_past_mean`` with the signatures below can be passed as the
    ``tracker`` argument.
    """

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a time-update (prediction) step.

        Returns
        -------
        m_pred : ndarray, shape (nx,)
            Predicted state mean.
        P_pred : ndarray, shape (nx, nx)
            Predicted state covariance.
        """
        ...

    def update(self, z_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a measurement-update step.

        Parameters
        ----------
        z_k : ndarray, shape (nz,)
            Current observation vector.

        Returns
        -------
        m_post : ndarray, shape (nx,)
            Posterior state mean.
        P_post : ndarray, shape (nx, nx)
            Posterior state covariance.
        """
        ...

    def get_past_mean(self) -> np.ndarray:
        """Return the most recent posterior mean.

        Returns
        -------
        m : ndarray, shape (nx,)
            Most recent posterior (or predicted) state mean.
        """
        ...

GFn = Callable[[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]], tf.Tensor]
HFn = Callable[[tf.Tensor], tf.Tensor]
JacobianHFn = Callable[[tf.Tensor], tf.Tensor]
LogTransPdf = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
LogLikePdf  = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


def systematic_resample_tf(weights: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
    """Return ancestor indices via systematic resampling.

    Parameters
    ----------
    weights : Tensor, shape (N,)
        Non-negative (unnormalised) particle weights.
    seed : int, optional
        Random seed for reproducibility.  If ``None`` no seed is set.

    Returns
    -------
    idx : Tensor of int32, shape (N,)
        Ancestor indices in ``[0, N)``.
    """
    n = tf.shape(weights)[0]
    n_float = tf.cast(n, tf.float32)
    w = weights / tf.reduce_sum(weights)
    
    if seed is not None:
        tf.random.set_seed(seed)
    u = tf.random.uniform([], dtype=tf.float32)
    positions = (u + tf.cast(tf.range(n), tf.float32)) / n_float
    
    cdf = tf.cumsum(w)
    idx = tf.searchsorted(cdf, positions, side='right')
    idx = tf.minimum(idx, n - 1)
    
    return idx

def effective_sample_size(weights: tf.Tensor) -> tf.Tensor:
    """Compute the effective sample size (ESS).

    Uses the standard formula :math:`\\text{ESS} = 1 / \\sum_i w_i^2`
    after normalising ``weights`` to sum to one.

    Parameters
    ----------
    weights : Tensor, shape (N,)
        Non-negative (unnormalised) particle weights.

    Returns
    -------
    ess : Tensor, scalar float32
        Effective sample size in the range ``(0, N]``.
    """
    w = weights / tf.reduce_sum(weights)
    return 1.0 / tf.reduce_sum(w * w)


@dataclass
class LEDHConfig:
    """Hyper-parameter container for :class:`LEDHFlowPF_TF`.

    Attributes
    ----------
    n_particles : int
        Number of particles.  Default ``512``.
    n_lambda_steps : int
        Number of pseudo-time sub-steps used to integrate the LEDH flow
        from :math:`\\lambda = 0` to :math:`\\lambda = 1`.  Default ``8``.
    resample_ess_ratio : float
        Resampling is triggered when
        :math:`\\text{ESS} < \\text{resample\\_ess\\_ratio} \\times N`.
        Set to ``0.0`` to disable resampling entirely.  Default ``0.0``.
    use_ot_resampling : bool
        If ``True``, use entropy-regularised optimal-transport resampling
        (Sinkhorn) instead of systematic resampling.  Default ``False``.
    ot_epsilon : float
        Entropic regularisation strength for Sinkhorn OT.  Default ``0.1``.
    ot_sinkhorn_iters : int
        Number of Sinkhorn iterations.  Default ``50``.
    random_seed : int or None
        Global random seed for reproducible runs.  Default ``None``.
    """
    n_particles: int = 512
    n_lambda_steps: int = 8
    resample_ess_ratio: float = 0.0
    use_ot_resampling: bool = False
    ot_epsilon: float = 0.1
    ot_sinkhorn_iters: int = 50
    random_seed: Optional[int] = None

@dataclass
class PFState:
    """Particle-filter state snapshot at a single time step.

    Attributes
    ----------
    particles : Tensor, shape (N, nx)
        Current particle positions.
    weights : Tensor, shape (N,)
        Normalised particle weights (sum to one).
    mean : Tensor, shape (nx,)
        Weighted-mean estimate of the state.
    cov : Tensor, shape (nx, nx)
        Weighted-covariance estimate of the state.
    diagnostics : dict or None
        Optional dictionary of per-step diagnostic values (e.g. ESS,
        condition numbers).  Defaults to ``None``.
    """
    particles: tf.Tensor
    weights: tf.Tensor
    mean: tf.Tensor
    cov: tf.Tensor
    diagnostics: dict = None


class LEDHFlowPF_TF(tf.Module):
    """TensorFlow implementation of EKF/UKF-assisted LEDH particle-flow PF.

    Each time step transports particles from the prior to the posterior
    by integrating a per-particle linearised LEDH flow over
    :math:`\\lambda \\in [0, 1]`.  A :class:`GaussianTracker` (EKF or UKF)
    provides the global predicted covariance :math:`P` used in the flow
    equations.

    Parameters
    ----------
    tracker : GaussianTracker
        EKF or UKF wrapper that provides ``predict``, ``update``, and
        ``get_past_mean``.
    g : callable
        Process model ``g(x, u, v) -> x_next`` as a TensorFlow function.
    h : callable
        Observation model ``h(x) -> z`` as a TensorFlow function.
    jacobian_h : callable
        Returns the Jacobian ``∂h/∂x`` evaluated at a single particle.
    log_trans_pdf : callable
        ``log_trans_pdf(x_new, x_old) -> scalar`` log transition density.
    log_like_pdf : callable
        ``log_like_pdf(z, x) -> scalar`` log likelihood.
    R : ndarray, shape (nz, nz)
        Observation noise covariance matrix.
    config : LEDHConfig, optional
        Algorithm hyper-parameters.  Defaults to :class:`LEDHConfig` with
        its default values.
    name : str, optional
        ``tf.Module`` name.  Default ``"LEDHFlowPF_TF"``.
    """

    def __init__(
        self,
        tracker: GaussianTracker,
        g: GFn,
        h: HFn,
        jacobian_h: JacobianHFn,
        log_trans_pdf: LogTransPdf,
        log_like_pdf: LogLikePdf,
        R: np.ndarray,
        config: Optional[LEDHConfig] = None,
        name: str = "LEDHFlowPF_TF",
    ) -> None:
        """Initialise the LEDH particle-flow filter.

        Parameters
        ----------
        tracker : GaussianTracker
            EKF or UKF wrapper supplying the global covariance estimates.
        g : callable
            Process model ``g(x, u, v) -> x_next`` (TensorFlow function).
        h : callable
            Observation model ``h(x) -> z`` (TensorFlow function).
        jacobian_h : callable
            Returns the Jacobian ``∂h/∂x``, shape ``(nz, nx)``, evaluated
            at a single particle (TensorFlow function).
        log_trans_pdf : callable
            Log transition density ``log p(x_new | x_old)`` returning a
            scalar ``tf.float32`` tensor.
        log_like_pdf : callable
            Log likelihood ``log p(z | x)`` returning a scalar
            ``tf.float32`` tensor.
        R : ndarray, shape (nz, nz)
            Observation noise covariance matrix (converted to
            ``tf.float32`` internally).
        config : LEDHConfig, optional
            Hyper-parameters; defaults to :class:`LEDHConfig`.
        name : str, optional
            ``tf.Module`` name.  Default ``"LEDHFlowPF_TF"``.
        """
        super().__init__(name=name)
        self.tracker = tracker
        self.g = g
        self.h = h
        self.Jh = jacobian_h
        self.log_trans_pdf = log_trans_pdf
        self.log_like_pdf = log_like_pdf
        self.R = tf.constant(R, dtype=tf.float32)
        self.cfg = config or LEDHConfig()

    def init_from_gaussian(self, mean0: np.ndarray, cov0: np.ndarray) -> PFState:
        """Initialise particles by sampling from a Gaussian prior.

        Implements Algorithm lines 1–2: draw ``N`` particles from
        :math:`\\mathcal{N}(\\text{mean}_0, \\text{cov}_0)` and assign
        uniform weights.

        Parameters
        ----------
        mean0 : ndarray, shape (nx,)
            Prior mean.
        cov0 : ndarray, shape (nx, nx)
            Prior covariance.

        Returns
        -------
        state : PFState
            Initial filter state with uniformly weighted particles.
        """
        n, nx = self.cfg.n_particles, mean0.size
        
        eps = tf.random.normal([n, nx], dtype=tf.float32, seed=self.cfg.random_seed)
        cov0_chol = tf.linalg.cholesky(tf.constant(cov0, dtype=tf.float32))
        particles = tf.constant(mean0, dtype=tf.float32)[None, :] + eps @ tf.transpose(cov0_chol)
        
        weights = tf.ones(n, dtype=tf.float32) / tf.cast(n, tf.float32)
        mean, cov = self._weighted_stats(particles, weights)
        return PFState(particles=particles, weights=weights, mean=mean, cov=cov, diagnostics={})

    def step(
        self,
        state: PFState,
        z_k: np.ndarray,
        u_km1: Optional[np.ndarray] = None,
        process_noise_sampler: Optional[Callable[[int, int], tf.Tensor]] = None,
    ) -> PFState:
        """Run one LEDH particle-flow filter step.

        Implements Algorithm 1 (per-particle LEDH):

        1. EKF/UKF prediction to obtain the predicted covariance :math:`P`.
        2. Propagate each particle through the process model to obtain
           :math:`\\eta_0^i`.
        3. Integrate the per-particle LEDH flow over
           :math:`\\lambda \\in [0, 1]` using ``n_lambda_steps`` Euler
           sub-steps, accumulating log-Jacobian weights.
        4. Compute and normalise importance weights, then run the
           EKF/UKF measurement update.
        5. Optionally resample when ESS falls below the threshold.

        Parameters
        ----------
        state : PFState
            Filter state from the previous time step.
        z_k : ndarray, shape (nz,)
            Current observation vector.
        u_km1 : ndarray, optional
            Control input at time ``k-1`` passed to the process model.
            If ``None``, no control is applied.
        process_noise_sampler : callable, optional
            ``sampler(N, nx) -> Tensor`` that generates process-noise
            samples.  If ``None``, zero noise is used.

        Returns
        -------
        new_state : PFState
            Updated filter state containing posterior particles, normalised
            weights, weighted mean, weighted covariance, and diagnostics.
        """
        N = tf.shape(state.particles)[0]
        nx = tf.shape(state.particles)[1]
        I = tf.eye(nx, dtype=tf.float32)

        # Algorithm line 5: EKF/UKF prediction to obtain P^i
        m_pred, P_np = self.tracker.predict()
        P = tf.constant(P_np, dtype=tf.float32)
        P = 0.5 * (P + tf.transpose(P))

        # Algorithm line 7: Propagate particles η_0^i = g_k(x_{k-1}^i, v_k)
        if process_noise_sampler is None:
            v = tf.zeros_like(state.particles)
        else:
            v = process_noise_sampler(N, nx)
        
        u_km1_tf = tf.constant(u_km1, dtype=tf.float32) if u_km1 is not None else None
        
        eta0 = tf.map_fn(
            lambda i: self.g(state.particles[i], u_km1_tf, v[i]),
            tf.range(N),
            fn_output_signature=tf.TensorSpec(shape=[nx], dtype=tf.float32)
        )

        # Algorithm lines 8-9: Initialize flow variables
        eta = tf.identity(eta0)
        etabar = tf.identity(eta0)  # Per line 15: ȳ_0^i = η_0^i
        theta_log = tf.zeros(N, dtype=tf.float32)  # log θ^i for numerical stability

        cond_numbers = []

        # Algorithm lines 11-21: Pseudo-time integration λ ∈ [0,1]
        n_steps = max(1, int(self.cfg.n_lambda_steps))
        dlam = 1.0 / float(n_steps)
        lam = 0.0
        
        z_k_tf = tf.constant(z_k, dtype=tf.float32)

        for step_idx in range(n_steps):
            lam = min(1.0, lam + dlam)
            
            # Per-particle flow update
            eta_list = []
            etabar_list = []
            theta_log_list = []
            
            for i in range(int(N.numpy())):
                # Algorithm line 16: Linearize at η_λ^i (PARTICLE position)
                Hi = self.Jh(eta[i])  # [nz, nx]
                h_eta_i = self.h(eta[i])  # [nz]
                ei = h_eta_i - tf.linalg.matvec(Hi, eta[i])  # [nz]
                
                # Algorithm line 16: Calculate A^i(λ)
                Si = lam * Hi @ P @ tf.transpose(Hi) + self.R
                # Add regularization
                nz = tf.shape(Si)[0]
                Si = Si + 1e-6 * tf.eye(nz, dtype=tf.float32)
                
                # Track condition number (only first particle)
                if i == 0:
                    cond_numbers.append(float('nan'))
                
                Si_inv_Hi = tf.linalg.lstsq(Si, Hi, l2_regularizer=1e-6)
                Ai = -0.5 * P @ tf.transpose(Hi) @ Si_inv_Hi
                
                # Algorithm line 16: Calculate b^i(λ)
                innov_i = z_k_tf - ei  # [nz]
                innov_i_mat = tf.reshape(innov_i, [-1, 1])  # [nz, 1]
                Rin_innov_i_mat = tf.linalg.solve(self.R + 1e-6 * tf.eye(nz, dtype=tf.float32), innov_i_mat)
                Rin_innov_i = tf.reshape(Rin_innov_i_mat, [-1])  # [nz]
                PHt_Rinv_innov_i = tf.linalg.matvec(P @ tf.transpose(Hi), Rin_innov_i)  # [nx]
                eta0_contrib = tf.linalg.matvec(Ai, eta0[i])  # [nx]
                bi = tf.linalg.matvec((I + 2.0 * lam * Ai),
                                       tf.linalg.matvec((I + lam * Ai), PHt_Rinv_innov_i) + eta0_contrib)
                
                # Algorithm line 17: Migrate ȳ_j^i
                etabar_new = etabar[i] + dlam * (tf.linalg.matvec(Ai, etabar[i]) + bi)
                etabar_list.append(etabar_new)
                
                # Algorithm line 18: Migrate particles η_j^i
                eta_new = eta[i] + dlam * (tf.linalg.matvec(Ai, eta[i]) + bi)
                eta_list.append(eta_new)
                
                # Algorithm line 19: Calculate θ^i = θ^i |det(I + ε_j A^i)|
                M = I + dlam * Ai
                sign, logdet = tf.linalg.slogdet(M)
                # Handle negative determinants
                if sign <= 0:
                    M = M + 1e-12 * I
                    sign, logdet = tf.linalg.slogdet(M)
                theta_log_new = theta_log[i] + logdet
                theta_log_list.append(theta_log_new)
            
            # Stack updated particles
            eta = tf.stack(eta_list)
            etabar = tf.stack(etabar_list)
            theta_log = tf.stack(theta_log_list)

        # Algorithm line 23: Set x_k^i = η_1^i
        xk = eta

        # Algorithm line 24: Calculate weights
        logw = tf.math.log(state.weights + 1e-300) + theta_log
        
        # Vectorized weight computation
        log_trans_xk = tf.map_fn(
            lambda i: self.log_trans_pdf(xk[i], state.particles[i]),
            tf.range(N),
            fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.float32)
        )
        log_like = tf.map_fn(
            lambda i: self.log_like_pdf(z_k_tf, xk[i]),
            tf.range(N),
            fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.float32)
        )
        log_trans_eta0 = tf.map_fn(
            lambda i: self.log_trans_pdf(eta0[i], state.particles[i]),
            tf.range(N),
            fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.float32)
        )
        
        logw = logw + log_trans_xk + log_like - log_trans_eta0
        
        # Algorithm line 27: Normalize weights
        logw = logw - tf.reduce_max(logw)
        w = tf.exp(logw)
        w = w / tf.reduce_sum(w)

        # Algorithm line 28: EKF/UKF measurement update
        self.tracker.update(z_k)

        # Algorithm line 31: (Optional) Resample
        if self.cfg.resample_ess_ratio > 0.0:
            ess = effective_sample_size(w)
            if ess < self.cfg.resample_ess_ratio * tf.cast(N, tf.float32):
                if self.cfg.use_ot_resampling:
                    # Use entropy-regularized OT resampling
                    xk, w, ot_diag = sinkhorn_ot_resample(
                        xk, w,
                        epsilon=self.cfg.ot_epsilon,
                        n_iters=self.cfg.ot_sinkhorn_iters,
                        return_diagnostics=True
                    )
                    cond_numbers.append(('OT', ot_diag))
                else:
                    # Use systematic resampling
                    idx = systematic_resample_tf(w, seed=self.cfg.random_seed)
                    xk = tf.gather(xk, idx)
                    w = tf.ones_like(w) / tf.cast(N, tf.float32)

        # Algorithm line 30: Estimate x̂_k
        mean, cov = self._weighted_stats(xk, w)
        
        diagnostics = {
            'condition_numbers': cond_numbers,
            'ess': float(effective_sample_size(w).numpy())
        }
        
        return PFState(particles=xk, weights=w, mean=mean, cov=cov, diagnostics=diagnostics)

    
    @staticmethod
    def _weighted_stats(x: tf.Tensor, w: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute the weighted mean and covariance of a particle cloud.

        Parameters
        ----------
        x : Tensor, shape (N, nx)
            Particle positions.
        w : Tensor, shape (N,)
            Non-negative particle weights (normalised internally).

        Returns
        -------
        mean : Tensor, shape (nx,)
            Weighted mean.
        cov : Tensor, shape (nx, nx)
            Weighted covariance (symmetrised).
        """
        w = w / tf.reduce_sum(w)
        mean = tf.reduce_sum(x * w[:, None], axis=0)
        xc = x - mean[None, :]
        # Weighted covariance: cov = sum_i w_i * (x_i - mean)(x_i - mean)^T
        weighted_xc = xc * tf.sqrt(w)[:, None]  # (N, d)
        cov = tf.linalg.matmul(weighted_xc, weighted_xc, transpose_a=True)  # (d, d)
        cov = 0.5 * (cov + tf.transpose(cov))
        return mean, cov
