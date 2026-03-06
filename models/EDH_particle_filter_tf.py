"""
TensorFlow implementation of EKF/UKF-assisted EDH Particle-Flow PF.

Provides a TF-native EDH particle flow filter compatible with
entropy-regularised OT resampling (``DPF_OT_resampling``).

Classes
-------
GaussianTracker (Protocol) – Interface for auxiliary EKF/UKF components.
EDHConfig                   – Filter hyper-parameters.
PFState                     – Output state container.
EDHFlowPF_TF                – Main EDH flow filter (``tf.Module`` subclass).

Functions
---------
rk4_step              – One fourth-order Runge-Kutta step.
systematic_resample_tf – TF systematic resampling.
effective_sample_size  – ESS from normalised weights.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple
import tensorflow as tf
import numpy as np

# Import OT resampling
from DPF_OT_resampling import sinkhorn_ot_resample

# Protocols 

class GaussianTracker(Protocol):
    """Protocol for auxiliary EKF/UKF trackers used by the EDH flow filter.

    Any object that implements the three methods below (with matching
    signatures) satisfies this protocol.
    """

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform the prediction step and advance the internal clock.

        Returns
        -------
        m_pred : ndarray, shape (nx,)
            Predicted state mean ``m_{k|k-1}``.
        P_pred : ndarray, shape (nx, nx)
            Predicted state covariance ``P_{k|k-1}``.
        """

    def update(self, z_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform the measurement-update step.

        Parameters
        ----------
        z_k : ndarray, shape (nz,)
            Current observation.

        Returns
        -------
        m_post : ndarray, shape (nx,)
            Posterior state mean ``m_{k|k}``.
        P_post : ndarray, shape (nx, nx)
            Posterior state covariance ``P_{k|k}``.
        """

    def get_past_mean(self) -> np.ndarray:
        """Return the filtered mean from the previous time step.

        Returns
        -------
        ndarray, shape (nx,)
            Mean ``m_{k-1|k-1}`` (used to seed the mean flow trajectory
            ``η̄_0 = g_k(m_{k-1|k-1}, 0)``).
        """

GFn = Callable[[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]], tf.Tensor]
HFn = Callable[[tf.Tensor], tf.Tensor]
JacobianHFn = Callable[[tf.Tensor], tf.Tensor]
LogTransPdf = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
LogLikePdf  = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]

# Utils
def rk4_step(x: tf.Tensor, f: Callable[[tf.Tensor], tf.Tensor], dt: float) -> tf.Tensor:
    """Integrate ``x' = f(x)`` by one step using fourth-order Runge-Kutta.

    Parameters
    ----------
    x : tf.Tensor, shape (n,)
        Current state vector.
    f : callable
        Right-hand-side function ``x -> dx/dt``.
    dt : float
        Step size.

    Returns
    -------
    tf.Tensor, shape (n,)
        State after one RK4 step.
    """
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def systematic_resample_tf(weights: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
    """Systematic resampling in TensorFlow.

    Parameters
    ----------
    weights : tf.Tensor, shape (N,)
        Non-negative particle weights (need not be normalised).
    seed : int, optional
        Random seed for the single uniform draw.  Default is ``None``.

    Returns
    -------
    tf.Tensor, shape (N,), dtype int32
        Integer ancestor indices in ``[0, N-1]``.
    """
    n = tf.shape(weights)[0]
    n_float = tf.cast(n, tf.float32)
    w = weights / tf.reduce_sum(weights)
    
    # Generate systematic positions
    if seed is not None:
        tf.random.set_seed(seed)
    u = tf.random.uniform([], dtype=tf.float32)
    positions = (u + tf.cast(tf.range(n), tf.float32)) / n_float
    
    # Compute CDF
    cdf = tf.cumsum(w)
    
    # Find indices using searchsorted
    idx = tf.searchsorted(cdf, positions, side='right')
    idx = tf.minimum(idx, n - 1)
    
    return idx

def effective_sample_size(weights: tf.Tensor) -> tf.Tensor:
    """Compute the Effective Sample Size (ESS).

    Parameters
    ----------
    weights : tf.Tensor, shape (N,)
        Non-negative particle weights (normalised internally).

    Returns
    -------
    tf.Tensor
        Scalar ESS = ``1 / Σ_i w̃_i²`` where ``w̃`` are normalised weights.
    """
    w = weights / tf.reduce_sum(weights)
    return 1.0 / tf.reduce_sum(w * w)

# Config/State

@dataclass
class EDHConfig:
    """Configuration hyper-parameters for the TF EDH Particle-Flow PF.

    Attributes
    ----------
    n_particles : int
        Number of particles.  Default is ``512``.
    n_lambda_steps : int
        Number of pseudo-time steps per observation.  Default is ``8``.
    resample_ess_ratio : float
        Resample when ``ESS < ratio * N``.  Set to ``0`` to disable.
        Default is ``0.5``.
    flow_integrator : str
        Integration scheme for the flow ODE.  ``'rk4'`` (default) or
        ``'euler'``.
    use_ot_resampling : bool
        Use Sinkhorn-OT resampling instead of systematic.  Default is
        ``False``.
    ot_epsilon : float
        Entropic regularisation for OT resampling.  Default is ``0.1``.
    ot_sinkhorn_iters : int
        Number of Sinkhorn iterations.  Default is ``50``.
    random_seed : int, optional
        Random seed.  Default is ``None``.
    """
    n_particles: int = 512
    n_lambda_steps: int = 8
    resample_ess_ratio: float = 0.5
    flow_integrator: str = "rk4"  # "rk4" or "euler"
    use_ot_resampling: bool = False
    ot_epsilon: float = 0.1
    ot_sinkhorn_iters: int = 50
    random_seed: Optional[int] = None

@dataclass
class PFState:
    """Particle filter state container.

    Attributes
    ----------
    particles : tf.Tensor, shape (N, nx)
        Particle ensemble.
    weights : tf.Tensor, shape (N,)
        Normalised particle weights.
    mean : tf.Tensor, shape (nx,)
        Weighted ensemble mean.
    cov : tf.Tensor, shape (nx, nx)
        Weighted ensemble covariance.
    diagnostics : dict, optional
        Diagnostic information from the last step (ESS, condition numbers,
        etc.).
    """
    particles: tf.Tensor  # (N, nx)
    weights: tf.Tensor    # (N,)
    mean: tf.Tensor       # (nx,)
    cov: tf.Tensor        # (nx, nx)
    diagnostics: dict = None

# EDH Flow PF 

class EDHFlowPF_TF(tf.Module):
    """TensorFlow EDH Particle-Flow Filter assisted by an EKF or UKF.

    Moves particles from the prior to an approximate posterior by integrating
    a linearisation-based flow in pseudo-time λ ∈ [0, 1].  A ``GaussianTracker``
    (EKF or UKF) provides the linearisation trajectory.
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
        config: Optional[EDHConfig] = None,
        name: str = "EDHFlowPF_TF",
    ) -> None:
        """Initialise the EDH Particle-Flow Filter.

        Parameters
        ----------
        tracker : GaussianTracker
            EKF or UKF that provides the linearisation trajectory
            ``(m_{k|k-1}, P_{k|k-1})`` and advances to ``(m_{k|k}, P_{k|k})``.
        g : callable
            Vectorised process function ``(x, u, v) -> x_next``
            (``tf.Tensor`` inputs).
        h : callable
            Observation function ``x -> z`` (``tf.Tensor``).
        jacobian_h : callable
            Jacobian of ``h`` at a single state ``x -> J`` with shape
            ``(nz, nx)``.
        log_trans_pdf : callable
            Log transition density ``(x_new, x_old) -> scalar``.
        log_like_pdf : callable
            Log likelihood ``(z, x) -> scalar``.
        R : ndarray, shape (nz, nz)
            Measurement-noise covariance used in the EDH flow matrices.
        config : EDHConfig, optional
            Filter hyper-parameters.  Defaults to ``EDHConfig()``.
        name : str, optional
            TensorFlow module name.  Default is ``'EDHFlowPF_TF'``.
        """
        super().__init__(name=name)
        self.tracker = tracker
        self.g = g
        self.h = h
        self.Jh = jacobian_h
        self.log_trans_pdf = log_trans_pdf
        self.log_like_pdf = log_like_pdf
        self.R = tf.constant(R, dtype=tf.float32)
        self.cfg = config or EDHConfig()


    def init_from_gaussian(self, mean0: np.ndarray, cov0: np.ndarray) -> PFState:
        """Initialise particles by sampling from N(mean0, cov0).

        Parameters
        ----------
        mean0 : ndarray, shape (nx,)
            Prior mean.
        cov0 : ndarray, shape (nx, nx)
            Prior covariance.

        Returns
        -------
        PFState
            Initial state with uniform weights and ``t = 0``.
        """
        n, nx = self.cfg.n_particles, mean0.size
        
        # Sample particles
        eps = tf.random.normal([n, nx], dtype=tf.float32, seed=self.cfg.random_seed)
        cov0_chol = tf.linalg.cholesky(tf.constant(cov0, dtype=tf.float32))
        particles = tf.constant(mean0, dtype=tf.float32)[None, :] + eps @ tf.transpose(cov0_chol)
        
        # Uniform weights
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
        """Run one EDH-PF step.

        Executes the following stages:

        1. **EKF/UKF predict** — ``(m_{k-1|k-1}, P_{k-1|k-1}) → (m_{k|k-1}, P_{k|k-1})``.
        2. **Propagate particles** — ``η₀ⁱ = g(x_{k-1}ⁱ, u, v)``.
        3. **Flow update** — integrate the EDH velocity field in pseudo-time
           λ ∈ [0, 1] with the configured integrator.
        4. **Weight update** — importance weights via transition, likelihood,
           and proposal densities.
        5. **EKF/UKF update** — ``(m_{k|k-1}, P_{k|k-1}) → (m_{k|k}, P_{k|k})``.
        6. **Optional resampling** — systematic or OT when ESS is low.

        Parameters
        ----------
        state : PFState
            Filter state from the previous time step.
        z_k : ndarray, shape (nz,)
            Current observation.
        u_km1 : ndarray, optional
            Control input at time ``k-1``.  Default is ``None``.
        process_noise_sampler : callable, optional
            Function ``(N, nx) -> tf.Tensor`` that returns process noise
            samples.  If ``None``, zero noise is used.

        Returns
        -------
        PFState
            Updated filter state for time step ``k``.
        """
        N = tf.shape(state.particles)[0]
        nx = tf.shape(state.particles)[1]

        # --- EKF/UKF prediction: (m_{k|k-1}, P_{k|k-1}) ---
        m_pred, P_np = self.tracker.predict()
        P = tf.constant(P_np, dtype=tf.float32)
        P = 0.5 * (P + tf.transpose(P))  # Enforce symmetry

        # --- Propagate particles to η_0^i = g(x_{k-1}^i, v) ---
        if process_noise_sampler is None:
            v = tf.zeros_like(state.particles)
        else:
            v = process_noise_sampler(N, nx)
        
        u_km1_tf = tf.constant(u_km1, dtype=tf.float32) if u_km1 is not None else None
        
        # Vectorized propagation
        eta0 = tf.map_fn(
            lambda i: self.g(state.particles[i], u_km1_tf, v[i]),
            tf.range(N),
            fn_output_signature=tf.TensorSpec(shape=[nx], dtype=tf.float32)
        )

        # --- Initialize flow states ---
        eta = tf.identity(eta0)
        past_mean = tf.constant(self.tracker.get_past_mean(), dtype=tf.float32)
        etabar = self.g(past_mean, u_km1_tf, tf.zeros(nx, dtype=tf.float32))
        
        # --- Flow update in pseudo-time λ ∈ [0,1] ---
        n_steps = max(1, int(self.cfg.n_lambda_steps))
        dlam = 1.0 / float(n_steps)
        lam = 0.0

        I = tf.eye(nx, dtype=tf.float32)
        z_k_tf = tf.constant(z_k, dtype=tf.float32)
        
        cond_numbers = []

        for step_idx in range(n_steps):
            lam = min(1.0, lam + dlam)
            
            # Linearize at mean trajectory
            H = self.Jh(etabar)  # [nz, nx]
            h_bar = self.h(etabar)  # [nz]
            # Matrix-vector multiply: H @ etabar requires reshaping
            e = h_bar - tf.linalg.matvec(H, etabar)  # [nz]
            
            # Flow matrices
            S = lam * H @ P @ tf.transpose(H) + self.R
            # Add regularization for numerical stability
            nz = tf.shape(S)[0]
            S = S + 1e-6 * tf.eye(nz, dtype=tf.float32)
            
            # Track condition number
            cond_numbers.append(float('nan'))  # Skip condition number for now
            
            # Use lstsq instead of solve for better numerical stability
            S_inv_H = tf.linalg.lstsq(S, H, l2_regularizer=1e-6)
            A = -0.5 * P @ tf.transpose(H) @ S_inv_H
            
            # Compute b
            innov = z_k_tf - e  # [nz]
            innov_mat = tf.reshape(innov, [-1, 1])  # [nz, 1]
            R_inv_innov_mat = tf.linalg.solve(self.R, innov_mat)  # [nz, 1]
            R_inv_innov = tf.reshape(R_inv_innov_mat, [-1])  # [nz]
            PHt_Rinv_innov = tf.linalg.matvec(P @ tf.transpose(H), R_inv_innov)  # [nx]
            etabar_contrib = tf.linalg.matvec(A, etabar)  # [nx]
            b = tf.linalg.matvec((I + 2.0 * lam * A), 
                                  tf.linalg.matvec((I + lam * A), PHt_Rinv_innov) + etabar_contrib)
            
            # Update particles and mean trajectory
            def field(vec: tf.Tensor) -> tf.Tensor:
                return tf.linalg.matvec(A, vec) + b
            
            if self.cfg.flow_integrator.lower() == "euler":
                # Euler integration (vectorized)
                # For particles eta: shape [N, nx]
                eta_updates = tf.map_fn(
                    lambda p: tf.linalg.matvec(A, p),
                    eta,
                    fn_output_signature=tf.TensorSpec(shape=[nx], dtype=tf.float32)
                )
                eta = eta + dlam * (eta_updates + b[None, :])
                etabar = etabar + dlam * field(etabar)
            else:
                # RK4 integration
                eta = tf.map_fn(
                    lambda i: rk4_step(eta[i], field, dlam),
                    tf.range(N),
                    fn_output_signature=tf.TensorSpec(shape=[nx], dtype=tf.float32)
                )
                etabar = rk4_step(etabar, field, dlam)

        # --- Weight update ---
        xk = eta
        
        logw = tf.math.log(state.weights + 1e-300)
        
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
        
        # Normalize
        logw = logw - tf.reduce_max(logw)
        w = tf.exp(logw)
        w = w / tf.reduce_sum(w)

        # --- EKF/UKF measurement update ---
        self.tracker.update(z_k)

        # --- Optional resampling ---
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

        # --- Estimate mean and covariance ---
        mean, cov = self._weighted_stats(xk, w)
        
        diagnostics = {
            'condition_numbers': cond_numbers,
            'ess': float(effective_sample_size(w).numpy())
        }
        
        return PFState(particles=xk, weights=w, mean=mean, cov=cov, diagnostics=diagnostics)

    
    @staticmethod
    def _weighted_stats(x: tf.Tensor, w: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute weighted mean and covariance.

        Parameters
        ----------
        x : tf.Tensor, shape (N, nx)
            Particle ensemble.
        w : tf.Tensor, shape (N,)
            Non-negative weights (normalised internally).

        Returns
        -------
        mean : tf.Tensor, shape (nx,)
            Weighted mean.
        cov : tf.Tensor, shape (nx, nx)
            Weighted sample covariance (symmetrised).
        """
        w = w / tf.reduce_sum(w)
        mean = tf.reduce_sum(x * w[:, None], axis=0)
        xc = x - mean[None, :]
        # Weighted covariance: cov = sum_i w_i * (x_i - mean)(x_i - mean)^T
        cov = tf.linalg.matmul(xc, xc, transpose_a=True)  # (d, N) @ (N, d) = (d, d)
        cov = cov * tf.reduce_sum(w)  # Already normalized, but keep for clarity
        # Alternative correct formula:
        weighted_xc = xc * tf.sqrt(w)[:, None]  # (N, d)
        cov = tf.linalg.matmul(weighted_xc, weighted_xc, transpose_a=True)  # (d, d)
        cov = 0.5 * (cov + tf.transpose(cov))
        return mean, cov
