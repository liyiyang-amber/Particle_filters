"""
TensorFlow/TFP core utilities: JIT-compiled Kalman filter, EKF, and UKF.

Provides TF-native implementations of the core Gaussian filters that can
be used as differentiable ``GaussianTracker`` components inside the EDH and
LEDH particle flow filters.

Classes
-------
EKFStateTF  – State dataclass for the step-by-step EKF tracker.
EKFTrackerTF – Extended Kalman Filter implemented with ``tf.GradientTape``
               Jacobians and compatible with the ``GaussianTracker`` protocol.
UKFStateTF  – State dataclass for the step-by-step UKF tracker.
UKFTrackerTF – Unscented Kalman Filter with sigma-point propagation,
               also compatible with ``GaussianTracker``.

Functions
---------
lgssm_log_likelihood – JIT-compiled LGSSM log-likelihood via Kalman recursion.
kalman_filter_tf     – Full forward Kalman filter returning all arrays.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf

# Try to import TFP; gracefully degrade if not installed.
try:
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    _HAS_TFP = True
except ImportError:  # pragma: no cover
    _HAS_TFP = False

# Type aliases
Tensor = tf.Tensor
F32 = tf.float32


# Helpers
def _to_f32(x) -> Tensor:
    """Cast *x* to a ``tf.float32`` tensor."""
    return tf.cast(tf.convert_to_tensor(x), F32)


def _symmetrise(M: Tensor) -> Tensor:
    """Return ``(M + Mᵀ) / 2`` to enforce exact symmetry."""
    return 0.5 * (M + tf.linalg.matrix_transpose(M))


def _chol_logdet(L: Tensor) -> Tensor:
    """Return ``log|M|`` given the Cholesky factor ``L`` where ``M = L Lᵀ``."""
    return 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))


def _chol_solve_vec(L: Tensor, v: Tensor) -> Tensor:
    """Solve ``(L Lᵀ) x = v`` for ``x`` using two triangular solves.

    Parameters
    ----------
    L : Tensor, shape (..., n, n)
        Lower-triangular Cholesky factor.
    v : Tensor, shape (..., n)
        Right-hand-side vector.

    Returns
    -------
    Tensor, shape (..., n)
        Solution vector ``x = (L Lᵀ)⁻¹ v``.
    """
    # v shape: (..., n)  ->  reshape to (..., n, 1), solve, squeeze
    v_ = tf.expand_dims(v, -1)
    x_ = tf.linalg.triangular_solve(L, v_, lower=True)
    x_ = tf.linalg.triangular_solve(tf.linalg.matrix_transpose(L), x_, lower=False)
    return tf.squeeze(x_, axis=-1)


# Result dataclass
# Result namedtuple – namedtuples are valid @tf.function return values
# (TF treats them as flat tuples of tensors, preserving field names).

KFResultsTF = collections.namedtuple(
    "KFResultsTF",
    ["x_pred", "P_pred", "x_filt", "P_filt", "K", "innov", "S", "loglik"],
)
"""Outputs of the TF Kalman filter (all tf.Tensor).

Fields
------
x_pred : (T, nx)
P_pred : (T, nx, nx)
x_filt : (T, nx)
P_filt : (T, nx, nx)
K      : (T, nx, ny)
innov  : (T, ny)
S      : (T, ny, ny)
loglik : scalar
"""


# LGSSM log-likelihood (JIT-compiled, gradient-safe)
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=F32),   # Y   (T, ny)
        tf.TensorSpec(shape=[None, None], dtype=F32),   # F   (nx, nx)
        tf.TensorSpec(shape=[None, None], dtype=F32),   # H   (ny, nx)
        tf.TensorSpec(shape=[None, None], dtype=F32),   # Q   (nx, nx)
        tf.TensorSpec(shape=[None, None], dtype=F32),   # R   (ny, ny)
        tf.TensorSpec(shape=[None],       dtype=F32),   # m0  (nx,)
        tf.TensorSpec(shape=[None, None], dtype=F32),   # P0  (nx, nx)
    ]
)
def lgssm_log_likelihood(
    Y:  Tensor,
    F:  Tensor,
    H:  Tensor,
    Q:  Tensor,
    R:  Tensor,
    m0: Tensor,
    P0: Tensor,
) -> Tensor:
    """Kalman-filter log-likelihood log p(y_{1:T} | F, H, Q, R, m0, P0).

    Computes the marginal log-likelihood by running the Kalman recursion and
    summing the per-step Gaussian log-likelihoods.  The function is compiled
    with ``@tf.function`` and accepts only ``tf.float32`` inputs.

    Parameters
    ----------
    Y  : Tensor, shape (T, ny)
        Observation sequence.
    F  : Tensor, shape (nx, nx)
        State-transition matrix.
    H  : Tensor, shape (ny, nx)
        Observation matrix.
    Q  : Tensor, shape (nx, nx)
        Process-noise covariance.
    R  : Tensor, shape (ny, ny)
        Measurement-noise covariance.
    m0 : Tensor, shape (nx,)
        Initial state mean.
    P0 : Tensor, shape (nx, nx)
        Initial state covariance.

    Returns
    -------
    Tensor
        Scalar log p(y_{1:T}).
    """
    T    = tf.shape(Y)[0]
    ny   = tf.shape(Y)[1]
    nx   = tf.shape(m0)[0]
    ny_f = tf.cast(ny, F32)
    log2pi = tf.constant(math.log(2.0 * math.pi), dtype=F32)

    # --- inner loop body ---
    def step(carry, y_k):
        m, P, _ = carry       # m: (nx,),  P: (nx, nx),  _: previous ll (unused)

        # Prediction
        m_pred = tf.linalg.matvec(F, m)                          # (nx,)
        P_pred = _symmetrise(F @ P @ tf.linalg.matrix_transpose(F) + Q)  # (nx, nx)

        # Innovation
        innov = y_k - tf.linalg.matvec(H, m_pred)               # (ny,)
        S     = _symmetrise(H @ P_pred @ tf.linalg.matrix_transpose(H) + R)  # (ny, ny)

        # Cholesky of S
        L = tf.linalg.cholesky(S + 1e-9 * tf.eye(ny, dtype=F32))  # (ny, ny)

        # Kalman gain  K = P_pred H^T S^{-1}
        # Solve S K^T = H P_pred^T  i.e. L L^T K^T = H P_pred
        PHt  = P_pred @ tf.linalg.matrix_transpose(H)            # (nx, ny)
        # Solve triangularly: K^T = S^{-1} (H P_pred)
        KT   = tf.linalg.cholesky_solve(L, tf.linalg.matrix_transpose(PHt))  # (ny, nx)
        K    = tf.linalg.matrix_transpose(KT)                    # (nx, ny)

        # Update
        m_filt = m_pred + tf.linalg.matvec(K, innov)             # (nx,)
        I      = tf.eye(nx, dtype=F32)
        ImKH   = I - K @ H
        P_filt = _symmetrise(ImKH @ P_pred @ tf.linalg.matrix_transpose(ImKH)
                              + K @ R @ tf.linalg.matrix_transpose(K))  # Joseph form

        # Per-step log-likelihood:  -½ (ν^T S^{-1} ν + log|S| + ny log 2π)
        alpha   = _chol_solve_vec(L, innov)                       # S^{-1} ν
        quad    = tf.reduce_sum(innov * alpha)
        logdet  = _chol_logdet(L)
        step_ll = -0.5 * (quad + logdet + ny_f * log2pi)

        return (m_filt, P_filt, step_ll)

    # tf.scan: fn returns the new carry (m, P, ll_k) at each step.
    # The stacked outputs have shape [(T,nx), (T,nx,nx), (T,)].
    # We extract the per-step log-likelihoods from index [2] and sum them.
    init = (m0, P0, tf.constant(0.0, dtype=F32))

    _, _, log_steps = tf.scan(
        fn          = step,
        elems       = Y,
        initializer = init,
    )

    return tf.reduce_sum(log_steps)


# Full forward Kalman filter (returns all arrays, still JIT-compiled)
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=F32),
        tf.TensorSpec(shape=[None, None], dtype=F32),
        tf.TensorSpec(shape=[None, None], dtype=F32),
        tf.TensorSpec(shape=[None, None], dtype=F32),
        tf.TensorSpec(shape=[None, None], dtype=F32),
        tf.TensorSpec(shape=[None],       dtype=F32),
        tf.TensorSpec(shape=[None, None], dtype=F32),
    ]
)
def kalman_filter_tf(
    Y: Tensor, F: Tensor, H: Tensor, Q: Tensor, R: Tensor,
    m0: Tensor, P0: Tensor,
) -> KFResultsTF:
    """Full Kalman forward pass returning all intermediate arrays.

    Runs the same recursion as ``lgssm_log_likelihood`` but stores all
    predicted and filtered means/covariances, gains, innovations, and
    innovation covariances.

    Parameters
    ----------
    Y  : Tensor, shape (T, ny)
        Observation sequence.
    F  : Tensor, shape (nx, nx)
        State-transition matrix.
    H  : Tensor, shape (ny, nx)
        Observation matrix.
    Q  : Tensor, shape (nx, nx)
        Process-noise covariance.
    R  : Tensor, shape (ny, ny)
        Measurement-noise covariance.
    m0 : Tensor, shape (nx,)
        Initial state mean.
    P0 : Tensor, shape (nx, nx)
        Initial state covariance.

    Returns
    -------
    KFResultsTF
        Named tuple with fields ``x_pred``, ``P_pred``, ``x_filt``,
        ``P_filt``, ``K``, ``innov``, ``S``, ``loglik`` — all
        ``tf.Tensor`` values.
    """
    T    = tf.shape(Y)[0]
    ny   = tf.shape(Y)[1]
    ny_f = tf.cast(ny, F32)
    log2pi = tf.constant(math.log(2.0 * math.pi), dtype=F32)

    # Build zero placeholders whose shapes match the carry elements.
    nx_i   = tf.shape(m0)[0]
    ny_i   = tf.shape(Y)[1]
    z_nx   = tf.zeros([nx_i],       dtype=F32)
    z_nxnx = tf.zeros([nx_i, nx_i], dtype=F32)
    z_nxny = tf.zeros([nx_i, ny_i], dtype=F32)
    z_ny   = tf.zeros([ny_i],       dtype=F32)
    z_nyny = tf.zeros([ny_i, ny_i], dtype=F32)

    # Carry: (m_filt, P_filt, m_pred, P_pred, m_filt_out, P_filt_out, K, innov, S, ll)
    # m_filt / P_filt are the "rolling" filtered state forwarded to the next step.
    # m_filt_out / P_filt_out are identical copies kept so tf.scan stacks x_filt.
    def step(carry, y_k):
        m, P, _, _, _, _, _, _, _, _ = carry

        m_pred = tf.linalg.matvec(F, m)
        P_pred = _symmetrise(F @ P @ tf.linalg.matrix_transpose(F) + Q)

        innov = y_k - tf.linalg.matvec(H, m_pred)
        S     = _symmetrise(H @ P_pred @ tf.linalg.matrix_transpose(H) + R)
        L     = tf.linalg.cholesky(S + 1e-9 * tf.eye(ny_i, dtype=F32))

        PHt = P_pred @ tf.linalg.matrix_transpose(H)
        KT  = tf.linalg.cholesky_solve(L, tf.linalg.matrix_transpose(PHt))
        K   = tf.linalg.matrix_transpose(KT)

        m_filt = m_pred + tf.linalg.matvec(K, innov)
        ImKH   = tf.eye(nx_i, dtype=F32) - K @ H
        P_filt = _symmetrise(ImKH @ P_pred @ tf.linalg.matrix_transpose(ImKH)
                              + K @ R @ tf.linalg.matrix_transpose(K))

        alpha   = _chol_solve_vec(L, innov)
        step_ll = -0.5 * (tf.reduce_sum(innov * alpha) + _chol_logdet(L) + ny_f * log2pi)

        return (m_filt, P_filt, m_pred, P_pred, m_filt, P_filt, K, innov, S, step_ll)

    init = (m0, P0, z_nx, z_nxnx, z_nx, z_nxnx, z_nxny, z_ny, z_nyny,
            tf.constant(0.0, dtype=F32))

    (_, _, x_pred, P_pred_s, x_filt, P_filt_s, Ks, innov_s, S_all, ll_steps) = tf.scan(
        fn          = step,
        elems       = Y,
        initializer = init,
    )

    return KFResultsTF(
        x_pred = x_pred,
        P_pred = P_pred_s,
        x_filt = x_filt,
        P_filt = P_filt_s,
        K      = Ks,
        innov  = innov_s,
        S      = S_all,
        loglik = tf.reduce_sum(ll_steps),
    )


# TF EKF – step-by-step tracker
@dataclass
class EKFStateTF:
    mean: Tensor    # (nx,)
    cov:  Tensor    # (nx, nx)
    t:    int


class EKFTrackerTF:
    """Step-by-step Extended Kalman Filter implemented in TensorFlow.

    Compatible with the ``GaussianTracker`` protocol used by
    ``EDHFlowPF_TF`` and ``LEDHFlowPF_TF``.

    Parameters
    ----------
    g : callable
        State-transition function ``(x, u) -> x_next`` where both arguments
        are ``tf.Tensor`` objects.
    h : callable
        Observation function ``x -> z`` (``tf.Tensor -> tf.Tensor``).
    Q : ndarray or Tensor, shape (nx, nx)
        Process-noise covariance.
    R : ndarray or Tensor, shape (nz, nz)
        Measurement-noise covariance.
    jac_g : callable, optional
        Analytic Jacobian of ``g`` w.r.t. ``x``.  If ``None``, computed
        automatically via ``tf.GradientTape``.
    jac_h : callable, optional
        Analytic Jacobian of ``h`` w.r.t. ``x``.  If ``None``, computed
        automatically via ``tf.GradientTape``.
    joseph : bool, optional
        Use the Joseph-form covariance update for numerical stability.
        Default is ``True``.
    jitter : float, optional
        Small diagonal added to the innovation covariance ``S`` before
        inversion.  Default is ``1e-6``.
    """

    def __init__(
        self,
        g:     Callable,
        h:     Callable,
        Q:     np.ndarray,
        R:     np.ndarray,
        *,
        jac_g: Optional[Callable] = None,
        jac_h: Optional[Callable] = None,
        joseph: bool = True,
        jitter: float = 1e-6,
    ) -> None:
        self.g      = g
        self.h      = h
        self.Q      = _to_f32(Q)
        self.R      = _to_f32(R)
        self.jac_g  = jac_g
        self.jac_h  = jac_h
        self.joseph = joseph
        self.jitter = jitter
        self._state: Optional[EKFStateTF] = None
        self._past_mean: Optional[np.ndarray] = None

    def init(self, mean0: np.ndarray, cov0: np.ndarray) -> EKFStateTF:
        self._state     = EKFStateTF(mean=_to_f32(mean0), cov=_to_f32(cov0), t=0)
        self._past_mean = mean0.copy()
        return self._state

    # ---- GaussianTracker protocol ----

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict step.

        Returns
        -------
        m_pred : ndarray, shape (nx,)
            Predicted state mean.
        P_pred : ndarray, shape (nx, nx)
            Predicted state covariance.
        """
        assert self._state is not None, "Call init() first."
        self._past_mean = self._state.mean.numpy()
        m, P = self._state.mean, self._state.cov

        # Jacobian of g
        if self.jac_g is not None:
            G = _to_f32(self.jac_g(m.numpy(), None))
        else:
            G = _jacobian_tape(self.g, m)

        m_pred = self.g(m, None)
        P_pred = _symmetrise(G @ P @ tf.linalg.matrix_transpose(G) + self.Q)
        self._state = EKFStateTF(mean=m_pred, cov=P_pred, t=self._state.t + 1)
        return m_pred.numpy(), P_pred.numpy()

    def update(self, z_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Measurement update step.

        Parameters
        ----------
        z_k : ndarray, shape (nz,)
            Observed measurement vector.

        Returns
        -------
        m_post : ndarray, shape (nx,)
            Posterior state mean.
        P_post : ndarray, shape (nx, nx)
            Posterior state covariance.
        """
        assert self._state is not None
        m, P = self._state.mean, self._state.cov
        z    = _to_f32(z_k)

        if self.jac_h is not None:
            H = _to_f32(self.jac_h(m.numpy()))
        else:
            H = _jacobian_tape(self.h, m)

        innov = z - self.h(m)
        S     = _symmetrise(H @ P @ tf.linalg.matrix_transpose(H) + self.R)
        if self.jitter > 0:
            S = S + self.jitter * tf.eye(tf.shape(S)[0], dtype=F32)

        PHt = P @ tf.linalg.matrix_transpose(H)
        K   = PHt @ tf.linalg.inv(S)
        m_post = m + tf.linalg.matvec(K, innov)

        nx = tf.shape(P)[0]
        I  = tf.eye(nx, dtype=F32)
        if self.joseph:
            ImKH   = I - K @ H
            P_post = _symmetrise(ImKH @ P @ tf.linalg.matrix_transpose(ImKH)
                                  + K @ self.R @ tf.linalg.matrix_transpose(K))
        else:
            P_post = _symmetrise((I - K @ H) @ P)

        self._state = EKFStateTF(mean=m_post, cov=P_post, t=self._state.t)
        return m_post.numpy(), P_post.numpy()

    def get_past_mean(self) -> np.ndarray:
        """Return the state mean from before the most recent predict call.

        Returns
        -------
        ndarray, shape (nx,)
            Prior state mean ``m_{t-1|t-1}``.
        """
        return self._past_mean


# TF UKF – step-by-step tracker
@dataclass
class UKFStateTF:
    mean: Tensor
    cov:  Tensor
    t:    int


class UKFTrackerTF:
    """Step-by-step Unscented Kalman Filter implemented in TensorFlow.

    Compatible with the ``GaussianTracker`` protocol used by
    ``EDHFlowPF_TF`` and ``LEDHFlowPF_TF``.

    Parameters
    ----------
    g : callable
        State-transition function ``(x, u) -> x_next`` (``tf.Tensor``).
    h : callable
        Observation function ``x -> z`` (``tf.Tensor -> tf.Tensor``).
    Q : ndarray or Tensor, shape (nx, nx)
        Process-noise covariance.
    R : ndarray or Tensor, shape (nz, nz)
        Measurement-noise covariance.
    alpha : float, optional
        Sigma-point spread parameter.  Default is ``1e-3``.
    beta : float, optional
        Distribution parameter (``2`` is optimal for Gaussians).
        Default is ``2.0``.
    kappa : float, optional
        Secondary scaling parameter.  Default is ``0.0``.
    jitter : float, optional
        Small diagonal added to covariance before Cholesky decomposition.
        Default is ``1e-6``.
    """

    def __init__(
        self,
        g:     Callable,
        h:     Callable,
        Q:     np.ndarray,
        R:     np.ndarray,
        *,
        alpha: float = 1e-3,
        beta:  float = 2.0,
        kappa: float = 0.0,
        jitter: float = 1e-6,
    ) -> None:
        self.g      = g
        self.h      = h
        self.Q      = _to_f32(Q)
        self.R      = _to_f32(R)
        self.alpha  = alpha
        self.beta   = beta
        self.kappa  = kappa
        self.jitter = jitter
        self._state: Optional[UKFStateTF] = None
        self._past_mean: Optional[np.ndarray] = None

    def init(self, mean0: np.ndarray, cov0: np.ndarray) -> UKFStateTF:
        self._state     = UKFStateTF(mean=_to_f32(mean0), cov=_to_f32(cov0), t=0)
        self._past_mean = mean0.copy()
        return self._state

    # ---- sigma-point weights ----

    def _weights(self, nx: int):
        lam = self.alpha ** 2 * (nx + self.kappa) - nx
        n   = 2 * nx + 1
        Wm  = np.full(n, 0.5 / (nx + lam), dtype=np.float32)
        Wc  = Wm.copy()
        Wm[0] = lam / (nx + lam)
        Wc[0] = lam / (nx + lam) + (1.0 - self.alpha ** 2 + self.beta)
        scale = math.sqrt(nx + lam)
        return _to_f32(Wm), _to_f32(Wc), scale

    def _sigma_points(self, m: Tensor, P: Tensor, scale: float) -> Tensor:
        """Compute the ``(2nx+1, nx)`` sigma-point matrix.

        Parameters
        ----------
        m : Tensor, shape (nx,)
            Current state mean.
        P : Tensor, shape (nx, nx)
            Current state covariance.
        scale : float
            Sigma-point scale ``sqrt(nx + λ)``.

        Returns
        -------
        Tensor, shape (2 nx + 1, nx)
            Sigma points.
        """
        nx = tf.shape(m)[0]
        L  = tf.linalg.cholesky(P + self.jitter * tf.eye(nx, dtype=F32))
        cols = tf.unstack(scale * L, axis=1)          # nx vectors of length nx
        pts  = [m] + [m + c for c in cols] + [m - c for c in cols]
        return tf.stack(pts)                           # (2nx+1, nx)

    # ---- GaussianTracker protocol ----

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict step using unscented transform.

        Returns
        -------
        m_pred : ndarray, shape (nx,)
            Predicted state mean.
        P_pred : ndarray, shape (nx, nx)
            Predicted state covariance.
        """
        assert self._state is not None, "Call init() first."
        self._past_mean = self._state.mean.numpy()
        m, P = self._state.mean, self._state.cov
        nx_int = int(m.shape[0])
        Wm, Wc, scale = self._weights(nx_int)

        sigma = self._sigma_points(m, P, scale)                # (2nx+1, nx)
        prop  = tf.map_fn(lambda s: self.g(s, None), sigma,
                          fn_output_signature=tf.TensorSpec([nx_int], F32))  # (2nx+1, nx)

        m_pred = tf.reduce_sum(Wm[:, None] * prop, axis=0)
        diff   = prop - m_pred[None, :]
        P_pred = _symmetrise(
            tf.einsum('i,ij,ik->jk', Wc, diff, diff) + self.Q
        )
        self._state = UKFStateTF(mean=m_pred, cov=P_pred, t=self._state.t + 1)
        return m_pred.numpy(), P_pred.numpy()

    def update(self, z_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Measurement update step using unscented transform.

        Parameters
        ----------
        z_k : ndarray, shape (nz,)
            Observed measurement vector.

        Returns
        -------
        m_post : ndarray, shape (nx,)
            Posterior state mean.
        P_post : ndarray, shape (nx, nx)
            Posterior state covariance.
        """
        assert self._state is not None
        m, P = self._state.mean, self._state.cov
        z    = _to_f32(z_k)
        nx_int = int(m.shape[0])
        nz_int = int(z.shape[0])
        Wm, Wc, scale = self._weights(nx_int)

        sigma = self._sigma_points(m, P, scale)                # (2nx+1, nx)
        z_pts = tf.map_fn(self.h, sigma,
                          fn_output_signature=tf.TensorSpec([nz_int], F32))   # (2nx+1, nz)

        z_pred = tf.reduce_sum(Wm[:, None] * z_pts, axis=0)   # (nz,)
        innov  = z - z_pred

        dz   = z_pts - z_pred[None, :]                        # (2nx+1, nz)
        dx   = sigma  - m[None, :]                            # (2nx+1, nx)
        Pzz  = _symmetrise(tf.einsum('i,ij,ik->jk', Wc, dz, dz) + self.R)
        Pxz  = tf.einsum('i,ij,ik->jk', Wc, dx, dz)          # (nx, nz)

        K = Pxz @ tf.linalg.inv(Pzz)
        m_post = m + tf.linalg.matvec(K, innov)
        P_post = _symmetrise(P - K @ Pzz @ tf.linalg.matrix_transpose(K))

        self._state = UKFStateTF(mean=m_post, cov=P_post, t=self._state.t)
        return m_post.numpy(), P_post.numpy()

    def get_past_mean(self) -> np.ndarray:
        """Return the state mean from before the most recent predict call.

        Returns
        -------
        ndarray, shape (nx,)
            Prior state mean ``m_{t-1|t-1}``.
        """
        return self._past_mean


# Automatic Jacobian via tf.GradientTape
def _jacobian_tape(fn: Callable, x: Tensor) -> Tensor:
    """Compute the Jacobian ``∂fn(x)/∂x`` using ``tf.GradientTape``.

    Parameters
    ----------
    fn : callable
        Differentiable function ``x -> y`` (``tf.Tensor -> tf.Tensor``).
    x : Tensor, shape (n,)
        Point at which to evaluate the Jacobian.

    Returns
    -------
    Tensor, shape (m, n)
        Jacobian matrix ``J[i, j] = ∂fn(x)_i / ∂x_j``.

    Raises
    ------
    RuntimeError
        If the Jacobian is ``None`` (i.e. ``fn`` is not differentiable).
    """
    x = tf.cast(x, F32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = fn(x)
    # GradientTape.jacobian (TF ≥ 2.x)
    J = tape.jacobian(y, x)   # (nz, nx)
    if J is None:
        raise RuntimeError("Jacobian is None; ensure fn is differentiable.")
    return J
