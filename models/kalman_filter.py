"""
Kalman filter implementation.
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union
import numpy as np
from numpy.typing import NDArray
Array = NDArray[np.float64]


@dataclass
class KFResults:
    """Outputs of the general Kalman filter.

    Parameters
    ----------
    x_pred : ndarray, shape (N, nx)
        Predicted (prior) state means, \hat{x}_k^-.
    P_pred : ndarray, shape (N, nx, nx)
        Predicted state covariances, P_k^-.
    x_filt : ndarray, shape (N, nx)
        Filtered (posterior) state means, \hat{x}_k^+.
    P_filt : ndarray, shape (N, nx, nx)
        Filtered state covariances, P_k^+.
    K : ndarray, shape (N, nx, ny)
        Kalman gains K_k.
    innov : ndarray, shape (N, ny)
        Innovations \tilde{y}_k = y_k - H_k \hat{x}_k^-.
    S : ndarray, shape (N, ny, ny)
        Innovation covariances S_k = H_k P_k^- H_k^T + R_k.
    loglik : float
        Total Gaussian log-likelihood of observations under the model.
    """

    x_pred: Array
    P_pred: Array
    x_filt: Array
    P_filt: Array
    K: Array
    innov: Array
    S: Array
    loglik: float


def _as_sequence(M: Union[Array, Sequence[Array]], N: int, name: str) -> Sequence[Array]:
    """Return a per-time-step sequence for matrix/tensor parameters.
    If `M` is a single array, repeat it N times; otherwise validate length.
    """
    if isinstance(M, np.ndarray):
        return [M] * N
    Mseq = list(M)
    if len(Mseq) != N:
        raise ValueError(f"{name} must have length N={N} (got {len(Mseq)}).")
    return Mseq


def _chol_solve(L: Array, B: Array) -> Array:
    """Solve (L L^T) X = B for X using triangular solves with Cholesky factor L.

    Parameters
    ----------
    L : ndarray, shape (m, m)
        Lower-triangular Cholesky factor.
    B : ndarray, shape (m, k)
        Right-hand side.

    Returns
    -------
    ndarray
        Solution X of shape (m, k).
    """
    # Solve L Z = B
    Z = np.linalg.solve(L, B)
    # Solve L^T X = Z
    X = np.linalg.solve(L.T, Z)
    return X


def kalman_filter_general(
    Y: Array,
    Phi: Union[Array, Sequence[Array]],
    H: Union[Array, Sequence[Array]],
    Gamma: Union[Array, Sequence[Array]],
    Q: Union[Array, Sequence[Array]],
    R: Union[Array, Sequence[Array]],
    *,
    B: Optional[Union[Array, Sequence[Array]]] = None,
    U: Optional[Array] = None,
    x0: Array,
    P0: Array,
    use_joseph: bool = False,
    jitter: float = 1e-9,
) -> KFResults:
    """
    Run the general (time-varying) Kalman filter using your notes' notation.

    The model is
        x_k   = Phi_{k-1} x_{k-1} + B_{k-1} u_{k-1} + Gamma_{k-1} w_{k-1},
        y_k   = H_k x_k + v_k,

    with E[w_k]=E[v_k]=0,  E[w_k w_ell^T] = Q_k δ_{k_ell},  E[v_k v_ell^T] = R_k δ_{k_ell}.

    Parameters
    ----------
    Y : ndarray, shape (N, ny)
        Observation sequence {y_k}.
    Phi : ndarray or sequence of ndarrays
        State transition matrices Phi_k (nx*nx). If a single array is provided, it
        is assumed time-invariant and reused for all k.
    H : ndarray or sequence of ndarrays
        Observation matrices H_k (ny*nx).
    Gamma : ndarray or sequence of ndarrays
        Process-noise input matrices Gamma_k (nx*n_w).
    Q : ndarray or sequence of ndarrays
        Process-noise covariances Q_k (n_w*n_w).
    R : ndarray or sequence of ndarrays
        Measurement-noise covariances R_k (ny*ny).
    B : ndarray or sequence of ndarrays, optional
        Control matrices B_k (nx*n_u). If None, treated as zeros.
    U : ndarray, optional
        Control inputs u_k, shape (N, n_u). If None, treated as zeros.
    x0 : ndarray, shape (nx,)
        Initial mean \hat{x}_0^+.
    P0 : ndarray, shape (nx, nx)
        Initial covariance P_0^+.
    use_joseph : bool, default=False
        If True, use the Joseph-stabilized covariance update.
    jitter : float, default=1e-9
        Non-negative diagonal jitter added to S_k to ensure PD for Cholesky.

    Returns
    -------
    KFResults
        Dataclass containing priors, posteriors, gains, innovations, S_k, and
        total log-likelihood.
    """

    if Y.ndim != 2:
        raise ValueError("Y must be 2D with shape (N, ny).")
    N, ny = Y.shape

    # Expand time-invariant parameters to sequences of length N.
    Phi_seq = _as_sequence(Phi, N, "Phi")
    H_seq = _as_sequence(H, N, "H")
    Gamma_seq = _as_sequence(Gamma, N, "Gamma")
    Q_seq = _as_sequence(Q, N, "Q")
    R_seq = _as_sequence(R, N, "R")

    nx = x0.shape[0]

    if B is None:
        # Zero control matrices per time step.
        B_seq = [np.zeros((nx, 1), dtype=float) for _ in range(N)]
        nu = 1
    else:
        B_seq = _as_sequence(B, N, "B")
        nu = B_seq[0].shape[1]

    if U is None:
        U_arr = np.zeros((N, nu), dtype=float)
    else:
        if U.shape[0] != N or U.shape[1] != nu:
            raise ValueError("U must have shape (N, n_u) matching B_k.")
        U_arr = U

    # Pre-allocate outputs.
    x_pred = np.zeros((N, nx), dtype=float)
    x_filt = np.zeros((N, nx), dtype=float)
    P_pred = np.zeros((N, nx, nx), dtype=float)
    P_filt = np.zeros((N, nx, nx), dtype=float)
    K_all = np.zeros((N, nx, ny), dtype=float)
    innov = np.zeros((N, ny), dtype=float)
    S_all = np.zeros((N, ny, ny), dtype=float)

    I = np.eye(nx, dtype=float)

    # Initial posterior.
    m = np.asarray(x0, dtype=float).reshape(nx)
    P = np.asarray(P0, dtype=float)

    loglik = 0.0

    for k in range(N):
        Phi_k = Phi_seq[k]
        H_k = H_seq[k]
        Gamma_k = Gamma_seq[k]
        Q_k = Q_seq[k]
        R_k = R_seq[k]
        B_k = B_seq[k]
        u_k = U_arr[k]

        # ---------- Prediction (time update) ----------
        m_minus = Phi_k @ m + B_k @ u_k
        P_minus = Phi_k @ P @ Phi_k.T + Gamma_k @ Q_k @ Gamma_k.T

        # Store priors.
        x_pred[k] = m_minus
        P_pred[k] = P_minus

        # ---------- Innovation ----------
        y_k = Y[k]
        nu_k = y_k - (H_k @ m_minus)
        S_k = H_k @ P_minus @ H_k.T + R_k
        if jitter > 0.0:
            S_k = S_k + jitter * np.eye(S_k.shape[0])

        # Cholesky factor of S_k
        try:
            L = np.linalg.cholesky(S_k)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Cholesky failed at step {k}; try larger 'jitter'."
            ) from e

        # K_k = P^- H^T S^{-1} via Cholesky solves 
        # Solve S X = (H P^-)^T for X^T -> K = (P^- H^T) S^{-1}
        PHt = P_minus @ H_k.T
        K_k = _chol_solve(L, PHt.T).T

        # Update state
        m_plus = m_minus + K_k @ nu_k

        # Update covariance
        if use_joseph:
            P_plus = (I - K_k @ H_k) @ P_minus @ (I - K_k @ H_k).T + K_k @ R_k @ K_k.T
        else:
            P_plus = P_minus - K_k @ (H_k @ P_minus)

        # Per-step log-likelihood using Cholesky:
        # log N(nu_k; 0, S_k) = -0.5*(nu^T S^{-1} nu + log|S| + ny*log 2\pi)
        alpha = _chol_solve(L, nu_k.reshape(-1, 1))  # S^{-1} nu
        quad = float(nu_k @ alpha.ravel())
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        loglik += -0.5 * (quad + logdet + ny * np.log(2.0 * np.pi))

        # Store posteriors & diagnostics
        x_filt[k] = m_plus
        P_filt[k] = P_plus
        K_all[k] = K_k
        innov[k] = nu_k
        S_all[k] = S_k

        # Forward
        m, P = m_plus, P_plus

    return KFResults(
        x_pred=x_pred,
        P_pred=P_pred,
        x_filt=x_filt,
        P_filt=P_filt,
        K=K_all,
        innov=innov,
        S=S_all,
        loglik=loglik,
    )