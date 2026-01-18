"""
Unscented Kalman Filter (UKF) 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


Array = np.ndarray
GFn = Callable[[Array, Optional[Array]], Array]
HFn = Callable[[Array], Array]


# State container
@dataclass
class UKFState:
    """Container for UKF posterior state.

    Parameters
    ----------
    mean : np.ndarray
        Posterior mean of shape (nx,).
    cov : np.ndarray
        Posterior covariance of shape (nx, nx).
    t : int
        Discrete time index of this posterior.
    """

    mean: Array
    cov: Array
    t: int


# Core UKF
class UnscentedKalmanFilter:
    """Unscented Kalman Filter for additive Gaussian noises.

    The dynamical system is:
        x_k = g(x_{k-1}, u_{k-1}) + w_{k-1},   w ~ N(0, Q)
        z_k = h(x_k)                + v_k,     v ~ N(0, R)

    This class implements the UKF using 2*nx+1 sigma points.

    Parameters
    ----------
    g : callable
        Process/motion function g(x, u) -> (nx,).
    h : callable
        Measurement function h(x) -> (nz,).
    Q : ndarray
        Process noise covariance (nx, nx).
    R : ndarray
        Measurement noise covariance (nz, nz).
    alpha : float, optional
        Primary spread parameter in (0, 1]. Default is 1e-3 or 0.1.
    beta : float, optional
        Prior knowledge parameter (2 for Gaussian). Default is 2.0.
    kappa : float, optional
        Secondary spread parameter (often 0 or 3-nx). Default is 0.0.
    jitter : float, optional
        Small diagonal added to covariance matrices before Cholesky. Default is 0.0.
    """

    def __init__(
        self,
        g: GFn,
        h: HFn,
        Q: Array,
        R: Array,
        *,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        jitter: float = 0.0,
    ) -> None:
        self.g = g
        self.h = h
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.jitter = float(jitter)

        # Dimensions and static checks
        self.nx = int(self.Q.shape[0])
        assert self.Q.shape == (self.nx, self.nx), "Q must be (nx, nx)."
        self.nz = int(self.R.shape[0])
        assert self.R.shape == (self.nz, self.nz), "R must be (nz, nz)."

        # Unscented transform weights
        self._lambda = self.alpha**2 * (self.nx + self.kappa) - self.nx
        self._gamma = np.sqrt(self.nx + self._lambda)

        wm = np.full(2 * self.nx + 1, 1.0 / (2.0 * (self.nx + self._lambda)))
        wc = wm.copy()
        wm[0] = self._lambda / (self.nx + self._lambda)
        wc[0] = wm[0] + (1.0 - self.alpha**2 + self.beta)
        self.Wm = wm
        self.Wc = wc

    # helpers
    def _sigma_points(self, mean: Array, cov: Array) -> Array:
        """Construct sigma points around a Gaussian (mean, cov)."""
        mean = np.asarray(mean, float)
        cov = np.asarray(cov, float)
        cov = 0.5 * (cov + cov.T)  # symmetrize

        try:
            L = np.linalg.cholesky(cov + self.jitter * np.eye(self.nx))
        except np.linalg.LinAlgError:
            # If still failing, inflate diagonal slightly
            eps = max(self.jitter, 1e-12)
            L = np.linalg.cholesky(cov + eps * np.eye(self.nx))

        X = np.empty((2 * self.nx + 1, self.nx), dtype=float)
        X[0] = mean
        for i in range(self.nx):
            col = self._gamma * L[:, i]
            X[i + 1] = mean + col
            X[i + 1 + self.nx] = mean - col
        return X

    # core UKF ops 
    def predict(self, state: UKFState, u: Optional[Array] = None) -> UKFState:
        """Run the UKF prediction step (unscented transform through g).

        Parameters
        ----------
        state
            Previous posterior state.
        u
            Optional control input u_{k-1}.

        Returns
        -------
        Predicted state UKFState(mean= x_{k|k-1}, cov= P_{k|k-1}, t= state.t + 1).
        """
        X = self._sigma_points(state.mean, state.cov)
        X_prop = np.array([self.g(xi, u) for xi in X])

        x_pred = np.sum(self.Wm[:, None] * X_prop, axis=0)
        P_pred = self.Q.copy()
        DX = X_prop - x_pred
        for i in range(X_prop.shape[0]):
            P_pred += self.Wc[i] * np.outer(DX[i], DX[i])

        return UKFState(mean=x_pred, cov=P_pred, t=state.t + 1)

    def update(self, pred: UKFState, z: Array) -> UKFState:
        """Run the UKF measurement update (unscented transform through h).

        Parameters
        ----------
        pred: Predicted state from `predict`.
        z: Measurement vector z_k of shape (nz,).

        Returns
        -------
        Posterior state UKFState(mean= x_{k|k}, cov= P_{k|k}, t= pred.t).
        """
        X = self._sigma_points(pred.mean, pred.cov)
        Z = np.array([self.h(xi) for xi in X])

        z_pred = np.sum(self.Wm[:, None] * Z, axis=0)

        # Innovation covariance S and cross-covariance Pxz
        S = self.R.copy()
        DZ = Z - z_pred
        for i in range(Z.shape[0]):
            S += self.Wc[i] * np.outer(DZ[i], DZ[i])

        DX = X - pred.mean
        Pxz = np.zeros((self.nx, self.nz), dtype=float)
        for i in range(Z.shape[0]):
            Pxz += self.Wc[i] * np.outer(DX[i], DZ[i])

        # Kalman gain via Cholesky-based solve for stability
        S = 0.5 * (S + S.T)
        L = np.linalg.cholesky(S + self.jitter * np.eye(self.nz))
        # Compute K = Pxz @ S^{-1} using two triangular solves
        K = np.linalg.solve(L.T, np.linalg.solve(L, Pxz.T)).T

        x_post = pred.mean + K @ (np.asarray(z, float) - z_pred)
        P_post = pred.cov - K @ S @ K.T
        P_post = 0.5 * (P_post + P_post.T)  # numerical symmetry

        return UKFState(mean=x_post, cov=P_post, t=pred.t)

    def step(self, state: UKFState, z: Array, u: Optional[Array] = None) -> UKFState:
        """Run a full UKF step (predict then update)."""
        pred = self.predict(state, u=u)
        return self.update(pred, z=z)