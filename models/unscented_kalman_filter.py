"""
Unscented Kalman Filter (UKF).

Implements a sigma-point Kalman filter for additive Gaussian noise SSMs::

    x_k  = g(x_{k-1}, u_{k-1}) + w_{k-1},   w ~ N(0, Q)
    z_k  = h(x_k)               + v_k,       v ~ N(0, R)

The UKF propagates 2*nx + 1 deterministically chosen sigma points through
the nonlinear functions g and h instead of linearising them.

Classes
-------
UKFState               – Gaussian posterior container.
UnscentedKalmanFilter  – UKF implementing :class:`~base_ssm.BaseFilter`.

Shared utilities from :mod:`base_ssm`
--------------------------------------
:func:`~base_ssm.symmetrise` is used to enforce symmetric covariances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:
    from models.base_ssm import (
        BaseFilter,
        GaussianFilterState,
        symmetrise,
    )
except ModuleNotFoundError:
    from base_ssm import (
        BaseFilter,
        GaussianFilterState,
        symmetrise,
    )


Array = np.ndarray
GFn = Callable[[Array, Optional[Array]], Array]
HFn = Callable[[Array], Array]


# State container
@dataclass
class UKFState(GaussianFilterState):
    """Gaussian posterior state for the UKF.

    Extends :class:`~base_ssm.GaussianFilterState` without adding new fields;
    the separate name keeps call-sites readable and allows isinstance checks.

    Parameters
    ----------
    mean : ndarray, shape (nx,)
        Posterior state mean.
    cov : ndarray, shape (nx, nx)
        Posterior state covariance.
    t : int
        Discrete time index of this posterior.
    """


# Core UKF
class UnscentedKalmanFilter(BaseFilter):
    """Unscented Kalman Filter for additive Gaussian noise SSMs.

    Derives from :class:`~base_ssm.BaseFilter`, providing a consistent
    ``initialize`` / ``predict`` / ``update`` / ``step`` / ``run`` interface.

    The additive-noise SSM is::

        x_k  = g(x_{k-1}, u_{k-1}) + w_{k-1},   w ~ N(0, Q)
        z_k  = h(x_k)               + v_k,       v ~ N(0, R)

    The UKF propagates 2*nx + 1 deterministically chosen sigma points through
    the (possibly nonlinear) functions g and h to obtain a second-order
    accurate Gaussian approximation of the predicted mean and covariance.

    Parameters
    ----------
    g : callable
        Process function g(x, u) → (nx,).
    h : callable
        Observation function h(x) → (nz,).
    Q : ndarray, shape (nx, nx)
        Process-noise covariance.
    R : ndarray, shape (nz, nz)
        Measurement-noise covariance.
    alpha : float, optional
        Primary sigma-point spread parameter in (0, 1]. Default is 1e-3.
    beta : float, optional
        Prior knowledge parameter (2 is optimal for Gaussian priors).
        Default is 2.0.
    kappa : float, optional
        Secondary spread parameter (often 0 or 3 - nx). Default is 0.0.
    jitter : float, optional
        Small diagonal regularisation added to covariance matrices before
        Cholesky decomposition. Default is 0.0.

    Notes
    -----
    Assumes additive Gaussian process and observation noise, and that the
    supplied process and observation functions preserve fixed state and
    observation dimensions across calls.
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
        if self.Q.shape != (self.nx, self.nx):
            raise ValueError("Q must be (nx, nx).")
        self.nz = int(self.R.shape[0])
        if self.R.shape != (self.nz, self.nz):
            raise ValueError("R must be (nz, nz).")

        # Unscented transform weights
        self._lambda = self.alpha**2 * (self.nx + self.kappa) - self.nx
        self._gamma = np.sqrt(self.nx + self._lambda)

        wm = np.full(2 * self.nx + 1, 1.0 / (2.0 * (self.nx + self._lambda)))
        wc = wm.copy()
        wm[0] = self._lambda / (self.nx + self._lambda)
        wc[0] = wm[0] + (1.0 - self.alpha**2 + self.beta)
        self.Wm = wm
        self.Wc = wc

    # ------------------------------------------------------------------
    # BaseFilter interface
    # ------------------------------------------------------------------

    def initialize(self, mean: Array, cov: Array) -> UKFState:
        """Create an initial UKF state from a Gaussian prior N(mean, cov).

        Parameters
        ----------
        mean : ndarray, shape (nx,)
            Initial state mean.
        cov : ndarray, shape (nx, nx)
            Initial state covariance.

        Returns
        -------
        UKFState
            Initial filter state at time t=0.
        """
        return UKFState(mean=np.asarray(mean, float), cov=np.asarray(cov, float), t=0)

    # helpers
    def _sigma_points(self, mean: Array, cov: Array) -> Array:
        """Construct 2*nx+1 sigma points around a Gaussian (mean, cov).

        Parameters
        ----------
        mean : ndarray, shape (nx,)
            Gaussian mean.
        cov : ndarray, shape (nx, nx)
            Gaussian covariance.

        Returns
        -------
        ndarray, shape (2*nx+1, nx)
            Sigma-point matrix.

        Notes
        -----
        Assumes ``cov`` is symmetric positive semidefinite up to the configured
        diagonal jitter.
        """
        mean = np.asarray(mean, float)
        cov = symmetrise(np.asarray(cov, float))

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
        """Run the UKF prediction (time-update) step.

        Propagates the sigma points through the process function g and
        reconstructs the predicted Gaussian.

        Parameters
        ----------
        state : UKFState
            Previous posterior state.
        u : ndarray, optional
            Control input u_{k-1}.  ``None`` if no input.

        Returns
        -------
        UKFState
            Predicted state with mean x_{k|k-1}, covariance P_{k|k-1},
            and time index state.t + 1.
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
        """Run the UKF measurement-update step.

        Propagates sigma points through the observation function h, builds
        the innovation covariance S and cross-covariance P_{xz}, and
        computes the Kalman gain via a Cholesky solve for stability.

        Parameters
        ----------
        pred : UKFState
            Predicted state returned by :meth:`predict`.
        z : ndarray, shape (nz,)
            Measurement vector z_k.

        Returns
        -------
        UKFState
            Posterior state with mean x_{k|k}, covariance P_{k|k},
            and time index pred.t.
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
        S = symmetrise(S)
        L = np.linalg.cholesky(S + self.jitter * np.eye(self.nz))
        # Compute K = Pxz @ S^{-1} using two triangular solves
        K = np.linalg.solve(L.T, np.linalg.solve(L, Pxz.T)).T

        x_post = pred.mean + K @ (np.asarray(z, float) - z_pred)
        P_post = symmetrise(pred.cov - K @ S @ K.T)

        return UKFState(mean=x_post, cov=P_post, t=pred.t)

    def step(self, state: UKFState, z: Array, u: Optional[Array] = None) -> UKFState:
        """Run a full UKF step: predict then update.

        Parameters
        ----------
        state : UKFState
            Previous posterior state.
        z : ndarray, shape (nz,)
            Measurement at the next time step.
        u : ndarray, optional
            Control input for the process model.  ``None`` if no input.

        Returns
        -------
        UKFState
            Updated posterior state at the next time step.
        """
        pred = self.predict(state, u=u)
        return self.update(pred, z=z)