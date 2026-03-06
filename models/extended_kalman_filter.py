"""
Extended Kalman Filter (EKF).

Implements a generic EKF with pluggable nonlinear process and measurement
models, optional analytic Jacobians, and a numerical Jacobian fallback.

The additive-noise SSM is::

    x_k  = g(x_{k-1}, u_{k-1}) + w_{k-1},   w ~ N(0, Q)
    z_k  = h(x_k)               + v_k,       v ~ N(0, R)

The EKF linearises g and h around the current posterior estimate to
propagate a Gaussian approximation.

Classes
-------
EKFState           – Gaussian posterior container.
ExtendedKalmanFilter – EKF implementing :class:`~base_ssm.BaseFilter`.

Shared utilities from :mod:`base_ssm`
--------------------------------------
:func:`~base_ssm.numerical_jacobian` replaces the private finite-difference
helpers ``numerical_jacobian_g`` and ``numerical_jacobian_h``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:
    from models.base_ssm import (
        BaseFilter,
        BaseFilterState,
        GaussianFilterState,
        numerical_jacobian as _num_jac,
        symmetrise,
    )
except ModuleNotFoundError:
    from base_ssm import (
        BaseFilter,
        BaseFilterState,
        GaussianFilterState,
        numerical_jacobian as _num_jac,
        symmetrise,
    )


Array = np.ndarray
JacFnG = Callable[[Array, Optional[Array]], Array]
JacFnH = Callable[[Array], Array]
GFn = Callable[[Array, Optional[Array]], Array]
HFn = Callable[[Array], Array]


@dataclass
class EKFState(GaussianFilterState):
    """Gaussian posterior state for the EKF.

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

    Notes
    -----
    Assumes ``cov`` represents the Gaussian uncertainty associated with
    ``mean`` at time ``t``.
    """


def numerical_jacobian_g(
    g: GFn,
    x: Array,
    u: Optional[Array],
    eps: float = 1e-6,
) -> Array:
    """Finite-difference Jacobian of the process model g w.r.t. x.

    Delegates to :func:`base_ssm.numerical_jacobian`; kept as a named
    convenience wrapper for backward compatibility.

    Parameters
    ----------
    g : callable
        Process function g(x, u) → (nx,).
    x : ndarray, shape (nx,)
        Expansion point.
    u : ndarray or None
        Control input, forwarded as a positional argument to ``g``.
    eps : float, optional
        Finite-difference step size. Default is 1e-6.

    Returns
    -------
    ndarray, shape (nx, nx)
        Jacobian ∂g/∂x evaluated at (x, u).

    Notes
    -----
    Assumes ``g`` returns a fixed-length state vector and is locally smooth
    around ``x``.
    """
    return _num_jac(lambda xp: g(xp, u), x, eps=eps)


def numerical_jacobian_h(
    h: HFn,
    x: Array,
    eps: float = 1e-6,
) -> Array:
    """Finite-difference Jacobian of the observation model h w.r.t. x.

    Delegates to :func:`base_ssm.numerical_jacobian`; kept as a named
    convenience wrapper for backward compatibility.

    Parameters
    ----------
    h : callable
        Observation function h(x) → (nz,).
    x : ndarray, shape (nx,)
        Expansion point.
    eps : float, optional
        Finite-difference step size. Default is 1e-6.

    Returns
    -------
    ndarray, shape (nz, nx)
        Jacobian ∂h/∂x evaluated at x.

    Notes
    -----
    Assumes ``h`` returns a fixed-length observation vector and is locally
    smooth around ``x``.
    """
    return _num_jac(h, x, eps=eps)


class ExtendedKalmanFilter(BaseFilter):
    """Extended Kalman Filter with additive Gaussian noises.

    Derives from :class:`~base_ssm.BaseFilter`, providing a consistent
    ``initialize`` / ``predict`` / ``update`` / ``step`` / ``run`` interface.

    The additive-noise SSM is::

        x_k  = g(x_{k-1}, u_{k-1}) + w_{k-1},   w ~ N(0, Q)
        z_k  = h(x_k)               + v_k,       v ~ N(0, R)

    The EKF linearises g and h at the current estimate:

    - G_k = ∂g/∂x  evaluated at (x_{k-1|k-1}, u_{k-1})
    - H_k = ∂h/∂x  evaluated at x_{k|k-1}

    Parameters
    ----------
    g : callable
        Process (motion) function g(x, u) → (nx,).
    h : callable
        Observation function h(x) → (nz,).
    Q : ndarray, shape (nx, nx)
        Process-noise covariance.
    R : ndarray, shape (nz, nz)
        Measurement-noise covariance.
    jac_g : callable, optional
        Analytic Jacobian jac_g(x, u) → (nx, nx).  If ``None``, a
        finite-difference approximation is used.
    jac_h : callable, optional
        Analytic Jacobian jac_h(x) → (nz, nx).  If ``None``, a
        finite-difference approximation is used.
    joseph : bool, optional
        If ``True``, use the numerically stable Joseph-form covariance
        update P = (I - K H) P (I - K H)^T + K R K^T.  Default is ``False``.
    jitter : float, optional
        Small positive value added to the innovation covariance diagonal
        for numerical stability.  Default is 0.0.

    Notes
    -----
    Assumes additive Gaussian process and observation noise and that the
    supplied callables preserve fixed state and observation dimensions.
    """

    def __init__(
        self,
        g: GFn,
        h: HFn,
        Q: Array,
        R: Array,
        jac_g: Optional[JacFnG] = None,
        jac_h: Optional[JacFnH] = None,
        *,
        joseph: bool = False,
        jitter: float = 0.0,
    ) -> None:
        self.g = g
        self.h = h
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.jac_g = jac_g
        self.jac_h = jac_h
        self.joseph = bool(joseph)
        self.jitter = float(jitter)

        nx = self.Q.shape[0]
        if self.Q.shape != (nx, nx):
            raise ValueError("Q must be square (nx, nx).")
        nz = self.R.shape[0]
        if self.R.shape != (nz, nz):
            raise ValueError("R must be square (nz, nz).")

    # ------------------------------------------------------------------
    # BaseFilter interface
    # ------------------------------------------------------------------

    def initialize(self, mean: Array, cov: Array) -> EKFState:
        """Create an initial EKF state from a Gaussian prior N(mean, cov).

        Parameters
        ----------
        mean : ndarray, shape (nx,)
            Initial state mean.
        cov : ndarray, shape (nx, nx)
            Initial state covariance.

        Returns
        -------
        EKFState
            Initial filter state at time t=0.

        Notes
        -----
        Assumes ``cov`` is square and compatible with ``mean``.
        """
        return EKFState(mean=np.asarray(mean, float), cov=np.asarray(cov, float), t=0)

    # ------------------------- core EKF ops -------------------------

    def predict(self, state: EKFState, u: Optional[Array] = None) -> EKFState:
        """Run the EKF prediction (time-update) step.

        Propagates the Gaussian posterior through the linearised process model.

        Parameters
        ----------
        state : EKFState
            Previous posterior state (mean, cov, t).
        u : ndarray, optional
            Control input u_{k-1}.  ``None`` if no input.

        Returns
        -------
        EKFState
            Predicted state with mean x_{k|k-1}, covariance P_{k|k-1},
            and time index state.t + 1.

        Notes
        -----
        Assumes the process Jacobian returned or approximated at ``state.mean``
        has shape ``(nx, nx)``.
        """
        x = np.asarray(state.mean, dtype=float)
        P = np.asarray(state.cov, dtype=float)
        nx = x.size

        x_pred = np.asarray(self.g(x, u), dtype=float)

        G = (
            self.jac_g(x, u)
            if self.jac_g is not None
            else numerical_jacobian_g(self.g, x, u)
        )
        if G.shape != (nx, nx):
            raise ValueError("jac_g must return shape (nx, nx).")

        P_pred = G @ P @ G.T + self.Q

        return EKFState(mean=x_pred, cov=P_pred, t=state.t + 1)

    def update(self, pred: EKFState, z: Array) -> EKFState:
        """Run the EKF measurement-update step.

        Corrects the predicted state with the linearised observation model.

        Parameters
        ----------
        pred : EKFState
            Predicted state returned by :meth:`predict`.
        z : ndarray, shape (nz,)
            Observation vector z_k.

        Returns
        -------
        EKFState
            Posterior state with mean x_{k|k}, covariance P_{k|k},
            and time index pred.t.

        Notes
        -----
        Assumes ``z`` is compatible with the observation dimension and the
        innovation covariance is invertible up to the configured jitter.
        """
        x_pred = np.asarray(pred.mean, dtype=float)
        P_pred = np.asarray(pred.cov, dtype=float)
        z = np.asarray(z, dtype=float)

        nz = z.size
        H = (
            self.jac_h(x_pred)
            if self.jac_h is not None
            else numerical_jacobian_h(self.h, x_pred)
        )
        if H.shape[0] != nz or H.shape[1] != x_pred.size:
            raise ValueError("jac_h must return shape (nz, nx).")

        z_pred = np.asarray(self.h(x_pred), dtype=float)
        y = z - z_pred                                 # innovation
        S = H @ P_pred @ H.T + self.R                  # innovation cov
        if self.jitter > 0.0:
            S = S + self.jitter * np.eye(nz)

        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Posterior mean
        x_post = x_pred + K @ y

        # Posterior covariance
        if self.joseph:
            I = np.eye(P_pred.shape[0])
            A = I - K @ H
            P_post = A @ P_pred @ A.T + K @ self.R @ K.T
        else:
            P_post = (np.eye(P_pred.shape[0]) - K @ H) @ P_pred

        return EKFState(mean=x_post, cov=P_post, t=pred.t)

    def step(self, state: EKFState, z: Array, u: Optional[Array] = None) -> EKFState:
        """Run a full EKF step: predict then update.

        Parameters
        ----------
        state : EKFState
            Previous posterior state.
        z : ndarray, shape (nz,)
            Measurement at the next time step.
        u : ndarray, optional
            Control input for the process model.  ``None`` if no input.

        Returns
        -------
        EKFState
            Updated posterior state at the next time step.

        Notes
        -----
        Equivalent to calling :meth:`predict` followed by :meth:`update`.
        """
        pred = self.predict(state, u=u)
        return self.update(pred, z=z)