"""
Extended Kalman Filter (EKF).

This module implements a generic Extended Kalman Filter with pluggable motion
and measurement models, their Jacobians, and optional numerical Jacobian
fallbacks. 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


Array = np.ndarray
JacFnG = Callable[[Array, Optional[Array]], Array]
JacFnH = Callable[[Array], Array]
GFn = Callable[[Array, Optional[Array]], Array]
HFn = Callable[[Array], Array]


@dataclass
class EKFState:
    """Container for EKF posterior state.

    Attributes:
        mean (np.ndarray): Posterior state mean (shape: (nx,)).
        cov (np.ndarray): Posterior state covariance (shape: (nx, nx)).
        t (int): Discrete time index of this posterior.
    """

    mean: Array
    cov: Array
    t: int


def numerical_jacobian_g(
    g: GFn,
    x: Array,
    u: Optional[Array],
    eps: float = 1e-6,
) -> Array:
    """Compute a finite-difference Jacobian of the motion model w.r.t. x.

    Args:
        g: Motion function g(x, u).
        x: Expansion point (nx,).
        u: Control input or None.
        eps: Finite-difference step.

    Returns:
        (nx, nx) Jacobian matrix.
    """
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(g(x, u), dtype=float)
    nx = x.size
    J = np.zeros((y0.size, nx), dtype=float)
    for j in range(nx):
        dx = np.zeros(nx, dtype=float)
        dx[j] = eps
        J[:, j] = (g(x + dx, u) - y0) / eps
    return J


def numerical_jacobian_h(
    h: HFn,
    x: Array,
    eps: float = 1e-6,
) -> Array:
    """Compute a finite-difference Jacobian of the measurement model w.r.t. x.

    Args:
        h: Measurement function h(x).
        x: Expansion point (nx,).
        eps: Finite-difference step.

    Returns:
        (nz, nx) Jacobian matrix.
    """
    x = np.asarray(x, dtype=float)
    z0 = np.asarray(h(x), dtype=float)
    nx = x.size
    J = np.zeros((z0.size, nx), dtype=float)
    for j in range(nx):
        dx = np.zeros(nx, dtype=float)
        dx[j] = eps
        J[:, j] = (h(x + dx) - z0) / eps
    return J


class ExtendedKalmanFilter:
    """Extended Kalman Filter with additive Gaussian noises.

    The process and measurement models are:
        x_k   = g(x_{k-1}, u_{k-1}) + w_{k-1},   w_{k-1} ~ N(0, Q)
        z_k   = h(x_k)                + v_k,     v_k     ~ N(0, R)

    where g and h can be nonlinear. The Jacobians are:
        G_k = ∂g/∂x evaluated at (x_{k-1|k-1}, u_{k-1})
        H_k = ∂h/∂x evaluated at x_{k|k-1}

    Args:
        g: Motion function g(x, u) → (nx,).
        h: Measurement function h(x) → (nz,).
        Q: Process noise covariance (nx, nx).
        R: Measurement noise covariance (nz, nz).
        jac_g: Optional analytic Jacobian function for g. If None, uses
            finite-difference numerical Jacobian.
        jac_h: Optional analytic Jacobian function for h. If None, uses
            finite-difference numerical Jacobian.
        joseph: If True, use Joseph-stabilized covariance update.
        jitter: Small positive value added to innovation covariance diagonal
            for numerical stability.
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
        assert self.Q.shape == (nx, nx), "Q must be square."
        nz = self.R.shape[0]
        assert self.R.shape == (nz, nz), "R must be square."

    # ------------------------- core EKF ops -------------------------

    def predict(self, state: EKFState, u: Optional[Array] = None) -> EKFState:
        """Run the EKF prediction step.

        Args:
            state: Previous posterior state (mean, cov, t).
            u: Optional control input u_{k-1}.

        Returns:
            Predicted state EKFState(mean= x_{k|k-1}, cov= P_{k|k-1}, t= state.t + 1).
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
        """Run the EKF measurement update.

        Args:
            pred: Predicted state (from `predict`).
            z: Observation vector z_k (shape: (nz,)).

        Returns:
            Posterior state EKFState(mean= x_{k|k}, cov= P_{k|k}, t= pred.t).
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
        """Run a full EKF step (predict then update).

        Args:
            state: Previous posterior state.
            z: Measurement at the next time.
            u: Optional control input for the motion model.

        Returns:
            Updated EKFState at the new time.
        """
        pred = self.predict(state, u=u)
        return self.update(pred, z=z)