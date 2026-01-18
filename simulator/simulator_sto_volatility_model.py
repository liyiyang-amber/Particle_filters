from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SV1DResults:
    """Simulation results for the 1-D stochastic volatility model.

    Parameters
    ----------
        X: np.ndarray
            Latent states of shape (n,).
        Y: np.ndarray
            Observations of shape (n,).
        alpha: float
            AR(1) coefficient |alpha| < 1.
        sigma: float
            State noise std (\sigma >= 0).
        beta: float
            Observation scale (\beta >= 0).
        n: int
            Number of time steps.
        seed (int | None): RNG seed used.
    """
    X: np.ndarray
    Y: np.ndarray
    alpha: float
    sigma: float
    beta: float
    n: int
    seed: Optional[int] = None

    def save(self, filename: str) -> None:
        """Save results to a .npz file."""
        np.savez(
            filename,
            X=self.X,
            Y=self.Y,
            alpha=self.alpha,
            sigma=self.sigma,
            beta=self.beta,
            n=self.n,
            seed=self.seed,
        )


def simulate_sv_1d(
    n: int,
    alpha: float,
    sigma: float,
    beta: float,
    *,
    seed: Optional[int] = None,
    x0: Optional[float] = None,
) -> SV1DResults:
    """Simulate the 1-D stochastic volatility model.

    Model:
        X_1 ~ N(0, \sigma^2 / (1 - \alpha^2)) if x0 is None
        X_t = \alpha X_{t-1} + \sigma V_t,  V_t ~ N(0, 1)
        Y_t = \beta \exp(0.5 X_t) W_t, W_t ~ N(0, 1)

    Parameters
    ----------
    n : int
        Number of time steps (n >= 1).
    alpha : float
        AR(1) coefficient (|\alpha| < 1 for stationarity).
    sigma : float
        Process noise standard deviation (\sigma >= 0).
    beta : float
        Observation scale (\beta >= 0).
    seed : int, optional
        RNG seed for reproducibility. Default is None.
    x0 : float, optional
        Initial state. If None, draw from the stationary N(0, \sigma^2/(1-\alpha^2)).
        Default is None.

    Returns
    -------
    SV1DResults
        Results with arrays X (n,) and Y (n,).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not np.isfinite(alpha) or abs(alpha) >= 1:
        raise ValueError("alpha must be finite with |alpha| < 1 for stationarity.")
    if sigma < 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be a finite, nonnegative scalar.")
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("beta must be a finite, nonnegative scalar.")

    rng = np.random.default_rng(seed)

    X = np.empty(n, dtype=float)
    Y = np.empty(n, dtype=float)

    # Initialize X_1
    if x0 is None:
        var0 = sigma**2 / (1.0 - alpha**2)
        # Guard tiny negative due to roundoff when alpha ~ 1
        var0 = max(var0, 0.0)
        X[0] = rng.normal(0.0, np.sqrt(var0))
    else:
        X[0] = float(x0)

    # Simulate states
    if n > 1:
        V = rng.standard_normal(n - 1)
        for t in range(1, n):
            X[t] = alpha * X[t - 1] + sigma * V[t - 1]

    # Simulate observations
    W = rng.standard_normal(n)
    s = beta * np.exp(0.5 * X)  
    Y[:] = s * W

    return SV1DResults(X=X, Y=Y, alpha=alpha, sigma=sigma, beta=beta, n=n, seed=seed)
