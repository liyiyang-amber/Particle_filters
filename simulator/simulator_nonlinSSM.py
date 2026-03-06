from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class NonlinearSSMSimulationResult:
    """Container for simulated non-linear state-space model data.
    
    This corresponds to the standard benchmark model from the SMC literature:
        X_n = X_{n-1}/2 + 25*X_{n-1}/(1 + X_{n-1}^2) + 8*cos(1.2*n) + V_n
        Y_n = X_n^2 / 20 + W_n
    where V_n ~ N(0, σ_V^2) and W_n ~ N(0, σ_W^2).

    Notes
    -----
    Assumes ``X`` and ``Y`` are one-dimensional, time-ordered arrays from a
    single simulation run.
    """
    X: NDArray[np.float64]
    """Array of shape (N,) containing latent states x_1, ..., x_N."""
    Y: NDArray[np.float64]
    """Array of shape (N,) containing observations y_1, ..., y_N."""
    sigma_v: float
    """Standard deviation of process noise V_n."""
    sigma_w: float
    """Standard deviation of measurement noise W_n."""
    
    def to_file(
        self,
        path: str,
        format: str = "npz",
        overwrite: bool = False,
    ) -> None:
        """Save the simulated data to a file.

        Parameters
        ----------
        path : str
            Destination file path (without extension for .npz format).
        format : str, default='npz'
            Output format.
        overwrite : bool, default=False
            If False, raises an error when the file already exists.

        Returns
        -------
        None
            Writes the simulation arrays and scalar parameters to disk.

        Raises
        ------
        FileExistsError
            If overwrite is False and the target file already exists.

        Notes
        -----
        Assumes the destination directory already exists. The ``format``
        parameter is accepted for API consistency although only ``npz`` output
        is currently implemented.
        """
        target = path if path.endswith(".npz") else f"{path}.npz"
        if os.path.exists(target) and not overwrite:
            raise FileExistsError(f"File already exists: {target}")
        np.savez(
            target,
            X=self.X,
            Y=self.Y,
            sigma_v=self.sigma_v,
            sigma_w=self.sigma_w,
        )


def simulate_nonlinear_ssm(
    N: int,
    sigma_v: float,
    sigma_w: float,
    *,
    seed: Optional[int] = None,
    x0: Optional[float] = None,
    burn_in: int = 0,
) -> NonlinearSSMSimulationResult:
    """Simulate data from a non-linear Gaussian state-space model.

    The model is defined as:
        X_1 ~ N(0, 5)  (if x0 is None)
        X_n = X_{n-1}/2 + 25*X_{n-1}/(1 + X_{n-1}^2) + 8*cos(1.2*n) + V_n
        Y_n = X_n^2 / 20 + W_n
    
    where V_n ~ N(0, σ_V^2) and W_n ~ N(0, σ_W^2) are IID Gaussian noises.

    This is a standard benchmark model used in the SMC literature to assess
    the performance of particle filters. The posterior density p(x_{1:T}|y_{1:T})
    is highly multimodal due to uncertainty about the sign of the state X_n,
    which is only observed through its square.

    Parameters
    ----------
    N : int
        Number of time steps to record (after burn-in).
    sigma_v : float
        Standard deviation of the process noise V_n (σ_V > 0).
    sigma_w : float
        Standard deviation of the measurement noise W_n (σ_W > 0).
    seed : int, optional
        Random-number-generator seed for reproducibility.
    x0 : float, optional
        Initial state value. If None, drawn from N(0, 5).
    burn_in : int, default=0
        Number of initial steps to simulate and discard before recording.

    Returns
    -------
    NonlinearSSMSimulationResult
        Dataclass containing the latent trajectory, observations, and the
        scalar noise parameters used for simulation.

    Notes
    -----
    This model is often referred to as Example 3.1 or the "standard non-linear
    state space model" in the particle filtering literature. See for example:
    - Arulampalam et al. (2002), "A Tutorial on Particle Filters"
    - Doucet & Johansen (2009), "A Tutorial on Particle Filtering and Smoothing"
    
    References
    ----------
    The model equations are:
        X_n = X_{n-1}/2 + 25*X_{n-1}/(1 + X_{n-1}^2) + 8*cos(1.2*n) + V_n  ... (14)
        Y_n = X_n^2 / 20 + W_n                                               ... (15)

    Notes
    -----
    Assumes ``sigma_v`` and ``sigma_w`` are positive finite scalars. The
    returned arrays exclude discarded burn-in samples and are ordered by time.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if sigma_v <= 0 or not np.isfinite(sigma_v):
        raise ValueError("sigma_v must be a positive finite scalar.")
    if sigma_w <= 0 or not np.isfinite(sigma_w):
        raise ValueError("sigma_w must be a positive finite scalar.")

    rng = np.random.default_rng(seed)

    X = np.empty(N, dtype=np.float64)
    Y = np.empty(N, dtype=np.float64)

    # Initial state X_1 ~ N(0, 5)
    if x0 is None:
        x = rng.normal(0.0, np.sqrt(5.0))
    else:
        x = float(x0)

    # Burn-in phase (optional)
    for n in range(-burn_in, 0):
        v = rng.normal(0.0, sigma_v)
        x = x / 2.0 + 25.0 * x / (1.0 + x**2) + 8.0 * np.cos(1.2 * n) + v

    # Main simulation
    for n in range(N):
        # Store current state
        X[n] = x
        
        # Generate observation
        w = rng.normal(0.0, sigma_w)
        y = x**2 / 20.0 + w
        Y[n] = y
        
        # State transition (n is 0-indexed, so time step is n+1)
        v = rng.normal(0.0, sigma_v)
        x = x / 2.0 + 25.0 * x / (1.0 + x**2) + 8.0 * np.cos(1.2 * (n + 1)) + v

    return NonlinearSSMSimulationResult(
        X=X,
        Y=Y,
        sigma_v=sigma_v,
        sigma_w=sigma_w,
    )
