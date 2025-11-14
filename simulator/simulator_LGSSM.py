from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class LGSSMSimulationResult:
    """Container for simulated linear Gaussian state-space model data."""
    X: NDArray[np.float64]
    """Array of shape (N, nx) containing latent states x_1, ..., x_N."""
    Y: NDArray[np.float64]
    """Array of shape (N, ny) containing observations y_1, ..., y_N."""
    A: NDArray[np.float64]
    """State transition matrix (nx, nx)."""
    B: NDArray[np.float64]
    """Process-noise input matrix (nx, nv)."""
    C: NDArray[np.float64]
    """Observation matrix (ny, nx)."""
    D: NDArray[np.float64]
    """Measurement-noise input matrix (ny, nw)."""
    
    def to_file(
        self,
        path: str,
        format: "npz",
        overwrite: bool = False,
    ) -> None:
        """
        Save the simulated data to a file for Kalman filtering.

        Parameters
        ----------
        path : str
            Destination file path (without extension for .npz format).
        format : 'npz'
            Output format. 
        overwrite : bool, default=False
            If False, raises an error when the file already exists.

        Raises
        ------
        FileExistsError
            If overwrite is False and the target file already exists.
        """
        
        target = path if path.endswith(".npz") else f"{path}.npz"
        if os.path.exists(target) and not overwrite:
            raise FileExistsError(f"File already exists: {target}")
        np.savez(target, X=self.X, Y=self.Y, A=self.A, B=self.B, C=self.C, D=self.D)

    

def simulate_lgssm(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    C: NDArray[np.float64],
    D: NDArray[np.float64],
    Sigma: NDArray[np.float64],
    N: int,
    *,
    seed: Optional[int] = None,
    burn_in: int = 0,
) -> LGSSMSimulationResult:
    """
    Simulate data from a linear Gaussian state-space model.

    The model is defined as:
        x_1 ~ N(0, \Sigma)
        x_{n+1} = A x_n + B v_n,   v_n ~ N(0, I)
        y_n   = C x_n + D w_n,   w_n ~ N(0, I)

    Parameters
    ----------
    A : ndarray of shape (nx, nx)
        State transition matrix.
    B : ndarray of shape (nx, nv)
        Process-noise input matrix.
    C : ndarray of shape (ny, nx)
        Observation matrix.
    D : ndarray of shape (ny, nw)
        Measurement-noise input matrix.
    Sigma : ndarray of shape (nx, nx)
        Covariance of the initial state x_1.
    N : int
        Number of time steps to record (after burn-in).
    seed : int, optional
        Random-number-generator seed for reproducibility.
    burn_in : int, default=0
        Number of initial steps to simulate and discard before recording.

    Returns
    -------
    LGSSMSimulationResult
        Dataclass with two fields:
            - X : ndarray of shape (N, nx)
                Simulated latent states x_1, ..., x_N.
            - Y : ndarray of shape (N, ny)
                Simulated observations y_1, ..., y_N.

    Notes
    -----
    The function follows the standard linear-Gaussian model.  The initial state is drawn from N(0, Σ), and all noises are
    independent standard normals.
    """
    rng = np.random.default_rng(seed)

    nx = A.shape[0]
    nv = B.shape[1]
    ny = C.shape[0]
    nw = D.shape[1]

    X = np.empty((N, nx))
    Y = np.empty((N, ny))

    # Initial state x_1 ~ N(0, Σ)
    x = rng.multivariate_normal(mean=np.zeros(nx), cov=Sigma)

    # Burn-in phase (optional)
    for _ in range(burn_in):
        v = rng.standard_normal(nv)
        x = A @ x + B @ v

    # Main simulation
    for n in range(N):
        w = rng.standard_normal(nw)
        y = C @ x + D @ w
        Y[n] = y
        X[n] = x
        v = rng.standard_normal(nv)
        x = A @ x + B @ v

    return LGSSMSimulationResult(X=X, Y=Y, A=A, B=B, C=C, D=D)