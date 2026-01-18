"""
Large Spatial Sensor Network (LSSN) Simulator — Linear–Gaussian Example.

This module produces synthetic state and observation sequences for a
linear–Gaussian spatial sensor network on an n×n grid (d = n^2). The
process noise has a squared–exponential (RBF) spatial covariance with a
small nugget for numerical stability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class SimConfig:
    """Configuration for the LSSN simulation.

    Parameters
    ----------
    d : int
        State dimension (must be a perfect square, d = n^2).
    alpha : float
        Scalar linear dynamics coefficient in x_t = alpha * x_{t-1} + v_t.
    alpha0 : float
        Amplitude for the squared–exponential kernel.
    alpha1 : float
        Nugget (diagonal jitter) added to the kernel.
    beta : float
        Lengthscale-squared denominator in exp(-||ri - rj||^2 / beta).
    T : int
        Time horizon (number of transitions).
    trials : int
        Number of independent replications.
    sigmas : tuple of float
        Observation noise standard deviations.
    seed : int
        Seed for the random number generator.
    """

    d: int = 64
    alpha: float = 0.9
    alpha0: float = 3.0
    alpha1: float = 0.01
    beta: float = 20.0
    T: int = 10
    trials: int = 100
    sigmas: Tuple[float, ...] = (2.0, 1.0, 0.5)
    seed: int = 123

    def __post_init__(self) -> None:
        """Validate configuration and raise informative errors if invalid."""
        n = int(round(self.d ** 0.5))
        if n * n != self.d:
            raise ValueError("d must be a perfect square (e.g., 64 = 8×8).")
        if self.T <= 0 or self.trials <= 0:
            raise ValueError("T and trials must be positive integers.")
        if any(s <= 0 for s in self.sigmas):
            raise ValueError("All observation std deviations must be positive.")
        if self.alpha1 < 0:
            raise ValueError("alpha1 (nugget) must be nonnegative.")
        if self.beta <= 0:
            raise ValueError("beta must be positive.")


def make_grid_coords(d: int) -> Array:
    """Return (d, 2) float array of 2D integer grid coordinates.

    The grid is n×n where n = sqrt(d), with row-major ordering. Indices
    range from 0 to n-1, which is translation-invariant for distance
    computation.

    Parameters
    ----------
    d : int
        State dimension, a perfect square.

    Returns
    -------
    ndarray
        Array of shape (d, 2) with [x, y] grid coordinates.
    """
    n = int(np.sqrt(d))
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    return coords


def se_kernel_cov(coords: Array, alpha0: float, beta: float, alpha1: float) -> Array:
    """Build a squared–exponential (RBF) covariance with a diagonal nugget.

    Sigma_ij = alpha0 * exp(-||ri - rj||^2 / beta) + alpha1 * 1{i=j}

    Parameters
    ----------
    coords : ndarray
        (d, 2) grid coordinates.
    alpha0 : float
        Kernel amplitude.
    beta : float
        Lengthscale-squared denominator in the exponent.
    alpha1 : float
        Diagonal nugget added for stability.

    Returns
    -------
    ndarray
        Symmetric positive definite (d, d) covariance matrix.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    K = alpha0 * np.exp(-dist2 / beta)
    # Add nugget and symmetrize numerically
    K[np.diag_indices_from(K)] += alpha1
    K = 0.5 * (K + K.T)
    return K


def cholesky_with_jitter(S: Array, max_tries: int = 5, base_jitter: float = 1e-10) -> Array:
    """Compute a Cholesky factor with progressively increasing jitter if needed.

    Parameters
    ----------
    S : ndarray
        Symmetric matrix intended to be SPD.
    max_tries : int, optional
        Maximum number of jitter attempts. Default is 5.
    base_jitter : float, optional
        Initial jitter magnitude. Default is 1e-10.

    Returns
    -------
    ndarray
        Lower-triangular Cholesky factor L such that L @ L.T ≈ S.

    Raises
    ------
    np.linalg.LinAlgError
        If factorization fails after max_tries.
    """
    jitter = 0.0
    for i in range(max_tries):
        try:
            return np.linalg.cholesky(S + jitter * np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            jitter = base_jitter * (10 ** i)
    # Final attempt or raise the last error
    return np.linalg.cholesky(S + jitter * np.eye(S.shape[0]))


def simulate_dataset(cfg: SimConfig) -> Tuple[Array, Array, Array, Array]:
    """Simulate states and observations for all trials and noise levels.

    The model is:
        x_t = alpha x_{t-1} + v_t,        v_t ~ N(0, Sigma)
        z_t = x_t + w_t,                  w_t ~ N(0, sigma_z^2 I)

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration.

    Returns
    -------
    tuple of ndarray
        Tuple ``(X, Z, coords, Sigma)`` with shapes:
            - X: (S, R, T+1, d), latent states including x0
            - Z: (S, R, T,   d), observations
            - coords: (d, 2), grid coordinates
            - Sigma: (d, d), process noise covariance
    """
    rng = np.random.default_rng(cfg.seed)
    S = len(cfg.sigmas)
    R = cfg.trials
    T = cfg.T
    d = cfg.d

    coords = make_grid_coords(d)
    Sigma = se_kernel_cov(coords, cfg.alpha0, cfg.beta, cfg.alpha1)
    L = cholesky_with_jitter(Sigma)

    X = np.zeros((S, R, T + 1, d), dtype=np.float64)
    Z = np.zeros((S, R, T, d), dtype=np.float64)

    for s_idx, sigma_z in enumerate(cfg.sigmas):
        for r in range(R):
            x = np.zeros(d, dtype=np.float64)  # x0
            X[s_idx, r, 0] = x
            for t in range(1, T + 1):
                v = L @ rng.standard_normal(d)
                x = cfg.alpha * x + v
                X[s_idx, r, t] = x

                w = sigma_z * rng.standard_normal(d)
                Z[s_idx, r, t - 1] = x + w

    return X, Z, coords, Sigma


def save_npz(
    path: str,
    X: Array,
    Z: Array,
    coords: Array,
    Sigma: Array,
    cfg: SimConfig,
) -> None:
    """Save dataset arrays and metadata to a compressed .npz file.

    Parameters
    ----------
    path: str
        Output file path for the .npz.
    X: Array
        Latent states, shape (S, R, T+1, d).
    Z: Array
        Observations, shape (S, R, T, d).
    coords: Array
        Grid coordinates, shape (d, 2).
    Sigma: Array
        Process covariance, shape (d, d).
    cfg: SimConfig
        Configuration used to generate the data.
    """
    np.savez_compressed(
        path,
        X=X,
        Z=Z,
        coords=coords,
        Sigma=Sigma,
        sigmas=np.array(cfg.sigmas, dtype=np.float64),
        alpha=np.array([cfg.alpha], dtype=np.float64),
        T=np.array([cfg.T], dtype=np.int32),
        trials=np.array([cfg.trials], dtype=np.int32),
        d=np.array([cfg.d], dtype=np.int32),
        seed=np.array([cfg.seed], dtype=np.int64),
    )


def dump_config_json(path: str, cfg: SimConfig) -> None:
    """Write the simulation configuration to a JSON file.

    Parameters
    ----------
    path: Output file path for the JSON.
    cfg: Configuration to serialize.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
