
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from numpy.linalg import cholesky, LinAlgError
from pathlib import Path

"""
Skewed-t Spatial Sensor Network Simulator.

This module simulates large spatial sensor networks with
heavy-tailed, skewed dynamics and Poisson count measurements.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable

import numpy as np
from numpy.linalg import cholesky, LinAlgError


Array = np.ndarray


# Configuration dataclasses
@dataclass
class GridConfig:
    """Configuration of the spatial grid and covariance.

    Parameters
    ----------
    d: int 
        Number of sensors (must be a perfect square).
    alpha0: float
        Spatial covariance amplitude for the exponential kernel.
    alpha1: float
        Diagonal jitter added to Sigma.
    beta: float
        Length-scale parameter for the exponential kernel.

    Notes
    -----
    Assumes ``d`` is a perfect square so the grid can be laid out on a square
    lattice.
    """
    d: int = 144
    alpha0: float = 1.0
    alpha1: float = 1e-3
    beta: float = 8.0


@dataclass
class DynConfig:
    """Configuration of the dynamics (skewed-t via N-InvGamma mixture).

    Parameters
    ----------
    alpha: float
        AR(1) coefficient for the state mean mu_k = alpha * x_{k-1}.
    nu: float
        Degrees of freedom (> 2) controlling the InvGamma mixing; lower => heavier tails.
    gamma_scale: float
        Scalar multiplier for the skew direction (applied to a unit vector if gamma_vec=None).
    gamma_vec: array, shape (d,), optional
        Optional explicit skew vector of shape (d,). If provided, overrides gamma_scale heuristic.
    clip_x: optional tuple of float
        Optional tuple (xmin, xmax) to clip latent x before exponentiation in the measurement.
    chol_jitter: float
        Added to the diagonal of Sigma when forming the Cholesky to improve numerical stability.
    seed: int | None
        RNG seed for reproducibility.

    Notes
    -----
    Assumes ``nu > 2`` for finite variance in the inverse-gamma mixture model
    typically used by this simulator.
    """
    alpha: float = 0.9
    nu: float = 8.0
    gamma_scale: float = 0.1
    gamma_vec: Optional[Array] = None
    clip_x: Optional[Tuple[float, float]] = (-10.0, 10.0)
    chol_jitter: float = 1e-8
    seed: Optional[int] = 123


@dataclass
class MeasConfig:
    """Configuration of the Poisson count measurement model.

    Parameters
    ----------
    m1: float
        Multiplicative intensity scale.
    m2: float
        Exponential sensitivity to the latent state.

    Notes
    -----
    Assumes the resulting Poisson rates remain numerically tractable after any
    clipping configured in :class:`DynConfig`.
    """
    m1: float = 1.0
    m2: float = 1.0 / 3.0


@dataclass
class SimConfig:
    """Top-level simulation configuration for multiple trials.

    Parameters
    ----------
    T: int
        Number of time steps per trial.
    n_trials: int
        Number of independent trials.
    save_lambda: bool
        Whether to save lambda_k (Poisson rates) as diagnostics.

    Notes
    -----
    Assumes trials are independent conditional on their per-trial random seed.
    """
    T: int = 10
    n_trials: int = 1
    save_lambda: bool = True

# Utility functions
def make_lattice(d: int) -> Array:
    """Return (d, 2) array of 2D coordinates for a sqrt(d) x sqrt(d) lattice.

    Sensors are laid out at integer coordinates ``0, …, s-1`` in both axes.

    Parameters
    ----------
    d : int
        Number of sensor locations, which must be a perfect square.

    Returns
    -------
    ndarray
        Array of shape ``(d, 2)`` containing integer lattice coordinates.

    Raises
    ------
    ValueError
        If ``d`` is not a perfect square.

    Notes
    -----
    Assumes a square lattice layout.
    """
    s = int(np.sqrt(d))
    if s * s != d:
        raise ValueError(f"d={d} is not a perfect square; got sqrt={s}.")
    xs, ys = np.meshgrid(np.arange(s), np.arange(s), indexing="xy")
    R = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    return R


def build_spatial_cov(R: Array, alpha0: float, alpha1: float, beta: float) -> Array:
    """Construct an exponential-squared spatial covariance matrix.

    Sigma_{ij} = alpha0 * exp(-||Ri - Rj||^2 / beta) + alpha1 * 1{i=j}.

    Parameters
    ----------
    R : ndarray
        (d, 2) sensor coordinates.
    alpha0 : float
        Spatial amplitude.
    alpha1 : float
        Diagonal nugget.
    beta : float
        Length-scale parameter.

    Returns
    -------
    ndarray
        (d, d) positive-definite covariance matrix.

    Notes
    -----
    Assumes ``R`` has shape ``(d, 2)`` and ``beta`` is positive.
    """
    d = R.shape[0]
    # Pairwise squared distances
    diffs = R[:, None, :] - R[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=-1)
    K = alpha0 * np.exp(-dist2 / beta)
    K.flat[:: d + 1] += alpha1
    return K


def cholesky_psd(Sigma: Array, jitter: float = 1e-8, max_tries: int = 5) -> Array:
    """Compute a numerically safe Cholesky factor for Sigma.

    If Sigma is not strictly PD due to numerical issues, progressively
    increase diagonal jitter until factorization succeeds.

    Parameters
    ----------
    Sigma : ndarray
        (d, d) covariance matrix.
    jitter : float, optional
        Initial jitter magnitude. Default is 1e-8.
    max_tries : int, optional
        Maximum number of jitter attempts. Default is 5.

    Returns
    -------
    ndarray
        Lower-triangular Cholesky factor L such that L @ L.T ≈ Sigma.

    Notes
    -----
    Assumes ``Sigma`` is symmetric and sufficiently close to positive definite
    for jitter-based recovery to succeed.
    """
    try_jitter = jitter
    for _ in range(max_tries):
        try:
            return cholesky(Sigma + try_jitter * np.eye(Sigma.shape[0]))
        except LinAlgError:
            try_jitter *= 10.0
    # Final attempt or raise
    return cholesky(Sigma + try_jitter * np.eye(Sigma.shape[0]))


def sample_inverse_gamma(shape: float, scale: float, rng: np.random.Generator) -> float:
    """Draw a sample W ~ InvGamma(shape, scale).

    Parameterization: pdf proportional to w^{-(shape+1)} exp(-scale / w).

    Uses the relation: if Y ~ Gamma(shape, 1/scale), then W = 1 / Y ~ InvGamma(shape, scale).

    Parameters
    ----------
    shape : float
        Shape parameter (a > 0).
    scale : float
        Scale parameter (b > 0).
    rng : np.random.Generator
        Numpy Generator.

    Returns
    -------
    float
        Sample from InvGamma(shape, scale).

    Notes
    -----
    Assumes ``shape`` and ``scale`` are positive.
    """
    # numpy.gamma uses shape=k and scale=theta; we want rate = 1/scale => theta = 1/scale
    y = rng.gamma(shape=shape, scale=1.0 / scale)
    return 1.0 / y


def prepare_gamma_vector(d: int, gamma_scale: float, gamma_vec: Optional[Array], rng: np.random.Generator) -> Array:
    """Prepare a skew vector gamma of shape (d,).

    If gamma_vec is provided, it is returned after basic checks.
    Otherwise, generate a random unit vector and scale it by gamma_scale.

    Parameters
    ----------
    d : int
        State dimension.
    gamma_scale : float
        Scale applied to a generated unit vector when ``gamma_vec`` is not
        provided.
    gamma_vec : ndarray, optional
        Explicit skew vector.
    rng : np.random.Generator
        Random generator used when a new vector must be sampled.

    Returns
    -------
    ndarray
        Skew vector of shape ``(d,)``.

    Notes
    -----
    Assumes any supplied ``gamma_vec`` already matches the intended skew
    direction and scale.
    """
    if gamma_vec is not None:
        gamma_vec = np.asarray(gamma_vec).reshape(-1)
        if gamma_vec.shape[0] != d:
            raise ValueError(f"gamma_vec shape {gamma_vec.shape} incompatible with d={d}")
        return gamma_vec
    # Random direction
    v = rng.normal(size=d)
    v_norm = np.linalg.norm(v) + 1e-12
    return gamma_scale * (v / v_norm)


# Core simulation routines
def simulate_trial(
    grid_cfg: GridConfig,
    dyn_cfg: DynConfig,
    meas_cfg: MeasConfig,
    sim_cfg: SimConfig,
) -> Dict[str, Any]:
    """Simulate a single trial for T time steps.

    The latent dynamics follow a skewed-t AR(1) using a Normal–Inverse-Gamma mixture.
    Measurements are Poisson with log-link to the latent state.

    Parameters
    ----------
    grid_cfg : GridConfig
        Spatial-grid and covariance configuration.
    dyn_cfg : DynConfig
        Latent dynamics configuration.
    meas_cfg : MeasConfig
        Measurement-model configuration.
    sim_cfg : SimConfig
        Trial configuration.

    Returns
    -------
    dict
        Dictionary with keys ``'X'``, ``'Z'``, optional ``'Lambda'``,
        ``'Sigma'``, ``'L'``, ``'R'``, ``'gamma'``, and ``'meta'``.

    Notes
    -----
    Assumes a single independent trial seeded by ``dyn_cfg.seed``.
    """
    # RNG
    rng = np.random.default_rng(dyn_cfg.seed)

    # Grid & covariance
    R = make_lattice(grid_cfg.d)
    Sigma = build_spatial_cov(R, grid_cfg.alpha0, grid_cfg.alpha1, grid_cfg.beta)
    L = cholesky_psd(Sigma, jitter=dyn_cfg.chol_jitter)

    d = grid_cfg.d
    T = sim_cfg.T

    # Skew vector
    gamma = prepare_gamma_vector(d, dyn_cfg.gamma_scale, dyn_cfg.gamma_vec, rng)

    # Initialize arrays
    X = np.zeros((T, d), dtype=float)
    Z = np.zeros((T, d), dtype=int)
    Lambda = np.zeros((T, d), dtype=float) if sim_cfg.save_lambda else None

    # Initial state x_0 = 0
    x = np.zeros(d, dtype=float)

    # Mixture parameters for inverse-gamma
    shape = dyn_cfg.nu / 2.0  # a
    scale = dyn_cfg.nu / 2.0  # b

    for k in range(T):
        # Draw mixture scalar W_k ~ InvGamma(nu/2, nu/2)
        W = sample_inverse_gamma(shape=shape, scale=scale, rng=rng)

        # Dynamics: x_k = alpha * x_{k-1} + W * gamma + sqrt(W) * L z
        z = rng.normal(size=d)
        mu = dyn_cfg.alpha * x
        x = mu + W * gamma + np.sqrt(W) * (L @ z)

        # Poisson rates 
        x_eff = x
        if dyn_cfg.clip_x is not None:
            xmin, xmax = dyn_cfg.clip_x
            x_eff = np.clip(x_eff, xmin, xmax)
        lam = meas_cfg.m1 * np.exp(meas_cfg.m2 * x_eff)

        # Counts
        z_counts = rng.poisson(lam)

        # Save
        X[k, :] = x
        Z[k, :] = z_counts
        if Lambda is not None:
            Lambda[k, :] = lam

    meta = {
        "grid_cfg": vars(grid_cfg),
        "dyn_cfg": {
            **{k: v for k, v in vars(dyn_cfg).items() if k != "gamma_vec"},
            "gamma_vec": "provided" if dyn_cfg.gamma_vec is not None else None,
        },
        "meas_cfg": vars(meas_cfg),
        "sim_cfg": vars(sim_cfg),
    }

    out = {"X": X, "Z": Z, "Sigma": Sigma, "L": L, "R": R, "gamma": gamma, "meta": meta}
    if Lambda is not None:
        out["Lambda"] = Lambda
    return out


def simulate_many(
    grid_cfg: GridConfig,
    dyn_cfg: DynConfig,
    meas_cfg: MeasConfig,
    sim_cfg: SimConfig,
) -> Dict[str, Any]:
    """Simulate multiple independent trials and stack outputs.

    Parameters
    ----------
    grid_cfg : GridConfig
        Spatial-grid and covariance configuration.
    dyn_cfg : DynConfig
        Dynamics configuration used as the trial template.
    meas_cfg : MeasConfig
        Measurement-model configuration.
    sim_cfg : SimConfig
        Multi-trial configuration.

    Returns
    -------
    dict
        Dictionary containing stacked latent states and observations, optional
        Poisson rates, and per-trial metadata lists.

    Notes
    -----
    Assumes trials are independent and are made reproducible by offsetting the
    base random seed when one is provided.
    """
    n = sim_cfg.n_trials
    T, d = sim_cfg.T, grid_cfg.d

    X_all = np.zeros((n, T, d), dtype=float)
    Z_all = np.zeros((n, T, d), dtype=int)
    Lambda_all = np.zeros((n, T, d), dtype=float) if sim_cfg.save_lambda else None

    Sigmas, Ls, gammas, metas = [], [], [], []

    # To vary trials but keep reproducibility, offset the seed per trial
    base_seed = dyn_cfg.seed
    for i in range(n):
        trial_dyn = DynConfig(**{**vars(dyn_cfg), "seed": None if base_seed is None else base_seed + i})
        out = simulate_trial(grid_cfg, trial_dyn, meas_cfg, SimConfig(T=T, n_trials=1, save_lambda=sim_cfg.save_lambda))

        X_all[i] = out["X"]
        Z_all[i] = out["Z"]
        if Lambda_all is not None:
            Lambda_all[i] = out["Lambda"]
        Sigmas.append(out["Sigma"])
        Ls.append(out["L"])
        gammas.append(out["gamma"])
        metas.append(out["meta"])

    result = {"X": X_all, "Z": Z_all, "Sigma_list": Sigmas, "L_list": Ls, "gamma_list": gammas, "meta_list": metas}
    if Lambda_all is not None:
        result["Lambda"] = Lambda_all
    return result


# Convenience save/load
def save_npz(path: str, data: Dict[str, Any]) -> None:
    """Save dictionary of numpy arrays / lists to a .npz file.

    Lists of arrays (e.g., ``Sigma_list``) are saved as an object array.

    Parameters
    ----------
    path : str
        Target ``.npz`` path.
    data : dict
        Dictionary produced by :func:`simulate_trial` or :func:`simulate_many`.

    Returns
    -------
    None
        Writes the provided data to disk.
    """
    to_save = {}
    for k, v in data.items():
        if isinstance(v, list):
            to_save[k] = np.array(v, dtype=object)
        else:
            to_save[k] = v
    np.savez_compressed(path, **to_save)


def load_npz(path: str) -> Dict[str, Any]:
    """Load a dataset saved by :func:`save_npz`.

    Parameters
    ----------
    path : str
        Path to a previously saved ``.npz`` archive.

    Returns
    -------
    dict
        Dictionary containing arrays and reconstructed object-list entries.
    """
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k].tolist() if f[k].dtype == object else f[k] for k in f.files}
    

