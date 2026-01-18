"""
Multi-Target Acoustic Tracking Simulator.

This module provides a vectorized simulator for generating
multi-target trajectories under constant-velocity (CV) dynamics and their
acoustic sensor measurements. 

Outputs
-------
- X: (T, C, 4) target states [x, y, vx, vy].
- P: (T, C, 2) target positions [x, y].
- S: (S, 2) sensor positions.
- Z: (T, S) sensor measurements.
- meta: array of scenario and dynamics scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray


# Configuration dataclasses
@dataclass(frozen=True)
class DynamicsConfig:
    """Configuration for 2D constant-velocity dynamics.

    Parameters
    ----------
    dt : float
        Time step (seconds).
    """

    dt: float = 1.0


@dataclass(frozen=True)
class ScenarioConfig:
    """High-level scenario configuration for multi-target acoustic simulation.

    Parameters
    ----------
    n_targets : int
        Number of targets (article uses 4).
    n_steps : int
        Number of time steps to simulate (including k=0).
    area_xy : tuple of float
        Tuple (width, height) in meters for a rectangular area.
    sensor_grid_shape : tuple of int
        (rows, cols) for a rectangular sensor grid.
    psi : float
        Source amplitude constant in the acoustic model.
    d0 : float
        Small positive offset to avoid singularity in amplitude model.
    seed : int
        Random generator seed for reproducibility.
    use_article_init : bool
        If True and n_targets == 4, use the article's X0 states.
    """

    n_targets: int = 4
    n_steps: int = 100
    area_xy: Tuple[float, float] = (40.0, 40.0)
    sensor_grid_shape: Tuple[int, int] = (5, 5)
    psi: float = 10.0
    d0: float = 0.1
    seed: int = 7
    use_article_init: bool = True


# Core linear algebra for CV model
def build_cv_transition(dt: float) -> Array:
    """Create the constant-velocity state transition matrix `F`.

    The state is [x, y, vx, vy].

    Parameters
    ----------
    dt : float
        Time step in seconds.

    Returns
    -------
    ndarray
        A (4, 4) numpy array representing the state transition.
    """
    F = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return F


def article_process_noise_cov() -> Array:
    """Return the article's process noise covariance `V` for the CV model.

    The covariance is fixed and does not depend on dt. It is given by:
        V = (1/20) * [[1/3, 0,   0.5, 0  ],
                      [0,   1/3, 0,   0.5],
                      [0.5, 0,   1,   0  ],
                      [0,   0.5, 0,   1  ]]

    Returns
    -------
    ndarray
        A (4, 4) numpy array.
    """
    V = (1.0 / 20.0) * np.array(
        [
            [1.0 / 3.0, 0.0, 0.5, 0.0],
            [0.0, 1.0 / 3.0, 0.0, 0.5],
            [0.5, 0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 1.0],
        ],
        dtype=float,
    )
    return V


def article_initial_states(n_targets: int) -> Array:
    """Return the article's deterministic initial states when n_targets == 4.

    The states are:
        [12,  6,   0.001,  0.001],
        [32,  32, -0.001, -0.005],
        [20,  13, -0.1,    0.01 ],
        [15,  35,  0.002,  0.002]

    Parameters
    ----------
    n_targets : int
        Number of targets; must be 4 to use this initialization.

    Returns
    -------
    ndarray
        A (4, 4) array of initial states.

    Raises
    ------
    ValueError
        If n_targets is not 4.
    """
    if n_targets != 4:
        raise ValueError("Article initial states are defined for n_targets == 4.")
    X0 = np.array(
        [
            [12.0, 6.0, 0.001, 0.001],
            [32.0, 32.0, -0.001, -0.005],
            [20.0, 13.0, -0.1, 0.01],
            [15.0, 35.0, 0.002, 0.002],
        ],
        dtype=float,
    )
    return X0


# Scenario utilities
def make_sensor_grid(area_xy: Tuple[float, float], grid_shape: Tuple[int, int]) -> Array:
    """Create a rectangular grid of sensor locations over the area.

    Sensors lie on grid intersections (inclusive of boundaries).

    Parameters
    ----------
    area_xy: Tuple 
        (width, height) of the rectangle in meters.
    grid_shape: Tuple 
        (rows, cols) for the grid.

    Returns
    -------
    ndarray
        sensors A (S, 2) array of sensor xy-locations.
    """
    width, height = area_xy
    n_r, n_c = grid_shape
    xs = np.linspace(0.0, width, n_c)
    ys = np.linspace(0.0, height, n_r)
    XX, YY = np.meshgrid(xs, ys)
    sensors = np.column_stack([XX.ravel(), YY.ravel()])
    return sensors


# Simulation routines
def simulate_cv_targets(
    n_steps: int,
    n_targets: int,
    area_xy: Tuple[float, float],
    dyn_cfg: DynamicsConfig,
    rng: np.random.Generator,
    use_article_init: bool = True,
    init_vel_std: float = 0.5,
    enforce_boundaries: bool = True,
) -> Array:
    """Simulate constant-velocity target trajectories with article-exact noise.

    Parameters
    ----------
        n_steps: Number of time steps to simulate (including step 0).
        n_targets: Number of targets.
        area_xy: Tuple (width, height) defining the tracking area bounds.
        dyn_cfg: Dynamics configuration (uses dt).
        rng: NumPy random generator.
        use_article_init: If True and n_targets == 4, use article initial states.
            Otherwise, use random positions near the center and small random velocities.
        init_vel_std: Standard deviation for random initial velocities when not
            using article initial states.
        enforce_boundaries: If True, reflect targets at boundaries to keep them
            within the area. Default True.

    Returns
    -------
        X: A (n_steps, n_targets, 4) array of states [x, y, vx, vy].
    """
    F = build_cv_transition(dyn_cfg.dt)
    V = article_process_noise_cov()

    X = np.zeros((n_steps, n_targets, 4), dtype=float)

    if use_article_init and n_targets == 4:
        X[0, :, :] = article_initial_states(n_targets)
    else:
        width, height = area_xy
        x0 = rng.uniform(0.25 * width, 0.75 * width, size=(n_targets, 1))
        y0 = rng.uniform(0.25 * height, 0.75 * height, size=(n_targets, 1))
        vx0 = rng.normal(0.0, init_vel_std, size=(n_targets, 1))
        vy0 = rng.normal(0.0, init_vel_std, size=(n_targets, 1))
        X[0, :, :] = np.hstack([x0, y0, vx0, vy0])

    # Cholesky factor for efficient sampling: w_k ~ N(0, V).
    L = np.linalg.cholesky(V + 1e-12 * np.eye(4))

    width, height = area_xy
    epsilon = 1e-6  # Small buffer to ensure strict inequality
    
    for k in range(1, n_steps):
        w = (L @ rng.normal(size=(4, n_targets))).T  # (n_targets, 4)
        X[k, :, :] = (X[k - 1, :, :] @ F.T) + w
        
        # Enforce boundary constraints with reflection (strict: 0 < pos < width/height)
        if enforce_boundaries:
            for c in range(n_targets):
                # Check X boundary (keep strictly within [0, width))
                if X[k, c, 0] <= 0:
                    X[k, c, 0] = -X[k, c, 0] + epsilon  # Reflect position
                    X[k, c, 2] = -X[k, c, 2]  # Reverse velocity
                elif X[k, c, 0] >= width:
                    X[k, c, 0] = 2 * width - X[k, c, 0] - epsilon  # Reflect position
                    X[k, c, 2] = -X[k, c, 2]  # Reverse velocity
                
                # Check Y boundary (keep strictly within [0, height))
                if X[k, c, 1] <= 0:
                    X[k, c, 1] = -X[k, c, 1] + epsilon  # Reflect position
                    X[k, c, 3] = -X[k, c, 3]  # Reverse velocity
                elif X[k, c, 1] >= height:
                    X[k, c, 1] = 2 * height - X[k, c, 1] - epsilon  # Reflect position
                    X[k, c, 3] = -X[k, c, 3]  # Reverse velocity

    return X


def acoustic_measurement_model(
    positions: Array,
    sensors: Array,
    psi: float,
    d0: float,
    # rng: Optional[np.random.Generator] = None,
    # noise_std: float = 0.1,
) -> Array:
    """Compute additive acoustic amplitudes at sensors from multiple targets.

    Parameters
    ----------
        positions: Array (T, C, 2) of target xy positions.
        sensors: Array (S, 2) of sensor xy locations.
        psi: Source amplitude constant.
        d0: Small positive offset to avoid division by zero.
        # rng: Optional RNG for additive Gaussian noise.
        # noise_std: Sensor noise standard deviation.

    Returns
    -------
        Z: Array (T, S) of measurements.
    """
    T, C = positions.shape[:2]
    S = sensors.shape[0]

    pos_exp = positions[:, :, None, :]          # (T, C, 1, 2)
    sen_exp = sensors[None, None, :, :]         # (1, 1, S, 2)
    d2 = np.sum((pos_exp - sen_exp) ** 2, axis=-1)  # (T, C, S)

    zbar = psi / (d2 + d0)                      # (T, C, S)
    zbar_sum = np.sum(zbar, axis=1)             # (T, S)

    # if rng is None:
    #     return zbar_sum
    # noise = rng.normal(0.0, noise_std, size=zbar_sum.shape)
    return zbar_sum 


def simulate_acoustic_dataset(cfg: ScenarioConfig, dyn_cfg: DynamicsConfig) -> Dict[str, Array]:
    """Simulate a complete multi-target acoustic dataset.

    Parameters
    ----------
    cfg: Scenario configuration.
    dyn_cfg: Dynamics configuration.

    Returns
    -------
    A dictionary with keys:
        - "X": (T, C, 4) states [x, y, vx, vy].
        - "P": (T, C, 2) positions [x, y].
        - "S": (S, 2) sensor locations.
        - "Z": (T, S) noiseless sensor measurements.
        - "meta": small array with [W, H, psi, d0, dt].
    """
    rng = np.random.default_rng(cfg.seed)
    sensors = make_sensor_grid(cfg.area_xy, cfg.sensor_grid_shape)
    X = simulate_cv_targets(
        cfg.n_steps,
        cfg.n_targets,
        cfg.area_xy,
        dyn_cfg,
        rng,
        use_article_init=cfg.use_article_init,
    )
    P = X[..., :2]
    Z = acoustic_measurement_model(P, sensors, psi=cfg.psi, d0=cfg.d0)

    meta = np.array(
        [cfg.area_xy[0], cfg.area_xy[1], cfg.psi, cfg.d0, dyn_cfg.dt],
        dtype=float,
    )
    return {"X": X, "P": P, "S": sensors, "Z": Z, "meta": meta}