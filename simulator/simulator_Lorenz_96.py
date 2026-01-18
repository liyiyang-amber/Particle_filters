"""
Lorenz 96 Model Simulator.

This module simulates the high-dimensional chaotic Lorenz 96 model with
ensemble forecasting and partial observations.

Model:
    dx_a/dt = (x_{a+1} - x_{a-2}) * x_{a-1} - x_a + F
    
    where a = 1, ..., nx with periodic boundary conditions.
    
    The model is integrated using the 4th-order Runge-Kutta (RK4) scheme.
    Initial conditions can be specified or use the standard pattern:
        x_a(0) = F if mod(a, 5) != 0
        x_a(0) = F + 1 if mod(a, 5) == 0
    
    Observations are sparse (e.g., every 4th variable) with Gaussian noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any
from pathlib import Path
import json

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.floating]


# Core Lorenz 96 dynamics functions
def l96_rhs(x: Array, F: float = 8.0) -> Array:
    """
    Compute the right-hand side of the Lorenz 96 equations.
    
    The equation is:
        dx_a/dt = (x_{a+1} - x_{a-2}) * x_{a-1} - x_a + F
    
    Parameters
    ----------
    x : ndarray of shape (nx,)
        Current state vector.
    F : float, default=8.0
        Forcing parameter.
    
    Returns
    -------
    ndarray of shape (nx,)
        Time derivative dx/dt.
    """
    x = np.asarray(x)
    n = x.size
    xp1 = np.roll(x, -1)  # x_{a+1}
    xm1 = np.roll(x, 1)   # x_{a-1}
    xm2 = np.roll(x, 2)   # x_{a-2}
    return (xp1 - xm2) * xm1 - x + F


def rk4_step(x: Array, dt: float, f: Callable[[Array], Array]) -> Array:
    """
    Perform a single 4th-order Runge-Kutta integration step.
    
    Parameters
    ----------
    x : ndarray of shape (nx,)
        Current state.
    dt : float
        Time step size.
    f : callable
        Function that computes dx/dt given x.
    
    Returns
    -------
    ndarray of shape (nx,)
        State after one RK4 step.
    """
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def l96_integrate(
    x0: Array,
    dt: float,
    steps: int,
    F: float = 8.0,
    q_std: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> Array:
    """
    Integrate the Lorenz 96 model forward in time using RK4.
    
    Parameters
    ----------
    x0 : ndarray of shape (nx,)
        Initial state.
    dt : float
        Time step size.
    steps : int
        Number of integration steps.
    F : float, default=8.0
        Forcing parameter.
    q_std : float, default=0.0
        Standard deviation of additive Gaussian noise at each step.
        If 0, no noise is added (deterministic integration).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    
    Returns
    -------
    ndarray of shape (steps+1, nx)
        Trajectory including the initial state at index 0.
    """
    rng = rng or np.random.default_rng()
    x = x0.copy()
    traj = np.empty((steps + 1, x0.size))
    traj[0] = x
    for t in range(1, steps + 1):
        x = rk4_step(x, dt, lambda z: l96_rhs(z, F))
        if q_std > 0:
            x = x + rng.normal(0.0, q_std, size=x.shape)
        traj[t] = x
    return traj


# Observation model
@dataclass
class ObsModel:
    """
    Linear observation model for partial state observation.
    
    Parameters
    ----------
    H_idx : ndarray of shape (ny,)
        Indices of observed state variables.
    R : ndarray of shape (ny, ny)
        Observation error covariance matrix.
    """
    H_idx: Array
    R: Array
    
    def H(self, x: Array) -> Array:
        """
        Observation operator: extract observed variables.
        
        Parameters
        ----------
        x : ndarray of shape (nx,)
            State vector.
        
        Returns
        -------
        ndarray of shape (ny,)
            Observed variables.
        """
        return x[self.H_idx]
    
    def JH(self, x: Array) -> Array:
        """
        Jacobian of the observation operator (linear case).
        
        Parameters
        ----------
        x : ndarray of shape (nx,)
            State vector (not used in linear case).
        
        Returns
        -------
        ndarray of shape (ny, nx)
            Observation Jacobian matrix.
        """
        m = self.H_idx.size
        n = x.size
        J = np.zeros((m, n))
        J[np.arange(m), self.H_idx] = 1.0
        return J


# Simulation result container
@dataclass
class Lorenz96SimulationResult:
    """
    Container for Lorenz 96 simulation data.
    
    Parameters
    ----------
    truth_traj : ndarray of shape (T+1, nx)
        Ground truth trajectory.
    ensemble_traj : ndarray of shape (Np, T+1, nx)
        Ensemble member trajectories.
    observations : ndarray of shape (n_obs_times, ny)
        Observations at observation times.
    obs_times : ndarray of shape (n_obs_times,)
        Time step indices when observations were taken.
    H_idx : ndarray of shape (ny,)
        Indices of observed state variables.
    R : ndarray of shape (ny, ny)
        Observation error covariance.
    config : dict
        Configuration parameters used for simulation.
    """
    truth_traj: Array
    ensemble_traj: Array
    observations: Array
    obs_times: Array
    H_idx: Array
    R: Array
    config: Dict[str, Any]
    
    def save(self, filepath: str, overwrite: bool = False) -> None:
        """
        Save simulation results to disk.
        
        Parameters
        ----------
        filepath : str
            Output file path. If ends with '.npz', saves in NumPy format.
            Otherwise, '.npz' is appended.
        overwrite : bool, default=False
            If False, raises FileExistsError if file exists.
        
        Raises
        ------
        FileExistsError
            If overwrite is False and file already exists.
        """
        path = Path(filepath)
        if not str(path).endswith('.npz'):
            path = path.with_suffix('.npz')
        
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")
        
        # Save arrays
        np.savez(
            path,
            truth_traj=self.truth_traj,
            ensemble_traj=self.ensemble_traj,
            observations=self.observations,
            obs_times=self.obs_times,
            H_idx=self.H_idx,
            R=self.R,
        )
        
        # Save config as JSON
        config_path = path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Saved simulation to {path}")
        print(f"Saved configuration to {config_path}")
    
    @classmethod
    def load(cls, filepath: str) -> Lorenz96SimulationResult:
        """
        Load simulation results from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the .npz file.
        
        Returns
        -------
        Lorenz96SimulationResult
            Loaded simulation data.
        """
        path = Path(filepath)
        if not str(path).endswith('.npz'):
            path = path.with_suffix('.npz')
        
        data = np.load(path)
        
        # Load config
        config_path = path.with_suffix('.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        return cls(
            truth_traj=data['truth_traj'],
            ensemble_traj=data['ensemble_traj'],
            observations=data['observations'],
            obs_times=data['obs_times'],
            H_idx=data['H_idx'],
            R=data['R'],
            config=config,
        )


# Main simulation function
def simulate_lorenz96(
    nx: int = 1000,
    F: float = 8.0,
    dt: float = 0.01,
    spinup_steps: int = 1000,
    total_steps: int = 1500,
    Np: int = 20,
    obs_interval: int = 20,
    obs_fraction: int = 4,
    obs_error_std: float = 1.0,
    perturbation_std: Optional[float] = None,
    x0: Optional[Array] = None,
    seed: Optional[int] = None,
) -> Lorenz96SimulationResult:
    """
    Simulate the Lorenz 96 model with ensemble forecasting and observations.
    
    The simulation follows this protocol:
    1. Initialize state according to standard pattern or provided x0
    2. Spin-up for spinup_steps to develop chaotic behavior
    3. Generate Np ensemble members with random perturbations
    4. Integrate truth and ensemble for total_steps
    5. Generate observations every obs_interval steps
    
    Parameters
    ----------
    nx : int, default=1000
        State dimension (number of variables).
    F : float, default=8.0
        Forcing parameter.
    dt : float, default=0.01
        Time step size for integration.
    spinup_steps : int, default=1000
        Number of spin-up steps before ensemble generation.
    total_steps : int, default=1500
        Number of steps after spin-up for data assimilation.
    Np : int, default=20
        Number of ensemble members.
    obs_interval : int, default=20
        Observation frequency (every obs_interval steps).
    obs_fraction : int, default=4
        Observe every obs_fraction-th variable (e.g., 4 means 25% observed).
    obs_error_std : float, default=1.0
        Standard deviation of observation noise.
    perturbation_std : float, optional
        Standard deviation of ensemble perturbations.
        If None, uses sqrt(2) as in standard Lorenz 96 experiments.
    x0 : ndarray of shape (nx,), optional
        Initial state. If None, uses standard pattern:
        x_a(0) = F if mod(a, 5) != 0, else F + 1.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    Lorenz96SimulationResult
        Container with truth, ensemble, observations, and configuration.
    """
    rng = np.random.default_rng(seed)
    
    # Set default perturbation std (sqrt(2) from standard experiments)
    if perturbation_std is None:
        perturbation_std = np.sqrt(2.0)
    
    # Step 1: Initialize state
    if x0 is None:
        x0 = np.full(nx, F, dtype=float)
        x0[np.arange(0, nx, 5)] = F + 1.0
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (nx,):
            raise ValueError(f"x0 must have shape ({nx},), got {x0.shape}")
    
    # Step 2: Spin-up
    x_spinup = l96_integrate(x0, dt, spinup_steps, F=F, q_std=0.0, rng=rng)
    x_at_spinup = x_spinup[-1]
    
    # Step 3: Generate ensemble members
    ensemble = np.empty((Np, nx))
    for i in range(Np):
        perturbation = rng.normal(0.0, perturbation_std, size=nx)
        ensemble[i] = x_at_spinup + perturbation
    
    # Truth is unperturbed
    x_truth_init = x_at_spinup.copy()
    
    # Step 4: Set up observation system
    H_idx = np.arange(0, nx, obs_fraction)
    ny = H_idx.size
    epsilon = obs_error_std ** 2
    R = epsilon * np.eye(ny)
    
    obs_model = ObsModel(H_idx=H_idx, R=R)
    
    # Step 5: Run forward simulation
    truth_traj = l96_integrate(x_truth_init, dt, total_steps, F=F, q_std=0.0, rng=rng)
    
    ensemble_traj = np.empty((Np, total_steps + 1, nx))
    for i in range(Np):
        ensemble_traj[i] = l96_integrate(ensemble[i], dt, total_steps, F=F, q_std=0.0, rng=rng)
    
    # Step 6: Generate observations
    obs_times = np.arange(0, total_steps + 1, obs_interval)
    n_obs_times = len(obs_times)
    observations = np.empty((n_obs_times, ny))
    
    for i, t in enumerate(obs_times):
        true_state = truth_traj[t]
        true_obs = obs_model.H(true_state)
        obs_noise = rng.normal(0.0, obs_error_std, size=ny)
        observations[i] = true_obs + obs_noise
    
    # Create config dictionary
    config = {
        'nx': int(nx),
        'F': float(F),
        'dt': float(dt),
        'spinup_steps': int(spinup_steps),
        'total_steps': int(total_steps),
        'Np': int(Np),
        'obs_interval': int(obs_interval),
        'obs_fraction': int(obs_fraction),
        'obs_error_std': float(obs_error_std),
        'perturbation_std': float(perturbation_std),
        'seed': seed,
        'ny': int(ny),
        'n_obs_times': int(n_obs_times),
    }
    
    return Lorenz96SimulationResult(
        truth_traj=truth_traj,
        ensemble_traj=ensemble_traj,
        observations=observations,
        obs_times=obs_times,
        H_idx=H_idx,
        R=R,
        config=config,
    )


# Utility functions
def compute_rmse(forecast: Array, truth: Array) -> float:
    """
    Compute root mean square error between forecast and truth.
    
    Parameters
    ----------
    forecast : ndarray
        Forecast values.
    truth : ndarray
        True values.
    
    Returns
    -------
    float
        RMSE value.
    """
    return np.sqrt(np.mean((forecast - truth) ** 2))


def compute_ensemble_spread(ensemble: Array, axis: int = 0) -> Array:
    """
    Compute ensemble spread (standard deviation).
    
    Parameters
    ----------
    ensemble : ndarray
        Ensemble array.
    axis : int, default=0
        Axis along which to compute spread.
    
    Returns
    -------
    ndarray
        Ensemble spread.
    """
    return np.std(ensemble, axis=axis)
