"""
Particle Filter (PF) for nonlinear state-space models.

Implements a standard Sequential Importance Resampling (SIR) Particle Filter
for additive-noise nonlinear systems, with optional systematic or multinomial
resampling. 

Model:
    x_k = g(x_{k-1}, u_{k-1}) + w_{k-1},   w ~ N(0, Q)
    z_k = h(x_k) + v_k,                   v ~ N(0, R)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

Array = np.ndarray
GFn = Callable[[Array, Optional[Array]], Array]
HFn = Callable[[Array], Array]



# State container
@dataclass
class PFState:
    """Container for Particle Filter posterior state.

    Parameters
    ----------
    particles : np.ndarray
        Particle states (Np, nx).
    weights : np.ndarray
        Normalized weights (Np,).
    mean : np.ndarray
        Weighted posterior mean (nx,).
    cov : np.ndarray
        Weighted posterior covariance (nx, nx).
    t : int
        Discrete time index.
    """

    particles: Array
    weights: Array
    mean: Array
    cov: Array
    t: int


# Particle Filter implementation
class ParticleFilter:
    """SIR Particle Filter with resampling and regularization.

    Parameters
    ----------
    g : callable
        State transition function g(x, u) → (nx,).
    h : callable
        Measurement function h(x) → (nz,).
    Q : ndarray
        Process noise covariance (nx, nx).
    R : ndarray
        Measurement noise covariance (nz, nz).
    Np : int, optional
        Number of particles. Default is 1000.
    resample_thresh : float, optional
        Fraction of Np triggering resample when Neff drops below. Default is 0.5.
    resample_method : str, optional
        Resampling method: 'systematic' or 'multinomial'. Default is 'systematic'.
    regularize_after_resample : bool, optional
        If True, add small jitter after resampling to mitigate particle impoverishment.
        Default is False.
    rng : np.random.Generator, optional
        Numpy random Generator. Default is None (uses default generator).
    """

    def __init__(
        self,
        g: GFn,
        h: HFn,
        Q: Array,
        R: Array,
        *,
        Np: int = 1000,
        resample_thresh: float = 0.5,
        resample_method: str = "systematic",
        regularize_after_resample: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.g = g
        self.h = h
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.Np = int(Np)
        self.resample_thresh = float(resample_thresh)
        self.resample_method = resample_method
        self.regularize_after_resample = regularize_after_resample
        self.rng = np.random.default_rng() if rng is None else rng

        self.nx = self.Q.shape[0]
        self.nz = self.R.shape[0]
        self.state: Optional[PFState] = None

        # Precompute Cholesky of R
        self.LR = np.linalg.cholesky(self.R + 1e-12 * np.eye(self.nz))

    # Initialization and resampling
    def initialize(self, mean: Array, cov: Array) -> PFState:
        """Initialize particles from a Gaussian N(mean, cov).

        Parameters
        ----------
        mean : ndarray
            Mean vector of shape (nx,).
        cov : ndarray
            Covariance matrix of shape (nx, nx).

        Returns
        -------
        PFState
            Filter posterior state with initialized particles.
        """
        mean = np.asarray(mean, float)
        cov = np.asarray(cov, float)
        Lc = np.linalg.cholesky(cov + 1e-10 * np.eye(len(mean)))
        particles = self.rng.standard_normal((self.Np, len(mean))) @ Lc.T + mean
        weights = np.ones(self.Np) / self.Np
        cov = np.atleast_2d(cov)
        self.state = PFState(particles, weights, mean, cov, 0)
        return self.state

    def effective_sample_size(self) -> float:
        """Return current effective sample size Neff.

        Returns
        -------
        float
            Effective sample size.
        """
        assert self.state is not None, "Filter not initialized."
        w = self.state.weights
        return 1.0 / np.sum(w ** 2)

    def _systematic_resample(self, weights: Array) -> Array:
        """Perform systematic resampling.

        Parameters
        ----------
        weights : ndarray
            Particle weights of shape (Np,).

        Returns
        -------
        ndarray
            Resampled particle indices.
        """
        N = len(weights)
        positions = (self.rng.random() + np.arange(N)) / N
        indexes = np.zeros(N, dtype=int)
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0  # avoid round-off error
        i, j = 0, 0
        while i < N:
            if positions[i] < cumsum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def _multinomial_resample(self, weights: Array) -> Array:
        """Perform multinomial resampling.

        Parameters
        ----------
        weights : ndarray
            Particle weights of shape (Np,).

        Returns
        -------
        ndarray
            Resampled particle indices.
        """
        return self.rng.choice(len(weights), size=len(weights), p=weights)

    def _resample(self, particles: Array, weights: Array) -> Tuple[Array, Array]:
        """Resample particles when degeneracy threshold reached.

        Parameters
        ----------
        particles : ndarray
            Current particle states of shape (Np, nx).
        weights : ndarray
            Current particle weights of shape (Np,).

        Returns
        -------
        tuple of ndarray
            Resampled particles and normalized weights.
        """
        Neff = 1.0 / np.sum(weights ** 2)
        if Neff < self.resample_thresh * self.Np:
            if self.resample_method == "systematic":
                idx = self._systematic_resample(weights)
            else:
                idx = self._multinomial_resample(weights)
            particles = particles[idx]
            weights = np.ones_like(weights) / len(weights)

            if self.regularize_after_resample:
                try:
                    Lq = np.linalg.cholesky(self.Q)
                except np.linalg.LinAlgError:
                    Lq = np.linalg.cholesky(self.Q + 1e-12 * np.eye(self.nx))
                jitter = self.rng.standard_normal(particles.shape) @ (0.001 * Lq.T)
                particles += jitter

        return particles, weights

    # Core filtering steps
    def predict(self, u: Optional[Array] = None) -> None:
        """Propagate particles through the transition model.

        Parameters
        ----------
        u : ndarray, optional
            Control input. Default is None.
        """
        assert self.state is not None, "Filter not initialized."
        try:
            Lq = np.linalg.cholesky(self.Q)
        except np.linalg.LinAlgError:
            Lq = np.linalg.cholesky(self.Q + 1e-10 * np.eye(self.nx))
        noise = self.rng.standard_normal((self.Np, self.nx)) @ Lq.T
        self.state.particles = np.array([self.g(x, u) for x in self.state.particles]) + noise

    def update(self, z: Array) -> PFState:
        """Update particle weights given measurement z.

        Parameters
        ----------
        z : ndarray
            Measurement of shape (nz,).

        Returns
        -------
        PFState
            Updated filter state.
        """
        assert self.state is not None, "Filter not initialized."
        z = np.asarray(z, float)
        particles = self.state.particles
        weights = self.state.weights

        z_pred = np.array([self.h(x) for x in particles])
        diffs = (z - z_pred).T  # shape (nz, Np)
        y = np.linalg.solve(self.LR, diffs)
        quad = np.sum(y * y, axis=0)
        logw = np.log(weights + 1e-300) - 0.5 * quad
        m = np.max(logw)
        w = np.exp(logw - (m + np.log(np.sum(np.exp(logw - m)))))  # stable normalize

        particles, w = self._resample(particles, w)
        mean = np.average(particles, axis=0, weights=w)
        cov = np.atleast_2d(np.cov(particles.T, aweights=w, bias=True))
        self.state = PFState(particles, w, mean, cov, self.state.t + 1)
        return self.state

    def step(self, z: Array, u: Optional[Array] = None) -> PFState:
        """Run one PF step (predict then update).

        Parameters
        ----------
        z : ndarray
            Measurement of shape (nz,).
        u : ndarray, optional
            Control input. Default is None.

        Returns
        -------
        PFState
            Updated filter state.
        """
        self.predict(u)
        return self.update(z)