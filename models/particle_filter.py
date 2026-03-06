"""
Particle Filter (PF) for nonlinear state-space models.

Implements a standard Sequential Importance Resampling (SIR) Particle Filter
for additive-noise nonlinear systems with systematic or multinomial
resampling, optional post-resample regularisation, and an optional
particle-impoverishment jitter.

The additive-noise SSM is::

    x_k  = g(x_{k-1}, u_{k-1}) + w_{k-1},   w ~ N(0, Q)
    z_k  = h(x_k) + v_k,                     v ~ N(0, R)

Classes
-------
PFState        – Particle-filter posterior container.
ParticleFilter – SIR PF implementing :class:`~base_ssm.BaseFilter`.

Shared utilities from :mod:`base_ssm`
--------------------------------------
:func:`~base_ssm.systematic_resample`, :func:`~base_ssm.multinomial_resample`,
:func:`~base_ssm.effective_sample_size`, and
:func:`~base_ssm.weighted_mean_cov` are imported and reused directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:
    from models.base_ssm import (
        BaseFilter,
        BaseFilterState,
        ParticleFilterState,
        systematic_resample as _systematic_resample,
        multinomial_resample as _multinomial_resample,
        effective_sample_size as _ess,
        weighted_mean_cov,
    )
except ModuleNotFoundError:
    from base_ssm import (
        BaseFilter,
        BaseFilterState,
        ParticleFilterState,
        systematic_resample as _systematic_resample,
        multinomial_resample as _multinomial_resample,
        effective_sample_size as _ess,
        weighted_mean_cov,
    )

Array = np.ndarray
GFn = Callable[[Array, Optional[Array]], Array]
HFn = Callable[[Array], Array]



# Keep PFState as an alias to ParticleFilterState for backward compatibility
# with notebooks and tests that import it directly from this module.
PFState = ParticleFilterState


# Particle Filter implementation
class ParticleFilter(BaseFilter):
    """SIR Particle Filter with resampling and optional regularisation.

    Derives from :class:`~base_ssm.BaseFilter`, providing a consistent
    ``initialize`` / ``predict`` / ``update`` / ``step`` / ``run`` interface.

    The additive-noise SSM is::

        x_k  = g(x_{k-1}, u_{k-1}) + w_{k-1},   w ~ N(0, Q)
        z_k  = h(x_k) + v_k,                     v ~ N(0, R)

    Parameters
    ----------
    g : callable
        State transition function g(x, u) → (nx,), evaluated per particle.
    h : callable
        Measurement function h(x) → (nz,), evaluated per particle.
    Q : ndarray, shape (nx, nx)
        Process-noise covariance.
    R : ndarray, shape (nz, nz)
        Measurement-noise covariance.
    Np : int, optional
        Number of particles. Default is 1000.
    resample_thresh : float, optional
        Resample when ESS < resample_thresh * Np. Default is 0.5.
    resample_method : str, optional
        Resampling method: ``'systematic'`` (default) or ``'multinomial'``.
    regularize_after_resample : bool, optional
        If ``True``, add small Gaussian jitter (scaled by Q) after resampling
        to mitigate particle impoverishment. Default is ``False``.
    rng : np.random.Generator, optional
        NumPy random generator for reproducibility. Default creates a new
        default generator.

    Notes
    -----
    Assumes ``g`` maps a single state (and optional control) to a state vector
    of fixed dimension, and ``h`` maps a state vector to an observation vector
    compatible with ``R``. The implementation uses Gaussian observation noise
    and additive Gaussian process noise.
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

        Notes
        -----
        Assumes ``cov`` is positive semidefinite up to a small diagonal jitter
        used for Cholesky factorisation.
        """
        mean = np.asarray(mean, float)
        cov = np.asarray(cov, float)
        Lc = np.linalg.cholesky(cov + 1e-10 * np.eye(len(mean)))
        particles = self.rng.standard_normal((self.Np, len(mean))) @ Lc.T + mean
        weights = np.ones(self.Np) / self.Np
        cov = np.atleast_2d(cov)
        self.state = PFState(particles=particles, weights=weights, mean=mean, cov=cov, t=0)
        return self.state

    def effective_sample_size(self) -> float:
        """Return the current Effective Sample Size (ESS).

        Delegates to :func:`~base_ssm.effective_sample_size`.

        Returns
        -------
        float
            ESS = 1 / Σ w_i^2 in the range [1, Np].

        Raises
        ------
        AssertionError
            If :meth:`initialize` has not been called yet.

        Notes
        -----
        Assumes stored particle weights are normalised.
        """
        assert self.state is not None, "Filter not initialized."
        return _ess(self.state.weights)

    def _systematic_resample(self, weights: Array) -> Array:
        """Perform systematic resampling.

        Delegates to :func:`~base_ssm.systematic_resample`.

        Parameters
        ----------
        weights : ndarray, shape (Np,)
            Normalized particle weights (sum to 1).

        Returns
        -------
        ndarray of int, shape (Np,)
            Resampled ancestor indices.
        """
        return _systematic_resample(weights, self.rng)

    def _multinomial_resample(self, weights: Array) -> Array:
        """Perform multinomial resampling.

        Delegates to :func:`~base_ssm.multinomial_resample`.

        Parameters
        ----------
        weights : ndarray, shape (Np,)
            Normalized particle weights (sum to 1).

        Returns
        -------
        ndarray of int, shape (Np,)
            Resampled ancestor indices.
        """
        return _multinomial_resample(weights, self.rng)

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
        Neff = _ess(weights)
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
    def predict(self, u: Optional[Array] = None) -> PFState:
        """Propagate particles through the process model (time-update step).

        Mutates ``self.state`` in-place and returns it.

        Parameters
        ----------
        u : ndarray, optional
            Control input u_{k-1}.  ``None`` if no input.

        Returns
        -------
        PFState
            Current filter state after particle propagation.

        Raises
        ------
        AssertionError
            If :meth:`initialize` has not been called yet.
        """
        assert self.state is not None, "Filter not initialized."
        try:
            Lq = np.linalg.cholesky(self.Q)
        except np.linalg.LinAlgError:
            Lq = np.linalg.cholesky(self.Q + 1e-10 * np.eye(self.nx))
        noise = self.rng.standard_normal((self.Np, self.nx)) @ Lq.T
        self.state.particles = np.array([self.g(x, u) for x in self.state.particles]) + noise
        return self.state

    def update(self, z: Optional[Array] = None) -> PFState:
        """Update particle weights given a new measurement (measurement-update step).

        Mutates ``self.state`` in-place and returns it.

        Parameters
        ----------
        z : ndarray, shape (nz,)
            Measurement at the current time step.

        Returns
        -------
        PFState
            Updated filter state.

        Raises
        ------
        AssertionError
            If :meth:`initialize` has not been called yet.
        ValueError
            If ``z`` is ``None``.
        """
        assert self.state is not None, "Filter not initialized."
        if z is None:
            raise ValueError("Observation z must be provided.")
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
        mean, cov = weighted_mean_cov(particles, w)
        self.state = PFState(particles=particles, weights=w, mean=mean, cov=cov, t=self.state.t + 1)
        return self.state

    def step(self, z: Optional[Array] = None, u: Optional[Array] = None) -> PFState:
        """Run one full PF step: predict then update.

        Parameters
        ----------
        z : ndarray, shape (nz,)
            Measurement at the current time step.
        u : ndarray, optional
            Control input (passed to the process model).  Default ``None``.

        Returns
        -------
        PFState
            Updated filter state after predict + update.
        """
        self.predict(u=u)
        return self.update(z=z)