"""
Base classes and shared utilities for state-space model (SSM) filtering.

This module defines:

- :class:`BaseSSM`        – Abstract description of a generic SSM (process and
                             observation models, noise covariances).
- :class:`BaseFilterState` – Minimal container for a filter posterior at one
                              time step.
- :class:`GaussianFilterState` – Gaussian filter posterior (mean + covariance).
- :class:`ParticleFilterState` – Particle-filter posterior (particles + weights).
- :class:`BaseFilter`     – Abstract filter interface (``initialize``,
                             ``predict``, ``update``, ``step``, ``run``).
- Shared numeric utilities used by ≥2 concrete algorithms:
    - :func:`weighted_mean_cov`
    - :func:`systematic_resample`
    - :func:`multinomial_resample`
    - :func:`effective_sample_size`
    - :func:`chol_solve`
    - :func:`numerical_jacobian`
    - :func:`symmetrise`
    - :func:`log_gaussian_pdf`

Design intent
-------------
Concrete algorithms (KF, EKF, UKF, PF, EDH-PF, LEDH-PF, SPF, KPF, DPF
variants) should:

1. Import the shared utilities from this module instead of reimplementing them.
2. Optionally subclass :class:`BaseFilter` to gain a uniform ``run()`` loop.

All public classes and functions follow the NumPy docstring convention.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Array = NDArray[np.float64]
GFn = Callable[[Array, Optional[Array]], Array]
HFn = Callable[[Array], Array]


# ===========================================================================
# Base state-space model description
# ===========================================================================

class BaseSSM(abc.ABC):
    """Abstract base class for a discrete-time state-space model (SSM).

    Subclasses must implement the process model :meth:`g` and the observation
    model :meth:`h`, together with their noise covariances ``Q`` and ``R``.

    The general (possibly nonlinear) SSM is::

        x_k  = g(x_{k-1}, u_{k-1}) + w_{k-1},   w ~ N(0, Q)
        z_k  = h(x_k)               + v_k,       v ~ N(0, R)

    Parameters
    ----------
    Q : ndarray, shape (nx, nx)
        Process-noise covariance.
    R : ndarray, shape (nz, nz)
        Measurement-noise covariance.

    Attributes
    ----------
    nx : int
        State dimension (inferred from ``Q``).
    nz : int
        Observation dimension (inferred from ``R``).
    Q : ndarray
        Process-noise covariance (nx, nx).
    R : ndarray
        Measurement-noise covariance (nz, nz).

    Notes
    -----
    Assumes additive Gaussian process and measurement noise with covariance
    matrices that are square and dimensionally consistent with the state and
    observation models.
    """

    def __init__(self, Q: Array, R: Array) -> None:
        self.Q: Array = np.asarray(Q, dtype=float)
        self.R: Array = np.asarray(R, dtype=float)
        self.nx: int = int(self.Q.shape[0])
        self.nz: int = int(self.R.shape[0])
        if self.Q.shape != (self.nx, self.nx):
            raise ValueError("Q must be square (nx, nx).")
        if self.R.shape != (self.nz, self.nz):
            raise ValueError("R must be square (nz, nz).")

    @abc.abstractmethod
    def g(self, x: Array, u: Optional[Array] = None) -> Array:
        """Process (transition) model.

        Parameters
        ----------
        x : ndarray, shape (nx,)
            Current state.
        u : ndarray, optional
            Control input. ``None`` if no input.

        Returns
        -------
        ndarray, shape (nx,)
            Predicted next state (noise-free).
        """

    @abc.abstractmethod
    def h(self, x: Array) -> Array:
        """Observation model.

        Parameters
        ----------
        x : ndarray, shape (nx,)
            Current state.

        Returns
        -------
        ndarray, shape (nz,)
            Predicted observation (noise-free).
        """


# ===========================================================================
# Filter state containers
# ===========================================================================

@dataclass
class BaseFilterState:
    """Minimal container for a filter posterior at one time step.

    Notes
    -----
    Concrete subclasses define the actual stored posterior representation,
    such as Gaussian moments or weighted particles.
    """
    pass


@dataclass
class GaussianFilterState(BaseFilterState):
    """Gaussian filter posterior (mean + full covariance).

    Parameters
    ----------
    mean : ndarray, shape (nx,)
        Posterior state mean.
    cov : ndarray, shape (nx, nx)
        Posterior state covariance.
    t : int, optional
        Discrete time index of this posterior.  Defaults to ``0``.

    Notes
    -----
    Assumes ``cov`` is symmetric and corresponds to the uncertainty of
    ``mean`` at time index ``t``.
    """

    mean: Array
    cov: Array
    t: int = 0


@dataclass
class ParticleFilterState(BaseFilterState):
    """Particle-filter posterior (weighted particle ensemble).

    Parameters
    ----------
    particles : ndarray, shape (Np, nx)
        Particle states.
    weights : ndarray, shape (Np,)
        Normalized importance weights (sum to 1).
    mean : ndarray, shape (nx,)
        Weighted posterior mean.
    cov : ndarray, shape (nx, nx)
        Weighted posterior covariance.
    t : int, optional
        Discrete time index of this posterior.  Defaults to ``0``.
    diagnostics : dict, optional
        Optional algorithm-specific diagnostics (e.g., ESS, condition numbers).

    Notes
    -----
    Assumes ``weights`` are normalised and aligned with the first dimension of
    ``particles``.
    """

    particles: Array
    weights: Array
    mean: Array
    cov: Array
    t: int = 0
    diagnostics: dict = field(default_factory=dict)


# ===========================================================================
# Abstract filter interface
# ===========================================================================

class BaseFilter(abc.ABC):
    """Abstract base class defining a uniform filter interface.

    All concrete filters (KF, EKF, UKF, PF, EDH-PF, …) should inherit from
    this class to guarantee a consistent API.

    The required interface methods are:

    - :meth:`initialize` – set up the filter from an initial distribution.
    - :meth:`predict`    – time-update / propagation step.
    - :meth:`update`     – measurement-update step.
    - :meth:`step`       – combined ``predict`` + ``update``.
    - :meth:`run`        – default sequential loop over an observation sequence.

    Subclasses that deviate from the standard predict→update cycle can
    override :meth:`run` directly.

    Notes
    -----
    Assumes concrete implementations preserve a consistent state type across
    ``initialize``, ``predict``, ``update``, ``step``, and ``run``.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def initialize(self, mean: Array, cov: Array) -> BaseFilterState:
        """Initialize the filter from a Gaussian prior N(mean, cov).

        Parameters
        ----------
        mean : ndarray, shape (nx,)
            Initial state mean.
        cov : ndarray, shape (nx, nx)
            Initial state covariance.

        Returns
        -------
        BaseFilterState
            Initial filter state.
        """

    @abc.abstractmethod
    def predict(self, state: BaseFilterState, u: Optional[Array] = None) -> BaseFilterState:
        """Propagate the filter state through the process model.

        Parameters
        ----------
        state : BaseFilterState
            Current (posterior) filter state.
        u : ndarray, optional
            Control input for the process model. ``None`` if no input.

        Returns
        -------
        BaseFilterState
            Predicted (prior) filter state at the next time step.
        """

    @abc.abstractmethod
    def update(self, pred: BaseFilterState, z: Array) -> BaseFilterState:
        """Correct the predicted state with a new observation.

        Parameters
        ----------
        pred : BaseFilterState
            Predicted filter state (from :meth:`predict`).
        z : ndarray, shape (nz,)
            Observed measurement at the current time step.

        Returns
        -------
        BaseFilterState
            Posterior filter state after the measurement update.
        """

    def step(
        self,
        state: BaseFilterState,
        z: Array,
        u: Optional[Array] = None,
    ) -> BaseFilterState:
        """Run a single full filter step: predict then update.

        Parameters
        ----------
        state : BaseFilterState
            Previous posterior filter state.
        z : ndarray, shape (nz,)
            Measurement at the next time step.
        u : ndarray, optional
            Control input for the process model. ``None`` if no input.

        Returns
        -------
        BaseFilterState
            Updated posterior filter state.
        """
        pred = self.predict(state, u=u)
        return self.update(pred, z)

    def run(
        self,
        Z: Array,
        mean0: Array,
        cov0: Array,
        U: Optional[Array] = None,
    ) -> list[BaseFilterState]:
        """Run the filter over a full observation sequence.

        Parameters
        ----------
        Z : ndarray, shape (T, nz)
            Sequence of T observations.
        mean0 : ndarray, shape (nx,)
            Initial state mean for ``initialize``.
        cov0 : ndarray, shape (nx, nx)
            Initial state covariance for ``initialize``.
        U : ndarray, shape (T, nu), optional
            Control inputs aligned with observations. ``None`` if no inputs.

        Returns
        -------
        list of BaseFilterState
            Posterior states at each of the T time steps.
        """
        if Z.ndim != 2:
            raise ValueError("Z must be 2D with shape (T, nz).")
        T = Z.shape[0]
        states: list[BaseFilterState] = []
        state = self.initialize(mean0, cov0)
        for k in range(T):
            u_k = None if U is None else U[k]
            state = self.step(state, Z[k], u=u_k)
            states.append(state)
        return states


# ===========================================================================
# Shared numeric utilities
# ===========================================================================

def symmetrise(M: Array) -> Array:
    """Enforce exact symmetry of a square matrix: (M + M^T) / 2.

    Parameters
    ----------
    M : ndarray, shape (n, n)
        Input matrix.

    Returns
    -------
    ndarray, shape (n, n)
        Symmetrised matrix.
    """
    return 0.5 * (M + M.T)


def chol_solve(L: Array, B: Array) -> Array:
    """Solve (L L^T) X = B using Cholesky triangular factors.

    Parameters
    ----------
    L : ndarray, shape (m, m)
        Lower-triangular Cholesky factor of the SPD matrix A = L L^T.
    B : ndarray, shape (m, k)
        Right-hand side matrix (or vector of shape (m,)).

    Returns
    -------
    ndarray, shape (m, k)
        Solution X satisfying A X = B.
    """
    Z = np.linalg.solve(L, B)
    return np.linalg.solve(L.T, Z)


def log_gaussian_pdf(x: Array, mean: Array, L: Array) -> float:
    """Evaluate the log-density of N(x | mean, L L^T).

    Parameters
    ----------
    x : ndarray, shape (n,)
        Query point.
    mean : ndarray, shape (n,)
        Distribution mean.
    L : ndarray, shape (n, n)
        Lower-triangular Cholesky factor of the covariance matrix.

    Returns
    -------
    float
        Log-density log N(x | mean, L L^T).
    """
    n = x.size
    r = x - mean
    alpha = np.linalg.solve(L, r)
    quad = float(alpha @ alpha)
    logdet = 2.0 * np.sum(np.log(np.abs(np.diag(L))))
    return -0.5 * (quad + logdet + n * np.log(2.0 * np.pi))


def numerical_jacobian(
    f: Callable[[Array], Array],
    x: Array,
    *,
    eps: float = 1e-6,
    args: tuple = (),
) -> Array:
    """Compute a first-order forward finite-difference Jacobian of f at x.

    Parameters
    ----------
    f : callable
        Scalar-to-vector function f(x, *args) with output shape (m,).
    x : ndarray, shape (n,)
        Expansion point.
    eps : float, optional
        Finite-difference step size. Default is 1e-6.
    args : tuple, optional
        Extra positional arguments forwarded to ``f``.

    Returns
    -------
    ndarray, shape (m, n)
        Jacobian matrix ∂f/∂x evaluated at x.
    """
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(f(x, *args), dtype=float)
    m, n = y0.size, x.size
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        dx = np.zeros(n, dtype=float)
        dx[j] = eps
        J[:, j] = (f(x + dx, *args) - y0) / eps
    return J


def effective_sample_size(weights: Array) -> float:
    """Compute the Effective Sample Size (ESS) from particle weights.

    Uses the standard formula ESS = 1 / Σ_i w_i^2, which equals N when
    weights are uniform and 1 when a single particle carries all weight.

    Parameters
    ----------
    weights : ndarray, shape (Np,)
        Normalized particle weights (should sum to 1).

    Returns
    -------
    float
        Effective sample size in the range [1, Np].
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float(1.0 / np.sum(w * w))


def systematic_resample(weights: Array, rng: np.random.Generator) -> Array:
    """Systematic resampling of a particle set.

    Draws a single uniform sample and generates N stratified positions
    to index into the cumulative weight distribution.  Has O(N) cost and
    lower variance than multinomial resampling.

    Parameters
    ----------
    weights : ndarray, shape (Np,)
        Normalized particle weights (should sum to 1).
    rng : np.random.Generator
        NumPy random generator used for reproducibility.

    Returns
    -------
    ndarray of int, shape (Np,)
        Resampled ancestor indices in [0, Np).
    """
    N = len(weights)
    w = np.asarray(weights, dtype=float)
    positions = (rng.random() + np.arange(N)) / N
    indexes = np.zeros(N, dtype=int)
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0  # guard against round-off
    i = j = 0
    while i < N:
        if positions[i] < cumsum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def multinomial_resample(weights: Array, rng: np.random.Generator) -> Array:
    """Multinomial resampling of a particle set.

    Independently draws N ancestor indices from the categorical distribution
    defined by the weights.  Higher variance than systematic resampling but
    simpler to analyse.

    Parameters
    ----------
    weights : ndarray, shape (Np,)
        Normalized particle weights (should sum to 1).
    rng : np.random.Generator
        NumPy random generator used for reproducibility.

    Returns
    -------
    ndarray of int, shape (Np,)
        Resampled ancestor indices in [0, Np).
    """
    return rng.choice(len(weights), size=len(weights), p=weights)


def weighted_mean_cov(
    particles: Array,
    weights: Array,
    *,
    symmetrize: bool = True,
) -> Tuple[Array, Array]:
    """Compute the weighted sample mean and covariance of a particle set.

    Parameters
    ----------
    particles : ndarray, shape (Np, nx)
        Particle state matrix.
    weights : ndarray, shape (Np,)
        Normalized importance weights (need not sum to 1; they are
        re-normalised internally).
    symmetrize : bool, optional
        If ``True`` (default), enforce symmetry of the returned covariance
        via (C + C^T) / 2.

    Returns
    -------
    mean : ndarray, shape (nx,)
        Weighted sample mean.
    cov : ndarray, shape (nx, nx)
        Weighted sample covariance.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mean = np.einsum("i,ij->j", w, particles)
    diff = particles - mean[None, :]
    cov = np.einsum("i,ij,ik->jk", w, diff, diff)
    if symmetrize:
        cov = symmetrise(cov)
    return mean, cov
