"""
EKF/UKF-assisted Local-Exact Daum–Huang (LEDH) Particle-Flow PF.

Implements the LEDH particle-flow particle filter, which evolves each
particle independently along its own per-particle linearised flow field
in pseudo-time λ ∈ [0, 1].

Classes
-------
GaussianTracker – Protocol for EKF/UKF trackers.
LEDHConfig      – Hyper-parameters for the LEDH-PF.
PFState         – Particle-filter state container (alias of
                  :class:`~base_ssm.ParticleFilterState`).
LEDHFlowPF      – Main LEDH particle-flow filter class.

Shared utilities from :mod:`base_ssm`
--------------------------------------
:func:`~base_ssm.systematic_resample`,
:func:`~base_ssm.effective_sample_size`, and
:func:`~base_ssm.weighted_mean_cov` replace private reimplementations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple
import numpy as np

try:
    from models.base_ssm import (
        ParticleFilterState,
        systematic_resample as _systematic_resample,
        effective_sample_size as _ess,
        weighted_mean_cov,
    )
except ModuleNotFoundError:
    from base_ssm import (
        ParticleFilterState,
        systematic_resample as _systematic_resample,
        effective_sample_size as _ess,
        weighted_mean_cov,
    )

Array = np.ndarray

# Protocols
class GaussianTracker(Protocol):
    """Protocol for auxiliary EKF/UKF trackers.

    Any object that supplies predicted and updated Gaussian moments
    (mean, covariance) at each step satisfies this protocol.
    """

    def predict(self) -> Tuple[Array, Array]:
        """Advance the tracker and return (m_{k|k-1}, P_{k|k-1}).

        Returns
        -------
        m_pred : ndarray, shape (nx,)
        P_pred : ndarray, shape (nx, nx)
        """

    def update(self, z_k: Array) -> Tuple[Array, Array]:
        """Incorporate an observation and return (m_{k|k}, P_{k|k}).

        Parameters
        ----------
        z_k : ndarray, shape (nz,)

        Returns
        -------
        m_post : ndarray, shape (nx,)
        P_post : ndarray, shape (nx, nx)
        """

    def get_past_mean(self) -> Array:
        """Return x_{k-1|k-1}.

        Returns
        -------
        ndarray, shape (nx,)
        """

GFn = Callable[[Array, Optional[Array], Optional[Array]], Array]
HFn = Callable[[Array], Array]
JacobianHFn = Callable[[Array], Array]
LogTransPdf = Callable[[Array, Array], float]
LogLikePdf  = Callable[[Array, Array], float]

# Utilities (backward-compat thin wrappers over base_ssm shared functions)

def systematic_resample(weights: Array, rng: np.random.Generator) -> Array:
    """Systematic resampling; returns ancestor indices.

    Delegates to :func:`~base_ssm.systematic_resample`.

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Normalized particle weights.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    ndarray of int, shape (N,)
    """
    return _systematic_resample(weights, rng)


def effective_sample_size(weights: Array) -> float:
    """Compute ESS = 1 / Σ w_i^2 from normalized weights.

    Delegates to :func:`~base_ssm.effective_sample_size`.

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Normalized particle weights.

    Returns
    -------
    float
    """
    return _ess(weights)


# Config/State
@dataclass
class LEDHConfig:
    """Hyper-parameters for the EKF/UKF-assisted LEDH particle-flow PF.

    Parameters
    ----------
    n_particles : int, optional
        Number of particles. Default is 512.
    n_lambda_steps : int, optional
        Number of pseudo-time sub-steps for λ ∈ [0, 1]. Default is 8.
    resample_ess_ratio : float, optional
        Resample when ESS < ratio * n_particles. Set to 0 to disable.
        Default is 0.0.
    rng : np.random.Generator, optional
        NumPy random generator for resampling. Default is
        ``np.random.default_rng(0)``.
    """

    n_particles: int = 512
    n_lambda_steps: int = 8
    resample_ess_ratio: float = 0.0
    rng: np.random.Generator = np.random.default_rng(0)


# Keep PFState as alias for backward compatibility
PFState = ParticleFilterState

# LEDH Flow PF
class LEDHFlowPF:
    """EKF/UKF-assisted LEDH particle-flow particle filter (Algorithm 1).

    The LEDH variant differs from EDH in that each particle is linearised
    at *its own current position* (local linearisation) rather than at the
    shared mean trajectory.

    Parameters
    ----------
    tracker : GaussianTracker
        EKF/UKF that provides global Gaussian moments (m, P) at each step.
        Use an :class:`~EDH_particle_filter.EKFTracker` or
        :class:`~EDH_particle_filter.UKFTracker`.
    g : callable
        Process function g(x, u, v) → (nx,).
    h : callable
        Observation function h(x) → (nz,).
    jacobian_h : callable
        Jacobian ∂h/∂x evaluated at x → (nz, nx).
    log_trans_pdf : callable
        Log transition density log p(x_k | x_{k-1}) → float.
    log_like_pdf : callable
        Log likelihood log p(z_k | x_k) → float.
    R : ndarray, shape (nz, nz)
        Observation-noise covariance used in the per-particle flow ODE.
    config : LEDHConfig, optional
        Algorithm hyper-parameters. Default constructs a :class:`LEDHConfig`
        with default values.
    """

    def __init__(
        self,
        tracker: GaussianTracker,
        g: GFn,
        h: HFn,
        jacobian_h: JacobianHFn,
        log_trans_pdf: LogTransPdf,
        log_like_pdf: LogLikePdf,
        R: Array,
        config: Optional[LEDHConfig] = None,
    ) -> None:
        self.tracker = tracker
        self.g = g
        self.h = h
        self.Jh = jacobian_h
        self.log_trans_pdf = log_trans_pdf
        self.log_like_pdf = log_like_pdf
        self.R = np.array(R, dtype=float)
        self.cfg = config or LEDHConfig()

    # API
    def init_from_gaussian(self, mean0: Array, cov0: Array) -> PFState:
        """Initialise particles from N(mean0, cov0) with uniform weights.

        Corresponds to Algorithm lines 1–2: sample prior particles and set
        uniform weights.

        Parameters
        ----------
        mean0 : ndarray, shape (nx,)
            Initial state mean.
        cov0 : ndarray, shape (nx, nx)
            Initial state covariance.

        Returns
        -------
        PFState
            Initial particle-filter state.
        """
        n, nx = self.cfg.n_particles, mean0.size
        eps = self.cfg.rng.multivariate_normal(np.zeros(nx), cov0, size=n)
        particles = mean0[None, :] + eps
        weights = np.full(n, 1.0 / n)
        mean, cov = weighted_mean_cov(particles, weights)
        return PFState(particles=particles, weights=weights, mean=mean, cov=cov, diagnostics={})

    def step(
        self,
        state: PFState,
        z_k: Array,
        u_km1: Optional[Array] = None,
        process_noise_sampler: Optional[Callable[[int, int], Array]] = None,
    ) -> PFState:
        """Run one LEDH step (per-particle local linearisation, Algorithm 1).

        Parameters
        ----------
        state : PFState
            Particle-filter state from the previous time step.
        z_k : ndarray, shape (nz,)
            Observation at time k.
        u_km1 : ndarray, optional
            Control input u_{k-1}.  ``None`` if no input.
        process_noise_sampler : callable, optional
            ``process_noise_sampler(N, nx) → (N, nx)`` that draws process
            noise samples.  If ``None``, zero noise is used.

        Returns
        -------
        PFState
            Updated particle-filter state at time k.
        """
        N, nx = state.particles.shape
        I = np.eye(nx)

        # EKF/UKF prediction to obtain P^i
        m_pred, P = self.tracker.predict()
        P = 0.5 * (P + P.T)  # symmetry

        # Propagate particles η_0^i = g_k(x_{k-1}^i, v_k)
        if process_noise_sampler is None:
            v = np.zeros((N, nx))  # provide sampler with Q in real runs
        else:
            v = process_noise_sampler(N, nx)
        eta0 = np.empty_like(state.particles)
        for i in range(N):
            eta0[i] = self.g(state.particles[i], u_km1, v[i])

        # Initialize flow variables
        # η_1^i = η_0^i (will evolve to η_λ^i during flow)
        eta = eta0.copy()
        
        # Calculate ȳ_0^i = g_k(x_{k-1}^i, 0)
        # Per line 15: ȳ_0 = ȳ_0^i suggests per-particle initialization
        etabar = eta0.copy()  # ȳ_0^i = η_0^i (strict LEDH per line 15)
        
        # θ^i = 1
        theta_log = np.zeros(N)  # log θ^i for numerical stability

        # Track condition numbers for diagnostics
        cond_numbers = []

        # Pseudo-time integration λ ∈ [0,1]
        n_steps = max(1, int(self.cfg.n_lambda_steps))
        dlam = 1.0 / float(n_steps)  # eps_j
        lam = 0.0

        for _ in range(n_steps):
            lam = min(1.0, lam + dlam)  # Algorithm line 13: λ = λ + ε_j
            
            # For i = 1, ..., N_p
            for i in range(N):
                # Linearize at η_λ^i (CRITICAL CORRECTION)
                # H^i(λ) = ∂h/∂η |_{η=η_λ^i}
                Hi = self.Jh(eta[i])  # Linearize at PARTICLE position, not mean path
                h_eta_i = self.h(eta[i])
                ei = h_eta_i - Hi @ eta[i]  # e^i(λ) = h(η_λ^i, 0) - H^i(λ) η_λ^i

                # Calculate A^i(λ)
                # A^i(λ) = -½ P H^i(λ)^T (λ H^i(λ) P H^i(λ)^T + R)^{-1} H^i(λ)
                Si = lam * Hi @ P @ Hi.T + self.R
                
                # Track condition number for diagnostics (only first particle to avoid redundancy)
                if i == 0:
                    try:
                        cond_Si = np.linalg.cond(Si)
                        cond_numbers.append(float(cond_Si))
                    except:
                        cond_numbers.append(np.nan)
                Si_inv_Hi = np.linalg.solve(Si, Hi)
                Ai = -0.5 * P @ Hi.T @ Si_inv_Hi

                # Calculate b^i(λ)
                # b^i(λ) = (I + 2λA^i)[(I + λA^i)PH^iT R^{-1}(z - e^i) + A^i η_0^i]
                Rin_innov_i = np.linalg.solve(self.R, (z_k - ei))
                PHt_Rinv_innov_i = P @ Hi.T @ Rin_innov_i
                bi = (I + 2.0 * lam * Ai) @ ((I + lam * Ai) @ PHt_Rinv_innov_i + Ai @ eta0[i])

                # Migrate ȳ_j^i
                etabar[i] = etabar[i] + dlam * (Ai @ etabar[i] + bi)

                # Migrate particles η_j^i
                eta[i] = eta[i] + dlam * (Ai @ eta[i] + bi)

                # Calculate θ^i = θ^i |det(I + ε_j A^i)|
                M = I + dlam * Ai
                sign, logdet = np.linalg.slogdet(M)
                if sign <= 0:
                    # Add small jitter for numerical robustness
                    sign, logdet = np.linalg.slogdet(M + 1e-12 * I)
                theta_log[i] += logdet

        # Set x_k^i = η_1^i
        xk = eta

        # Calculate weights
        # w_k^i = [p(z_k|x_k^i) p(x_k^i|x_{k-1}^i) θ^i] / [p(η_0^i|x_{k-1}^i)] w_{k-1}^i
        logw = np.log(state.weights + 1e-300) + theta_log
        for i in range(N):
            num = self.log_trans_pdf(xk[i], state.particles[i]) + self.log_like_pdf(z_k, xk[i])
            den = self.log_trans_pdf(eta0[i], state.particles[i])
            logw[i] += (num - den)
        logw -= np.max(logw)
        w = np.exp(logw)
        
        # Normalize weights
        w /= np.sum(w)

        # EKF/UKF measurement update
        self.tracker.update(z_k)

        # (Optional) Resample
        if self.cfg.resample_ess_ratio > 0.0:
            ess = effective_sample_size(w)
            if ess < self.cfg.resample_ess_ratio * N:
                idx = systematic_resample(w, self.cfg.rng)
                xk = xk[idx]
                w = np.full_like(w, 1.0 / N)

        # Algorithm line 30: Estimate x̂_k
        mean, cov = weighted_mean_cov(xk, w)

        # Package diagnostics
        diagnostics = {'condition_numbers': cond_numbers}

        return PFState(particles=xk, weights=w, mean=mean, cov=cov, diagnostics=diagnostics)

    # helpers
    @staticmethod
    def _weighted_stats(x: Array, w: Array) -> Tuple[Array, Array]:
        """Compute weighted mean and covariance.

        Delegates to :func:`~base_ssm.weighted_mean_cov`.

        Parameters
        ----------
        x : ndarray, shape (N, nx)
            Particle states.
        w : ndarray, shape (N,)
            Particle weights (need not sum to 1).

        Returns
        -------
        mean : ndarray, shape (nx,)
        cov : ndarray, shape (nx, nx)
        """
        return weighted_mean_cov(x, w)