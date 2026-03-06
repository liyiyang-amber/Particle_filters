""""
EKF/UKF-assisted EDH Particle-Flow Particle Filter (EDH-PF).

Implements the Exact Daum–Huang (EDH) particle-flow particle filter
linearised via an auxiliary EKF or UKF.  The flow migrates each particle
from the prior to the posterior by integrating an affine drift field in
pseudo-time λ ∈ [0, 1].

Classes
-------
GaussianTracker  – Protocol for any EKF/UKF that supplies (mean, cov) at
                   each step.
EKFTracker       – :class:`~extended_kalman_filter.ExtendedKalmanFilter`
                   wrapper implementing GaussianTracker.
UKFTracker       – :class:`~unscented_kalman_filter.UnscentedKalmanFilter`
                   wrapper implementing GaussianTracker.
EDHConfig        – Hyper-parameters for the EDH-PF.
PFState          – Particle-filter state container (alias of
                   :class:`~base_ssm.ParticleFilterState`).
EDHFlowPF        – Main EDH particle-flow filter class.

Shared utilities from :mod:`base_ssm`
--------------------------------------
:func:`~base_ssm.systematic_resample`,
:func:`~base_ssm.effective_sample_size`, and
:func:`~base_ssm.weighted_mean_cov` replace private reimplementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple, Union
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

class GaussianTracker(Protocol):
    """Protocol for auxiliary EKF/UKF trackers.

    Any object that supplies predicted and updated Gaussian moments
    (mean, covariance) at each time step satisfies this protocol.
    Concrete implementations are :class:`EKFTracker` and :class:`UKFTracker`.
    """

    def predict(self) -> Tuple[Array, Array]:
        """Advance the tracker one step and return the predicted moments.

        Returns
        -------
        m_pred : ndarray, shape (nx,)
            Predicted state mean x_{k|k-1}.
        P_pred : ndarray, shape (nx, nx)
            Predicted state covariance P_{k|k-1}.
        """

    def update(self, z_k: Array) -> Tuple[Array, Array]:
        """Incorporate an observation and return the posterior moments.

        Parameters
        ----------
        z_k : ndarray, shape (nz,)
            Observation at time k.

        Returns
        -------
        m_post : ndarray, shape (nx,)
            Posterior state mean x_{k|k}.
        P_post : ndarray, shape (nx, nx)
            Posterior state covariance P_{k|k}.
        """

    def get_past_mean(self) -> Array:
        """Return the posterior mean from the previous time step.

        Returns
        -------
        ndarray, shape (nx,)
            x_{k-1|k-1}, used to initialise the mean trajectory η̄_0.
        """

GFn = Callable[[Array, Optional[Array], Optional[Array]], Array]  # g(x, u, v)
HFn = Callable[[Array], Array]                                    # h(x)
JacobianHFn = Callable[[Array], Array]                            # ∂h/∂x
LogTransPdf = Callable[[Array, Array], float]                     # log p(x_k | x_{k-1})
LogLikePdf  = Callable[[Array, Array], float]                     # log p(z_k | x_k)


# Utilities
def rk4_step(x: Array, f: Callable[[Array], Array], dt: float) -> Array:
    """Perform one RK4 integration step for the ODE x' = f(x).

    Parameters
    ----------
    x : ndarray, shape (n,)
        Current state.
    f : callable
        Vector field f(x) → (n,).
    dt : float
        Step size.

    Returns
    -------
    ndarray, shape (n,)
        State after one RK4 step of size ``dt``.
    """
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


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
        Resampled ancestor indices.
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
        Effective sample size.
    """
    return _ess(weights)

# Config/State
@dataclass
class EDHConfig:
    """Hyper-parameters for the EKF/UKF-assisted EDH particle-flow PF.

    Parameters
    ----------
    n_particles : int, optional
        Number of particles. Default is 512.
    n_lambda_steps : int, optional
        Number of pseudo-time sub-steps for λ ∈ [0, 1]. Default is 8.
    resample_ess_ratio : float, optional
        Resample when ESS < ratio * n_particles. Default is 0.5.
        Set to 0 to disable resampling.
    flow_integrator : str, optional
        ODE integrator for the particle flow: ``'rk4'`` (default) or
        ``'euler'``.
    rng : np.random.Generator, optional
        NumPy random generator for resampling. Default is
        ``np.random.default_rng(0)``.
    """

    n_particles: int = 512
    n_lambda_steps: int = 8
    resample_ess_ratio: float = 0.5
    flow_integrator: str = "rk4"
    rng: np.random.Generator = np.random.default_rng(0)


# Keep PFState as alias for backward compatibility
PFState = ParticleFilterState

# Tracker Wrappers
class EKFTracker:
    """Wrapper adapting :class:`~extended_kalman_filter.ExtendedKalmanFilter`
    to the :class:`GaussianTracker` protocol.

    Parameters
    ----------
    ekf : ExtendedKalmanFilter
        Configured EKF instance.
    initial_state : EKFState
        Initial EKF state (mean, cov, t=0).
    """

    def __init__(self, ekf, initial_state):
        self.ekf = ekf
        self.state = initial_state
        self.past_mean = initial_state.mean.copy()

    def predict(self) -> Tuple[Array, Array]:
        """Run EKF prediction and return (m_{k|k-1}, P_{k|k-1}).

        Returns
        -------
        m_pred : ndarray, shape (nx,)
        P_pred : ndarray, shape (nx, nx)
        """
        self.past_mean = self.state.mean.copy()
        self.state = self.ekf.predict(self.state, u=None)
        return self.state.mean, self.state.cov

    def update(self, z_k: Array) -> Tuple[Array, Array]:
        """Run EKF measurement update and return (m_{k|k}, P_{k|k}).

        Parameters
        ----------
        z_k : ndarray, shape (nz,)
            Observation at time k.

        Returns
        -------
        m_post : ndarray, shape (nx,)
        P_post : ndarray, shape (nx, nx)
        """
        self.state = self.ekf.update(self.state, z_k)
        return self.state.mean, self.state.cov

    def get_past_mean(self) -> Array:
        """Return x_{k-1|k-1} (posterior mean from the previous step).

        Returns
        -------
        ndarray, shape (nx,)
        """
        return self.past_mean


class UKFTracker:
    """Wrapper adapting :class:`~unscented_kalman_filter.UnscentedKalmanFilter`
    to the :class:`GaussianTracker` protocol.

    Parameters
    ----------
    ukf : UnscentedKalmanFilter
        Configured UKF instance.
    initial_state : UKFState
        Initial UKF state (mean, cov, t=0).
    """

    def __init__(self, ukf, initial_state):
        self.ukf = ukf
        self.state = initial_state
        self.past_mean = initial_state.mean.copy()

    def predict(self) -> Tuple[Array, Array]:
        """Run UKF prediction and return (m_{k|k-1}, P_{k|k-1}).

        Returns
        -------
        m_pred : ndarray, shape (nx,)
        P_pred : ndarray, shape (nx, nx)
        """
        self.past_mean = self.state.mean.copy()
        self.state = self.ukf.predict(self.state, u=None)
        return self.state.mean, self.state.cov

    def update(self, z_k: Array) -> Tuple[Array, Array]:
        """Run UKF measurement update and return (m_{k|k}, P_{k|k}).

        Parameters
        ----------
        z_k : ndarray, shape (nz,)
            Observation at time k.

        Returns
        -------
        m_post : ndarray, shape (nx,)
        P_post : ndarray, shape (nx, nx)
        """
        self.state = self.ukf.update(self.state, z_k)
        return self.state.mean, self.state.cov

    def get_past_mean(self) -> Array:
        """Return x_{k-1|k-1} (posterior mean from the previous step).

        Returns
        -------
        ndarray, shape (nx,)
        """
        """Return \hat{x}_{k-1|k-1}."""
        return self.past_mean

# EDH Flow PF
class EDHFlowPF:
    """EKF/UKF-assisted EDH particle-flow particle filter.

    Implements the Exact Daum–Huang (EDH) flow filter, which migrates N
    particles from the prior to the posterior by integrating an affine
    drift field (parameterised by a linearised observation model) in
    pseudo-time λ ∈ [0, 1].  An auxiliary EKF or UKF (the *tracker*)
    supplies the global Gaussian moments used to construct the flow.

    Parameters
    ----------
    tracker : GaussianTracker
        EKF/UKF that provides (m_{k|k-1}, P) and is updated to
        (m_{k|k}, P_k) each step.  Use :class:`EKFTracker` or
        :class:`UKFTracker`.
    g : callable
        Process function g(x, u, v) → (nx,), where v is a noise sample.
    h : callable
        Observation function h(x) → (nz,).
    jacobian_h : callable
        Jacobian of h: ∂h/∂x evaluated at x → (nz, nx).
    log_trans_pdf : callable
        Log transition density log p(x_k | x_{k-1}) → float.
    log_like_pdf : callable
        Log likelihood log p(z_k | x_k) → float.
    R : ndarray, shape (nz, nz)
        Observation-noise covariance used in the flow ODE.
    config : EDHConfig, optional
        Algorithm hyper-parameters. Default constructs an :class:`EDHConfig`
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
        config: Optional[EDHConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        tracker : GaussianTracker
            EKF/UKF that provides (m_{k|k-1}, P) and updates to (m_{k|k}, P_k).
        g, h, jacobian_h : callables
            Process/observation models and Jacobian of h.
        log_trans_pdf, log_like_pdf : callables
            Log transition and log likelihood densities.
        R : Array
            Observation noise covariance (nz, nz) used in the flow.
        config : Optional[EDHConfig]
            Filter configuration.
        """
        self.tracker = tracker
        self.g = g
        self.h = h
        self.Jh = jacobian_h
        self.log_trans_pdf = log_trans_pdf
        self.log_like_pdf = log_like_pdf
        self.R = np.array(R, dtype=float)
        self.cfg = config or EDHConfig()

    # API
    def init_from_gaussian(self, mean0: Array, cov0: Array) -> PFState:
        """Initialise particles from N(mean0, cov0) with uniform weights.

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
        """Run one EDH-PF step (propagate + flow + weight update).

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
            noise samples for the propagation step.  If ``None``, zero noise
            is used (i.e. purely deterministic propagation).

        Returns
        -------
        PFState
            Updated particle-filter state at time k.
        """
        N, nx = state.particles.shape

        # EKF/UKF prediction: (m_{k|k-1}, P_{k|k-1}) 
        m_pred, P = self.tracker.predict()
        # Enforce symmetry on P
        P = 0.5 * (P + P.T)

        # Propagate particles to η_0^i = g(x_{k-1}^i, v) 
        if process_noise_sampler is None:
            # Default: zero process noise (caller should provide proper sampler)
            v = np.zeros((N, nx))
        else:
            v = process_noise_sampler(N, nx)
        
        eta0 = np.empty_like(state.particles)
        for i in range(N):
            eta0[i] = self.g(state.particles[i], u_km1, v[i])

        # Initialize flow states η_1^i <- η_0^i, \bar{\eta} <- \bar{\eta}_0 
        eta = eta0.copy()  # η_1^i
        # Compute mean trajectory initialization: \bar{\eta}_0 = g_k(\hat{x}_{k-1}, 0)
        etabar = self.g(self.tracker.get_past_mean(), u_km1, np.zeros(nx))
        
        # Flow update in pseudo-time \lambda \in [0,1]
        n_steps = max(1, int(self.cfg.n_lambda_steps))
        dlam = 1.0 / float(n_steps)  # \epsilon_j
        lam = 0.0

        I = np.eye(nx)
        
        # Track condition numbers for diagnostics
        cond_numbers = []

        for _ in range(n_steps):
            # \lambda <- \lambda + \epsilon_j (update at start for correct \lambda usage)
            lam = min(1.0, lam + dlam)
            
            # Linearize observation model at current mean \bar{\eta}_\lambda
            H = self.Jh(etabar)              # (nz, nx)
            h_bar = self.h(etabar)           # (nz,)
            e = h_bar - H @ etabar           # e(\lambda) = h(\bar{\eta}) - H \bar{\eta}

            # Compute flow matrices
            # S(\lambda) = \lambda H P H^T + R  
            S = lam * H @ P @ H.T + self.R
            
            # Track condition number for diagnostics
            try:
                cond_S = np.linalg.cond(S)
                cond_numbers.append(float(cond_S))
            except:
                cond_numbers.append(np.nan)
            
            # Use solves for numerical stability 
            # A(\lambda) = -1/2 P H^T S^{-1} H
            try:
                S_inv_H = np.linalg.solve(S, H)  # Solve S * X = H for X
            except np.linalg.LinAlgError:
                # Add small jitter if singular
                S = S + 1e-8 * np.eye(S.shape[0])
                S_inv_H = np.linalg.solve(S, H)
            
            A = -0.5 * P @ H.T @ S_inv_H     # (nx, nx)

            # b(\lambda) = (I + 2\lambda A)[(I + \lambda A) P H^T R^{-1}(z - e) + A \bar{\eta}]
            try:
                R_inv_innov = np.linalg.solve(self.R, (z_k - e))
            except np.linalg.LinAlgError:
                R_inv = np.linalg.inv(self.R + 1e-8 * np.eye(self.R.shape[0]))
                R_inv_innov = R_inv @ (z_k - e)
                
            PHt_Rinv_innov = P @ H.T @ R_inv_innov
            b = (I + 2.0 * lam * A) @ ( (I + lam * A) @ PHt_Rinv_innov + A @ etabar )

            # Update mean trajectory and particles
            # Affine field: d\eta/d\lambda = A \eta + b
            def field(vec: Array) -> Array:
                return A @ vec + b

            if self.cfg.flow_integrator.lower() == "euler":
                # Euler integration: \eta <- \eta + \epsilon_j [A \eta + b]
                # Vectorized for all particles
                eta    = eta    + dlam * (eta @ A.T + b)
                etabar = etabar + dlam * field(etabar)
            else:
                # RK4 integration for better accuracy
                for i in range(N):
                    eta[i] = rk4_step(eta[i], field, dlam)
                etabar = rk4_step(etabar, field, dlam)

        # Posterior correction (weight update) 
        xk = eta  # x_k^i <- \eta_1^i

        # Compute weights (log domain for numerical stability)
        # w_k^i \propto w_{k-1}^i · [p(x_k^i|x_{k-1}^i) · p(z_k|x_k^i)] / p(\eta_0^i|x_{k-1}^i)
        logw = np.log(state.weights + 1e-300)
        for i in range(N):
            log_trans_xk = self.log_trans_pdf(xk[i], state.particles[i])
            log_like = self.log_like_pdf(z_k, xk[i])
            log_trans_eta0 = self.log_trans_pdf(eta0[i], state.particles[i])
            
            logw[i] += (log_trans_xk + log_like - log_trans_eta0)
        
        # Normalize weights
        logw -= np.max(logw)  # For numerical stability
        w = np.exp(logw)
        w /= np.sum(w)

        # EKF/UKF measurement update (tracker only) 
        self.tracker.update(z_k)

        # Optional resampling based on ESS 
        if self.cfg.resample_ess_ratio > 0.0:
            ess = effective_sample_size(w)
            if ess < self.cfg.resample_ess_ratio * N:
                idx = systematic_resample(w, self.cfg.rng)
                xk = xk[idx]
                w = np.full_like(w, 1.0 / N)

        # Estimate mean and covariance
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

