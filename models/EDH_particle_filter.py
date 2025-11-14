from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple, Union
import numpy as np

Array = np.ndarray

# ----------------------------- Protocols ---------------------------------

class GaussianTracker(Protocol):
    """Auxiliary EKF/UKF that supplies (m, P) and carries them forward."""
    def predict(self) -> Tuple[Array, Array]:
        """Return (m_{k|k-1}, P_{k|k-1}) for current step and update internal clock."""
    def update(self, z_k: Array) -> Tuple[Array, Array]:
        """Measurement update; return (m_{k|k}, P_{k|k})."""
    def get_past_mean(self) -> Array:
        """Return \hat{x}_{k-1|k-1} (used to form \bar\eta_0 = g_k(\hat{x}_{k-1}, 0))."""

GFn = Callable[[Array, Optional[Array], Optional[Array]], Array]  # x->g(x,u,v)
HFn = Callable[[Array], Array]                                    # x->h(x)
JacobianHFn = Callable[[Array], Array]                            # x->\partial h/\partial x
LogTransPdf = Callable[[Array, Array], float]                     # log p(x_k|x_{k-1})
LogLikePdf  = Callable[[Array, Array], float]                     # log p(z_k|x_k)

# ----------------------------- Utils -------------------------------------

def rk4_step(x: Array, f: Callable[[Array], Array], dt: float) -> Array:
    """One RK4 step for x' = f(x)."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def systematic_resample(weights: Array, rng: np.random.Generator) -> Array:
    """Systematic resampling; returns ancestor indices."""
    n = weights.size
    w = weights / np.sum(weights)
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(w)
    idx = np.zeros(n, dtype=int)
    i = j = 0
    while i < n:
        if positions[i] < cdf[j]:
            idx[i] = j; i += 1
        else:
            j += 1
    return idx

def effective_sample_size(weights: Array) -> float:
    """ESS = 1 / sum_i w_i^2 (with normalized weights)."""
    w = weights / np.sum(weights)
    return 1.0 / float(np.sum(w * w))

# ----------------------------- Config/State -------------------------------

@dataclass
class EDHConfig:
    """Configuration for EKF/UKF-assisted EDH-PF."""
    n_particles: int = 512
    n_lambda_steps: int = 8                  # substeps for lambda in [0,1]
    resample_ess_ratio: float = 0.5          # resample when ESS < ratio * N
    flow_integrator: str = "rk4"             # "rk4" or "euler"
    rng: np.random.Generator = np.random.default_rng(0)

@dataclass
class PFState:
    """Particle filter state container."""
    particles: Array                         # (N, nx)
    weights: Array                           # (N,)
    mean: Array                              # (nx,)
    cov: Array                               # (nx, nx)
    diagnostics: dict = None                 # optional flow diagnostics (e.g., condition numbers)

# ----------------------------- Tracker Wrappers ---------------------------

class EKFTracker:
    """Wrapper for ExtendedKalmanFilter to match GaussianTracker protocol."""
    
    def __init__(self, ekf, initial_state):
        """
        Args:
            ekf: ExtendedKalmanFilter instance
            initial_state: EKFState with initial mean, cov, t
        """
        self.ekf = ekf
        self.state = initial_state
        self.past_mean = initial_state.mean.copy()
        
    def predict(self) -> Tuple[Array, Array]:
        """Run prediction and return (m_{k|k-1}, P_{k|k-1})."""
        self.past_mean = self.state.mean.copy()
        self.state = self.ekf.predict(self.state, u=None)
        return self.state.mean, self.state.cov
    
    def update(self, z_k: Array) -> Tuple[Array, Array]:
        """Run measurement update and return (m_{k|k}, P_{k|k})."""
        self.state = self.ekf.update(self.state, z_k)
        return self.state.mean, self.state.cov
    
    def get_past_mean(self) -> Array:
        """Return \hat{x}_{k-1|k-1}."""
        return self.past_mean


class UKFTracker:
    """Wrapper for UnscentedKalmanFilter to match GaussianTracker protocol."""
    
    def __init__(self, ukf, initial_state):
        """
        Args:
            ukf: UnscentedKalmanFilter instance
            initial_state: UKFState with initial mean, cov, t
        """
        self.ukf = ukf
        self.state = initial_state
        self.past_mean = initial_state.mean.copy()
        
    def predict(self) -> Tuple[Array, Array]:
        """Run prediction and return (m_{k|k-1}, P_{k|k-1})."""
        self.past_mean = self.state.mean.copy()
        self.state = self.ukf.predict(self.state, u=None)
        return self.state.mean, self.state.cov
    
    def update(self, z_k: Array) -> Tuple[Array, Array]:
        """Run measurement update and return (m_{k|k}, P_{k|k})."""
        self.state = self.ukf.update(self.state, z_k)
        return self.state.mean, self.state.cov
    
    def get_past_mean(self) -> Array:
        """Return \hat{x}_{k-1|k-1}."""
        return self.past_mean

# ----------------------------- EDH Flow PF --------------------------------

class EDHFlowPF:
    """EKF/UKF-assisted EDH particle-flow particle filter."""

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

    # ----------------------------- API ----------------------------------

    def init_from_gaussian(self, mean0: Array, cov0: Array) -> PFState:
        """Sample initial particles from N(mean0, cov0) with equal weights."""
        n, nx = self.cfg.n_particles, mean0.size
        eps = self.cfg.rng.multivariate_normal(np.zeros(nx), cov0, size=n)
        particles = mean0[None, :] + eps
        weights = np.full(n, 1.0 / n)
        mean, cov = self._weighted_stats(particles, weights)
        return PFState(particles=particles, weights=weights, mean=mean, cov=cov, diagnostics={})

    def step(
        self,
        state: PFState,
        z_k: Array,
        u_km1: Optional[Array] = None,
        process_noise_sampler: Optional[Callable[[int, int], Array]] = None,
    ) -> PFState:
        """
        Run one EDH-PF step following the pseudocode.
        
        Steps:
        1. EKF/UKF prediction: (\hat{x}_{k-1|k-1}, P_{k-1|k-1}) -> (m_{k|k-1}, P_{k|k-1})
        2. Propagate particles: \eta_0^i = g(x_{k-1}^i, v)
        3. Flow update in pseudo-time \lambda \in [0,1]
        4. Weight update with ratio: w \propto w_{k-1} \cdot p(x_k|x_{k-1}) \cdot p(z_k|x_k) / p(\eta_0|x_{k-1})
        5. EKF/UKF measurement update: (m_{k|k-1}, P_{k|k-1}) -> (m_{k|k}, P_{k|k})
        6. Optional resampling
        """
        N, nx = state.particles.shape

        # --- EKF/UKF prediction: (m_{k|k-1}, P_{k|k-1}) ---
        m_pred, P = self.tracker.predict()
        # Enforce symmetry on P
        P = 0.5 * (P + P.T)

        # --- Propagate particles to η_0^i = g(x_{k-1}^i, v) ---
        if process_noise_sampler is None:
            # Default: zero process noise (caller should provide proper sampler)
            v = np.zeros((N, nx))
        else:
            v = process_noise_sampler(N, nx)
        
        eta0 = np.empty_like(state.particles)
        for i in range(N):
            eta0[i] = self.g(state.particles[i], u_km1, v[i])

        # --- Initialize flow states η_1^i <- η_0^i, \bar{\eta} <- \bar{\eta}_0 ---
        eta = eta0.copy()  # η_1^i
        # Compute mean trajectory initialization: \bar{\eta}_0 = g_k(\hat{x}_{k-1}, 0)
        etabar = self.g(self.tracker.get_past_mean(), u_km1, np.zeros(nx))
        
        # --- Flow update in pseudo-time \lambda \in [0,1] ---
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
            
            # Use solves for numerical stability (avoid explicit inverses)
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

        # Posterior correction (weight update) ---
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

        # --- EKF/UKF measurement update (tracker only) ---
        self.tracker.update(z_k)

        # --- Optional resampling based on ESS ---
        if self.cfg.resample_ess_ratio > 0.0:
            ess = effective_sample_size(w)
            if ess < self.cfg.resample_ess_ratio * N:
                idx = systematic_resample(w, self.cfg.rng)
                xk = xk[idx]
                w = np.full_like(w, 1.0 / N)

        # --- Estimate mean and covariance ---
        mean, cov = self._weighted_stats(xk, w)
        
        # Package diagnostics
        diagnostics = {'condition_numbers': cond_numbers}
        
        return PFState(particles=xk, weights=w, mean=mean, cov=cov, diagnostics=diagnostics)

    # ----------------------------- helpers --------------------------------

    @staticmethod
    def _weighted_stats(x: Array, w: Array) -> Tuple[Array, Array]:
        """Weighted mean/covariance with symmetry enforcement."""
        w = w / np.sum(w)
        mean = np.sum(x * w[:, None], axis=0)
        xc = x - mean[None, :]
        cov = (xc.T * w) @ xc
        cov = 0.5 * (cov + cov.T)
        return mean, cov

