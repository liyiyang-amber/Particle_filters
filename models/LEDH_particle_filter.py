"""
EKF/UKF-assisted Local-Exact Daum–Huang (LEDH) Particle-Flow PF.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple
import numpy as np

Array = np.ndarray

# Protocols
class GaussianTracker(Protocol):
    def predict(self) -> Tuple[Array, Array]: ...
    def update(self, z_k: Array) -> Tuple[Array, Array]: ...
    def get_past_mean(self) -> Array: ...

GFn = Callable[[Array, Optional[Array], Optional[Array]], Array]
HFn = Callable[[Array], Array]
JacobianHFn = Callable[[Array], Array]
LogTransPdf = Callable[[Array, Array], float]
LogLikePdf  = Callable[[Array, Array], float]

# Utilities
def systematic_resample(weights: Array, rng: np.random.Generator) -> Array:
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
    w = weights / np.sum(weights)
    return 1.0 / float(np.sum(w * w))

# Config/State
@dataclass
class LEDHConfig:
    n_particles: int = 512
    n_lambda_steps: int = 8
    resample_ess_ratio: float = 0.0
    rng: np.random.Generator = np.random.default_rng(0)

@dataclass
class PFState:
    particles: Array
    weights: Array
    mean: Array
    cov: Array
    diagnostics: dict = None  # optional flow diagnostics (e.g., condition numbers)

# LEDH Flow PF
class LEDHFlowPF:
    """EKF/UKF-assisted LEDH particle-flow particle filter (Algorithm 1)."""

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
        """Algorithm lines 1-2: Initialize particles from prior and set uniform weights."""
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
        """Run one LEDH step aligned with Algorithm 1 (per-particle LEDH)."""
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
        mean, cov = self._weighted_stats(xk, w)
        
        # Package diagnostics
        diagnostics = {'condition_numbers': cond_numbers}
        
        return PFState(particles=xk, weights=w, mean=mean, cov=cov, diagnostics=diagnostics)

    # helpers
    @staticmethod
    def _weighted_stats(x: Array, w: Array) -> Tuple[Array, Array]:
        w = w / np.sum(w)
        mean = np.sum(x * w[:, None], axis=0)
        xc = x - mean[None, :]
        cov = (xc.T * w) @ xc
        cov = 0.5 * (cov + cov.T)
        return mean, cov