"""
Stochastic Particle Filter (SPF) with generalised homotopy methods.

Implements the normalised stochastic particle filter using a homotopy path
that bridges the prior and posterior distributions via a beta(λ) schedule.
Two beta schedules are provided: ``'linear'`` (β = λ) and ``'optimal'``
(found by solving a shooting boundary-value problem to minimise the spectral
condition number of the precision matrix along the path).

Classes
-------
LinearGaussianBayes – Helper dataclass that represents a single
                      linear-Gaussian assimilation problem and pre-computes
                      required Hessians and precision matrices.

Functions
---------
kappa2_and_derivative    – Spectral condition number and its β-derivative.
solve_beta_star_bisection – Shooting + bisection solver for the optimal β(λ).
run_generalized_spf       – Main SPF driver.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

Array = np.ndarray


# Linear-Gaussian model helpers
@dataclass
class LinearGaussianBayes:
    """Linear-Gaussian Bayesian model for a single assimilation step.

    Represents the static linear-Gaussian problem::

        prior:      x ~ N(m0, P0)
        likelihood: z | x ~ N(H x, R)

    Pre-computes precision matrices and Hessians needed by the SPF.

    Parameters
    ----------
    m0 : ndarray, shape (n,)
        Prior mean vector.
    P0 : ndarray, shape (n, n)
        Prior covariance, symmetric positive definite.
    H : ndarray, shape (d, n)
        Observation matrix.
    R : ndarray, shape (d, d)
        Observation-noise covariance, symmetric positive definite.
    z : ndarray, shape (d,)
        Observation vector.

    Notes
    -----
    Assumes ``P0`` is symmetric positive definite, ``R`` is symmetric positive
    definite, and all dimensions are mutually consistent.
    """
    m0: Array          # (n,)
    P0: Array          # (n,n) SPD
    H: Array           # (d,n)
    R: Array           # (d,d) SPD
    z: Array           # (d,)

    def __post_init__(self):
        self.m0 = np.asarray(self.m0).reshape(-1)
        self.z = np.asarray(self.z).reshape(-1)
        self.P0 = np.asarray(self.P0)
        self.H = np.asarray(self.H)
        self.R = np.asarray(self.R)

        self.n = self.m0.size
        self.d = self.z.size
        assert self.P0.shape == (self.n, self.n)
        assert self.H.shape == (self.d, self.n)
        assert self.R.shape == (self.d, self.d)

        # Precompute precision-like matrices
        self.P0_inv = np.linalg.solve(self.P0, np.eye(self.n))
        self.R_inv = np.linalg.solve(self.R, np.eye(self.d))

        # Hessians of log densities (constant for linear-Gaussian)
        # log p0(x) = -1/2 (x-m0)^T P0^{-1} (x-m0) + const
        # ∇^2 log p0 = -P0^{-1}
        self.Hess_log_p0 = -self.P0_inv

        # log h(x) = -1/2 (z-Hx)^T R^{-1} (z-Hx) + const
        # ∇ log h = H^T R^{-1} (z - Hx)
        # ∇^2 log h = -H^T R^{-1} H
        self.Hess_log_h = -(self.H.T @ self.R_inv @ self.H)

        # For stiffness/conditioning control: M = -∇^2 log p = M0 + beta*Mh
        self.M0 = -self.Hess_log_p0                 # = P0^{-1} (SPD)
        self.Mh = -self.Hess_log_h                  # = H^T R^{-1} H (PSD)

        # Symmetrize for safety
        self.Hess_log_p0 = 0.5 * (self.Hess_log_p0 + self.Hess_log_p0.T)
        self.Hess_log_h = 0.5 * (self.Hess_log_h + self.Hess_log_h.T)
        self.M0 = 0.5 * (self.M0 + self.M0.T)
        self.Mh = 0.5 * (self.Mh + self.Mh.T)

    def grad_log_p0(self, x: Array) -> Array:
        """Compute gradient of log prior density.

        Parameters
        ----------
        x : ndarray, shape (n,)
            State vector.

        Returns
        -------
        ndarray, shape (n,)
            Gradient ``∇ log p0(x) = -P0⁻¹ (x − m0)``.

        Notes
        -----
        Assumes ``x`` has the same dimension as ``m0``.
        """
        return -self.P0_inv @ (x - self.m0)

    def grad_log_h(self, x: Array) -> Array:
        """Compute gradient of log likelihood.

        Parameters
        ----------
        x : ndarray, shape (n,)
            State vector.

        Returns
        -------
        ndarray, shape (n,)
            Gradient ``∇ log h(x) = Hᵀ R⁻¹ (z − Hx)``.

        Notes
        -----
        Assumes ``x`` has the same dimension as the prior state.
        """
        return self.H.T @ (self.R_inv @ (self.z - self.H @ x))

    def kalman_posterior(self) -> Tuple[Array, Array]:
        """Compute analytic Kalman posterior mean and covariance.

        Returns
        -------
        m_post : ndarray, shape (n,)
            Posterior mean.
        P_post : ndarray, shape (n, n)
            Posterior covariance, symmetrised.

        Notes
        -----
        Returns the exact linear-Gaussian posterior implied by the stored
        model specification.
        """
        S = self.H @ self.P0 @ self.H.T + self.R
        K = self.P0 @ self.H.T @ np.linalg.solve(S, np.eye(self.d))
        m_post = self.m0 + K @ (self.z - self.H @ self.m0)
        P_post = (np.eye(self.n) - K @ self.H) @ self.P0
        P_post = 0.5 * (P_post + P_post.T)
        return m_post, P_post



# Condition number kappa_2(M) and its derivative
def kappa2_and_derivative(M: Array, dM_dbeta: Array, eps: float = 1e-12) -> Tuple[float, float]:
    """Compute spectral condition number and its β-derivative.

    For a symmetric positive-definite matrix ``M(β) = M0 + β Mh``, returns
    the spectral condition number κ₂(M) = λ_max / λ_min and its derivative
    with respect to β via first-order perturbation theory.

    Parameters
    ----------
    M : ndarray, shape (n, n)
        Symmetric positive-definite matrix.
    dM_dbeta : ndarray, shape (n, n)
        Derivative of ``M`` with respect to β (typically equals ``Mh``).
    eps : float, optional
        Small regularisation added to ``M`` for numerical stability.
        Default is ``1e-12``.

    Returns
    -------
    kappa : float
        Spectral condition number κ₂(M) = λ_max / λ_min.
    dkappa : float
        Derivative dκ₂/dβ evaluated at the current β.

    Notes
    -----
    Assumes ``M`` is symmetric positive definite after optional regularisation
    by ``eps``.
    """
    # Ensure symmetry
    M = 0.5 * (M + M.T)
    dM_dbeta = 0.5 * (dM_dbeta + dM_dbeta.T)
    
    # Add small regularization for numerical stability
    n = M.shape[0]
    M_reg = M + eps * np.eye(n)
    
    # Eigen-decomposition with error handling
    try:
        w, V = np.linalg.eigh(M_reg)
    except np.linalg.LinAlgError:
        # Fallback: return a large condition number with zero derivative
        return 1e10, 0.0
    
    # Ensure positive eigenvalues
    lam_min = float(max(np.abs(w[0]), eps))
    lam_max = float(max(np.abs(w[-1]), eps))

    vmin = V[:, 0]
    vmax = V[:, -1]

    dlam_min = float(vmin.T @ dM_dbeta @ vmin)
    dlam_max = float(vmax.T @ dM_dbeta @ vmax)

    kappa = lam_max / lam_min
    dkappa = (dlam_max / lam_min) - (lam_max * dlam_min) / (lam_min ** 2)
    return kappa, dkappa


# Solve optimal beta_(lambda)
def solve_beta_star_bisection(
    M0: Array,
    Mh: Array,
    mu: float,
    n_grid: int = 501,
    s_lo: float = -5.0,
    s_hi: float = 5.0,
    max_bracket_expand: int = 30,
    max_bisect_iter: int = 60,
) -> Tuple[Array, Array, Array]:
    """Solve optimal β(λ) schedule via shooting and bisection.

    Finds the initial shooting velocity ``s*`` such that integrating the ODE

        β''(λ) = μ · (dκ₂/dβ)(β(λ))

    with boundary conditions ``β(0) = 0``, ``β(1) = 1`` is satisfied.
    The ODE is integrated with a 4th-order Runge-Kutta scheme.

    Parameters
    ----------
    M0 : ndarray, shape (n, n)
        Prior precision matrix (symmetric positive definite).
    Mh : ndarray, shape (n, n)
        Likelihood contribution to the precision matrix (symmetric PSD).
    mu : float
        Smoothness/regularisation weight for the condition-number gradient.
    n_grid : int, optional
        Number of uniformly spaced λ grid points in [0, 1].  Default is 501.
    s_lo : float, optional
        Initial lower bound on the shooting parameter ``s``.  Default is -5.0.
    s_hi : float, optional
        Initial upper bound on the shooting parameter ``s``.  Default is 5.0.
    max_bracket_expand : int, optional
        Maximum number of doublings applied to ``[s_lo, s_hi]`` before giving
        up on root bracketing.  Default is 30.
    max_bisect_iter : int, optional
        Maximum number of bisection iterations.  Default is 60.

    Returns
    -------
    lam : ndarray, shape (n_grid,)
        Uniformly spaced λ grid from 0 to 1.
    beta : ndarray, shape (n_grid,)
        Optimal β values on the grid, clipped to [0, 1].
    betadot : ndarray, shape (n_grid,)
        Derivative β'(λ) on the grid.

    Raises
    ------
    RuntimeError
        If root bracketing fails after ``max_bracket_expand`` doublings.

    Notes
    -----
    Assumes the shooting residual is continuous enough for bracketing and
    bisection to succeed over the expanded interval.
    """
    M0 = 0.5 * (M0 + M0.T)
    Mh = 0.5 * (Mh + Mh.T)

    dM_dbeta = Mh  # since M(beta) = M0 + beta*Mh

    lam = np.linspace(0.0, 1.0, n_grid)
    h = lam[1] - lam[0]

    def rhs(beta: float) -> float:
        """Compute right-hand side of ODE beta'' = mu * dkappa/dbeta."""
        # Clip beta to valid range to avoid numerical issues
        beta = np.clip(beta, -0.5, 1.5)
        M = M0 + beta * Mh
        _, dk = kappa2_and_derivative(M, dM_dbeta)
        return mu * dk

    def integrate(s0: float) -> Tuple[Array, Array]:
        """Integrate ODE using RK4 method with initial velocity s0."""
        beta = np.zeros(n_grid)
        betadot = np.zeros(n_grid)
        beta[0] = 0.0
        betadot[0] = s0

        # RK4 on y1'=y2, y2'=rhs(y1)
        for k in range(n_grid - 1):
            y1, y2 = beta[k], betadot[k]

            def f1(b, bd): return bd
            def f2(b, bd): return rhs(b)

            k11 = f1(y1, y2)
            k12 = f2(y1, y2)

            k21 = f1(y1 + 0.5*h*k11, y2 + 0.5*h*k12)
            k22 = f2(y1 + 0.5*h*k11, y2 + 0.5*h*k12)

            k31 = f1(y1 + 0.5*h*k21, y2 + 0.5*h*k22)
            k32 = f2(y1 + 0.5*h*k21, y2 + 0.5*h*k22)

            k41 = f1(y1 + h*k31, y2 + h*k32)
            k42 = f2(y1 + h*k31, y2 + h*k32)

            beta[k+1] = y1 + (h/6.0)*(k11 + 2*k21 + 2*k31 + k41)
            betadot[k+1] = y2 + (h/6.0)*(k12 + 2*k22 + 2*k32 + k42)

        return beta, betadot

    def F(s0: float) -> float:
        """Shooting function: compute residual beta(1) - 1."""
        beta, _ = integrate(s0)
        return float(beta[-1] - 1.0)

    # Bracket root
    f_lo = F(s_lo)
    f_hi = F(s_hi)
    expand = 0
    while np.sign(f_lo) == np.sign(f_hi) and expand < max_bracket_expand:
        # Expand symmetrically
        s_lo *= 2.0
        s_hi *= 2.0
        f_lo = F(s_lo)
        f_hi = F(s_hi)
        expand += 1

    if np.sign(f_lo) == np.sign(f_hi):
        raise RuntimeError("Failed to bracket beta(1)=1 shooting root. Try wider initial s_lo/s_hi.")

    # Bisection
    for _ in range(max_bisect_iter):
        s_mid = 0.5 * (s_lo + s_hi)
        f_mid = F(s_mid)
        if abs(f_mid) < 1e-10:
            s_lo, s_hi = s_mid, s_mid
            break
        if np.sign(f_mid) == np.sign(f_lo):
            s_lo, f_lo = s_mid, f_mid
        else:
            s_hi, f_hi = s_mid, f_mid

    s_star = 0.5 * (s_lo + s_hi)
    beta, betadot = integrate(s_star)

    # Safety post-processing: enforce endpoints exactly and mild clipping
    beta[0] = 0.0
    beta[-1] = 1.0
    beta = np.clip(beta, 0.0, 1.0)

    return lam, beta, betadot


# Generalized SPF (normalized)
def run_generalized_spf(
    model: LinearGaussianBayes,
    N: int = 2000,
    n_steps: int = 300,
    beta_mode: Literal["linear", "optimal"] = "optimal",
    mu: float = 1e-2,
    Q_mode: Literal["scaled_identity", "inv_M"] = "inv_M",
    q_scale: float = 1e-2,
    seed: int = 0,
) -> Tuple[Array, Array, dict]:
    """Run generalised Stochastic Particle Filter with normalised homotopy.

    Propagates ``N`` particles from the prior ``p(x) = N(m0, P0)`` to the
    posterior ``p(x | z)`` along the homotopy path

        log πλ(x)  ∝  log p(x) + β(λ) log h(x),   λ ∈ [0, 1]

    using an Euler–Maruyama discretisation of the Langevin SDE.  Two β(λ)
    schedules are supported: a simple linear ramp and the condition-number-
    minimising optimal schedule found by ``solve_beta_star_bisection``.

    Parameters
    ----------
    model : LinearGaussianBayes
        Fully initialised linear-Gaussian model specification.
    N : int, optional
        Number of particles.  Default is 2000.
    n_steps : int, optional
        Number of Euler–Maruyama steps along the homotopy.  Default is 300.
    beta_mode : {'linear', 'optimal'}, optional
        Beta schedule.  ``'linear'`` uses β(λ) = λ; ``'optimal'`` solves the
        shooting BVP to minimise the condition-number gradient.  Default is
        ``'optimal'``.
    mu : float, optional
        Smoothness weight passed to ``solve_beta_star_bisection`` when
        ``beta_mode='optimal'``.  Default is ``1e-2``.
    Q_mode : {'scaled_identity', 'inv_M'}, optional
        Diffusion covariance choice.  ``'scaled_identity'`` uses
        ``Q = q_scale² I``; ``'inv_M'`` uses ``Q = M(β)⁻¹`` which is SPD and
        satisfies the normalisation condition.  Default is ``'inv_M'``.
    q_scale : float, optional
        Scale factor used when ``Q_mode='scaled_identity'``.  Default is
        ``1e-2``.
    seed : int, optional
        Integer seed for ``numpy.random.default_rng``.  Default is 0.

    Returns
    -------
    X : ndarray, shape (N, n)
        Final particle positions approximating the posterior.
    x_hat : ndarray, shape (n,)
        Empirical mean of the final particles.
    info : dict
        Dictionary with keys ``'lam'``, ``'beta'``, ``'betadot'`` containing
        the λ grid and the corresponding β and β' arrays used during the run.

    Raises
    ------
    ValueError
        If ``beta_mode`` or ``Q_mode`` is not one of the accepted literals.

    Notes
    -----
    Assumes ``model`` has been prevalidated and that the chosen diffusion mode
    yields a numerically stable Cholesky factor at each homotopy step.
    """
    rng = np.random.default_rng(seed)

    n = model.n
    # Sample initial particles from prior
    L0 = np.linalg.cholesky(model.P0)
    X = model.m0[None, :] + rng.standard_normal((N, n)) @ L0.T

    # Build beta(λ) and beta'(λ)
    if beta_mode == "linear":
        lam_grid = np.linspace(0.0, 1.0, n_steps + 1)
        beta_grid = lam_grid.copy()
        betadot_grid = np.ones_like(lam_grid)
    elif beta_mode == "optimal":
        lam_grid, beta_grid, betadot_grid = solve_beta_star_bisection(
            model.M0, model.Mh, mu=mu, n_grid=n_steps + 1
        )
    else:
        raise ValueError("beta_mode must be 'linear' or 'optimal'.")

    dlam = float(lam_grid[1] - lam_grid[0])

    # Precompute constant Hessians
    H0 = model.Hess_log_p0
    Hh = model.Hess_log_h

    info = {"lam": lam_grid, "beta": beta_grid, "betadot": betadot_grid}

    # Euler–Maruyama in λ
    for k in range(n_steps):
        beta = float(beta_grid[k])
        beta_p = float(betadot_grid[k])

        # S = Hess(log p) = Hess(log p0) + beta * Hess(log h)
        S = H0 + beta * Hh
        S = 0.5 * (S + S.T)

        # Solve for S^{-1} stably via linear solve
        Sinv = np.linalg.solve(S, np.eye(n))

        # Choose Q(λ) (must not depend on x for the theorem)
        if Q_mode == "scaled_identity":
            Q = (q_scale ** 2) * np.eye(n)
        elif Q_mode == "inv_M":
            # M = -S, so inv(M) = inv(-S) = -(S^{-1})
            # ensures positive definite diffusion (since M SPD)
            M = -S
            Q = np.linalg.solve(M, np.eye(n))
        else:
            raise ValueError("Q_mode must be 'scaled_identity' or 'inv_M'.")

        Q = 0.5 * (Q + Q.T)
        # Factor Q for noise
        # (if Q is only PSD, cholesky may fail; inv_M is SPD here)
        LQ = np.linalg.cholesky(Q + 1e-12 * np.eye(n))

        # K matrices (normalized alpha+beta=1)
        K2 = -beta_p * Sinv
        K1 = 0.5 * Q + 0.5 * beta_p * (Sinv @ Hh @ Sinv)

        # Propagate all particles
        # ∇log h and ∇log p computed per particle
        Z = rng.standard_normal((N, n))

        for i in range(N):
            x = X[i]
            g_h = model.grad_log_h(x)
            g_p = model.grad_log_p0(x) + beta * g_h  # from ∇log p = ∇log p0 + beta ∇log h

            f = K1 @ g_p + K2 @ g_h
            noise = (np.sqrt(dlam) * (Z[i] @ LQ.T))
            X[i] = x + dlam * f + noise

    # Return particles + simple estimate
    x_hat = X.mean(axis=0)
    return X, x_hat, info



