import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

Array = np.ndarray


# -----------------------------
# Linear-Gaussian model helpers
# -----------------------------
@dataclass
class LinearGaussianBayes:
    """Linear-Gaussian Bayesian model for filtering.

    Represents a linear-Gaussian system with:
    - Prior: x ~ N(m0, P0)
    - Likelihood: z | x ~ N(H x, R)

    Attributes:
        m0 (Array): Prior mean vector of shape (n,).
        P0 (Array): Prior covariance matrix of shape (n, n), symmetric positive definite.
        H (Array): Observation matrix of shape (d, n).
        R (Array): Observation noise covariance of shape (d, d), symmetric positive definite.
        z (Array): Observation vector of shape (d,).
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

        Args:
            x (Array): State vector of shape (n,).

        Returns:
            Array: Gradient ∇ log p0 = -P0^{-1} (x - m0).
        """
        return -self.P0_inv @ (x - self.m0)

    def grad_log_h(self, x: Array) -> Array:
        """Compute gradient of log likelihood.

        Args:
            x (Array): State vector of shape (n,).

        Returns:
            Array: Gradient ∇ log h = H^T R^{-1} (z - Hx).
        """
        return self.H.T @ (self.R_inv @ (self.z - self.H @ x))

    def kalman_posterior(self) -> Tuple[Array, Array]:
        """Compute analytic Kalman posterior mean and covariance.

        Returns:
            Tuple[Array, Array]: Posterior mean m_post and covariance P_post.
        """
        S = self.H @ self.P0 @ self.H.T + self.R
        K = self.P0 @ self.H.T @ np.linalg.solve(S, np.eye(self.d))
        m_post = self.m0 + K @ (self.z - self.H @ self.m0)
        P_post = (np.eye(self.n) - K @ self.H) @ self.P0
        P_post = 0.5 * (P_post + P_post.T)
        return m_post, P_post


# ---------------------------------------
# Condition number kappa_2(M) and its derivative
# ---------------------------------------
def kappa2_and_derivative(M: Array, dM_dbeta: Array, eps: float = 1e-12) -> Tuple[float, float]:
    """Compute spectral condition number and its derivative.

    Computes kappa2(M) = lambda_max/lambda_min for symmetric positive definite M,
    and its derivative with respect to beta:
    dkappa/dbeta = (dlambda_max/dbeta)/lambda_min - lambda_max/lambda_min^2 * (dlambda_min/dbeta),
    where dlambda/dbeta = v^T (dM/dbeta) v for symmetric matrices.

    Args:
        M (Array): Symmetric positive definite matrix of shape (n, n).
        dM_dbeta (Array): Derivative of M with respect to beta, shape (n, n).
        eps (float, optional): Regularization parameter for numerical stability. Defaults to 1e-12.

    Returns:
        Tuple[float, float]: Condition number kappa and its derivative dkappa/dbeta.
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


# --------------------------
# Solve optimal beta_(lambda)
# --------------------------
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
    """Solve optimal beta schedule using shooting method and bisection.

    Solves the ODE beta'' = mu * dkappa/dbeta with boundary conditions:
    beta(0) = 0, beta'(0) = s (unknown), target beta(1) = 1.

    The ODE is autonomous in beta for the linear-Gaussian case because
    M(beta) = M0 + beta*Mh, so dkappa/dbeta depends only on beta.

    Args:
        M0 (Array): Initial precision matrix of shape (n, n).
        Mh (Array): Likelihood contribution to precision, shape (n, n).
        mu (float): Regularization parameter controlling path smoothness.
        n_grid (int, optional): Number of grid points. Defaults to 501.
        s_lo (float, optional): Lower bound for shooting parameter. Defaults to -5.0.
        s_hi (float, optional): Upper bound for shooting parameter. Defaults to 5.0.
        max_bracket_expand (int, optional): Maximum bracket expansions. Defaults to 30.
        max_bisect_iter (int, optional): Maximum bisection iterations. Defaults to 60.

    Returns:
        Tuple[Array, Array, Array]: Arrays of lambda grid, beta values, and beta derivatives.

    Raises:
        RuntimeError: If root bracketing fails.
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


# -------------------------
# Generalized SPF (normalized)
# -------------------------
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
    """Run generalized Stochastic Particle Filter with normalized homotopy.

    Implements normalized homotopy with alpha + beta = 1 (alpha = 1 - beta),
    using the theorem-based drift:
        f = K1 ∇log p + K2 ∇log h
    where:
        S = ∇^2 log p = ∇^2 log p0 + beta ∇^2 log h
        K2 = -beta' S^{-1}  (under normalization alpha + beta = 1)
        K1 = 1/2 Q + (beta'/2) S^{-1} (∇^2 log h) S^{-1}  (because alpha' + beta' = 0)

    Args:
        model (LinearGaussianBayes): Linear-Gaussian model specification.
        N (int, optional): Number of particles. Defaults to 2000.
        n_steps (int, optional): Number of time steps for homotopy. Defaults to 300.
        beta_mode (Literal["linear", "optimal"], optional): Beta schedule mode.
            Defaults to "optimal".
        mu (float, optional): Smoothness parameter for optimal beta. Defaults to 1e-2.
        Q_mode (Literal["scaled_identity", "inv_M"], optional): Diffusion covariance mode.
            Defaults to "inv_M".
        q_scale (float, optional): Scale for scaled_identity mode. Defaults to 1e-2.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        Tuple[Array, Array, dict]: Final particle positions (N, n), mean estimate (n,),
            and info dict containing lambda, beta, and betadot grids.
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



