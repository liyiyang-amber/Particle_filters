"""Kernel Particle Filter (KPF) implementation.

Implements a matrix-kernel particle flow filter that transports an ensemble
of particles from the prior to the posterior in pseudo-time s ∈ [0, 1] by
integrating the velocity field

    f_s(x) = B · (1/N ∑_m [ K(x, x_m) ∇ log p(x_m | y) + ∇_x · K(x, x_m) ])

where K is either a diagonal matrix-valued RBF kernel or an isotropic scalar
RBF kernel, and B is the (optionally localised) prior sample covariance.

Classes
-------
Model              – User-supplied observation model.
KPFConfig          – Hyper-parameters controlling the flow integration.
KPFState           – Output container returned by ``KernelParticleFilter.analyze``.
KernelParticleFilter – Main filter class.

Functions
---------
gaspari_cohn             – Gaspari–Cohn correlation taper.
build_localization_matrix – Construct a Gaspari–Cohn localisation matrix.
rbf_1d                    – 1-D RBF kernel and x-derivative.
scalar_kernel_full_matrix – Isotropic scalar kernel with gradient and divergence.
matrix_kernel_and_divergence – Diagonal matrix-valued kernel and divergence.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np

Array = np.ndarray


# Utility: Gaspari–Cohn localization
def gaspari_cohn(r: Array) -> Array:
    """Gaspari–Cohn correlation taper with compact support.
    
    Parameters
    ----------
    r : np.ndarray
        Nonnegative distances divided by a user-chosen radius c, so that r=1
        corresponds to the cutoff radius.
    
    Returns
    -------
    np.ndarray
        Correlation values in [0, 1], same shape as `r`.

    Notes
    -----
    Assumes ``r`` represents distances scaled by the localisation radius and
    is interpreted elementwise.
    """
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    # 0 <= r <= 1
    m1 = (r >= 0) & (r <= 1)
    rm = r[m1]
    out[m1] = (
        1
        - 5 * rm**2 / 3
        + 5 * rm**3 / 8
        + rm**4 / 2
        - rm**5 / 4
    )
    # 1 < r <= 2
    m2 = (r > 1) & (r <= 2)
    rm = r[m2]
    out[m2] = (
        4
        - 5 * rm
        + 5 * rm**2 / 3
        + 5 * rm**3 / 8
        - rm**4 / 2
        + rm**5 / 12
        - 2 / (3 * rm)
    )
    # r > 2 => 0 
    return out


def build_localization_matrix(n: int, radius: float, metric: Optional[Array] = None) -> Array:
    """Build a Gaspari–Cohn localization matrix L (n x n).
    
    Parameters
    ----------
    n : int
        State dimension.
    radius : float
        Cutoff radius c (in the metric units). Use np.inf to disable localization.
    metric : Optional[np.ndarray], optional
        Pairwise distances matrix (n x n). If None, assumes 1D chain with
        distance |i-j|.
    
    Returns
    -------
    np.ndarray
        Localization matrix L with entries in [0,1].

    Notes
    -----
    Assumes ``metric`` is a symmetric pairwise-distance matrix when provided.
    Passing ``np.inf`` disables localisation and returns an all-ones matrix.
    """
    if np.isinf(radius):
        return np.ones((n, n))
    if metric is None:
        idx = np.arange(n)
        D = np.abs(idx[:, None] - idx[None, :])
    else:
        D = np.asarray(metric, dtype=float)
        if D.shape != (n, n):
            raise ValueError("metric must be (n, n).")
    r = D / float(radius)
    return gaspari_cohn(r)



# Kernel: diagonal matrix-valued RBF and its divergence term
def rbf_1d(d: Array, ell: float) -> Tuple[Array, Array]:
    """1D RBF kernel and its derivative wrt first argument.
    
    K(d) = exp(-0.5 * (d/ell)^2), d = x_j - z_j
    
    Parameters
    ----------
    d : np.ndarray
        Pairwise 1D differences.
    ell : float
        Lengthscale (>0).
    
    Returns
    -------
    (K, dKdx) : Tuple[np.ndarray, np.ndarray]
        K values and derivative wrt x (not z): dK/dx = -(d/ell^2) * K

    Notes
    -----
    Assumes ``ell`` is strictly positive.
    """
    s2 = (d / ell) ** 2
    K = np.exp(-0.5 * s2)
    dKdx = -(d / (ell**2)) * K
    return K, dKdx


def scalar_kernel_full_matrix(
    x: Array,
    ensemble: Array,
    lengthscale: float,
) -> Tuple[Array, Array, Array]:
    """Compute full-matrix scalar kernel K(x, x_m) and div_x K(x, x_m).
    
    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Query point.
    ensemble : np.ndarray, shape (Np, n)
        Particle locations {x_m}.
    lengthscale : float
        Isotropic RBF lengthscale (positive).
    
    Returns
    -------
    (K_full, grad_k, divK) : Tuple[np.ndarray, np.ndarray, np.ndarray]
        K_full : (Np,) scalar kernel values k(||x-x_m||) for each particle
        grad_k : (Np, n) gradient ∇k(r) for each particle
        divK   : (n,) divergence vector summed over all particles
    """
    x = np.asarray(x, dtype=float)
    X = np.asarray(ensemble, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be (n,)")
    if X.ndim != 2 or X.shape[1] != x.size:
        raise ValueError("ensemble must be (Np, n) matching x size.")
    
    n = x.size
    Np = X.shape[0]
    
    # Compute Euclidean distances
    D = x[None, :] - X  # (Np, n) differences
    r_sq = np.sum(D**2, axis=1)  # (Np,) squared distances
    
    # Scalar kernel value for each particle
    k = np.exp(-0.5 * r_sq / (lengthscale**2))  # (Np,)
    
    # Gradient of k(r) wrt x: ∇k = -k/ell^2 * (x-z)
    grad_k = -(k[:, None] / (lengthscale**2)) * D  # (Np, n)
    
    # Divergence: for full matrix K with all entries = k(r)
    # (∇·K)_i = Σ_j \partial k/\parital x_j = sum of gradient components
    # Sum over particles
    divK = np.zeros(n)
    for m in range(Np):
        divK += np.sum(grad_k[m])  # scalar: sum all components of gradient
    
    return k, grad_k, divK


def matrix_kernel_and_divergence(
    x: Array,
    ensemble: Array,
    lengthscales: Array,
) -> Tuple[Array, Array]:
    """Compute diagonal matrix-valued kernel K(x, x_m) and div_x K(x, x_m).
    
    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Query point.
    ensemble : np.ndarray, shape (Np, n)
        Particle locations {x_m}.
    lengthscales : np.ndarray, shape (n,)
        Per-component RBF lengthscales (positive).
    
    Returns
    -------
    (K_blocks, divK) : Tuple[np.ndarray, np.ndarray]
        K_blocks : (Np, n) holding diagonal entries for each particle m and dim j.
                   For each m, the diagonal matrix K(x, x_m) is represented by K_blocks[m, :].
        divK     : (n,) divergence vector at x accumulated over z (single z not summed).
                   Here we return per-z values; callers will sum over particles with weights.
    """
    x = np.asarray(x, dtype=float)
    X = np.asarray(ensemble, dtype=float)
    ell = np.asarray(lengthscales, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be (n,)")
    if X.ndim != 2 or X.shape[1] != x.size:
        raise ValueError("ensemble must be (Np, n) matching x size.")
    if ell.shape != (x.size,):
        raise ValueError("lengthscales must be shape (n,)")
    # Differences per particle and dimension
    D = (x[None, :] - X)  # (Np, n)
    # Compute per-dimension kernels and derivatives
    K_blocks = np.empty_like(D)
    divK = np.zeros(x.shape)
    for j in range(x.size):
        Kj, dKjdx = rbf_1d(D[:, j], ell[j])  # (Np,), (Np,)
        K_blocks[:, j] = Kj
        divK[j] = dKjdx.sum()  # divergence sums derivatives for each z (particle)
    return K_blocks, divK


# Model and configuration
ObsFn = Callable[[Array], Array]
JacFn = Callable[[Array], Array]

@dataclass
class Model:
    """User model with observation function and its Jacobian.
    
    Parameters
    ----------
    H : Callable[[np.ndarray], np.ndarray]
        Observation function: y = H(x). Input shape (n,), output shape (m,).
    JH : Callable[[np.ndarray], np.ndarray]
        Jacobian of H at x: JH(x) with shape (m, n).
    R : np.ndarray
        Observation noise covariance (m, m), positive definite.

    Notes
    -----
    Assumes ``H`` and ``JH`` are consistent with one another and operate on
    state vectors of the same dimension expected by the filter.
    """

    H: ObsFn
    JH: JacFn
    R: Array


@dataclass
class KPFConfig:
    """Hyper-parameters controlling the kernel particle flow integration.

    Attributes
    ----------
    ds_init : float
        Initial pseudo-time step size.  Default is ``0.2``.
    ds_min : float
        Minimum allowed pseudo-time step size.  Default is ``1e-3``.
    c_move_max : float
        Maximum allowed per-particle move measured in Mahalanobis norm per
        step.  Larger moves are scaled down.  Default is ``2.0``.
    min_steps : int
        Minimum number of pseudo-time steps regardless of convergence.
        Default is ``5``.
    max_steps : int
        Hard cap on the total number of pseudo-time steps.  Default is ``100``.
    kernel_type : str
        Kernel flavour.  ``'diagonal'`` uses a diagonal matrix-valued RBF;
        ``'scalar'`` uses an isotropic scalar RBF.  Default is ``'diagonal'``.
    lengthscale_mode : str
        How to choose RBF lengthscales.  ``'std'`` sets each dimension's
        lengthscale to the ensemble standard deviation; ``'fixed'`` uses
        ``fixed_lengthscale``.  Default is ``'std'``.
    fixed_lengthscale : float
        Lengthscale used when ``lengthscale_mode='fixed'``.  Default is ``1.0``.
    reg : float
        Tikhonov regularisation added to the sample covariance before
        inversion.  Default is ``1e-6``.
    localization_radius : float
        Gaspari–Cohn cutoff radius in state-space units.  ``np.inf`` disables
        localisation.  Default is ``np.inf``.
    random_order : bool
        If ``True``, shuffle the particle evaluation order at each pseudo-time
        step.  Default is ``True``.
    """
    ds_init: float = 0.2          # initial pseudo-time step
    ds_min: float = 1e-3          # minimum step size
    c_move_max: float = 2.0       # max allowed move in Mahalanobis norm per step
    min_steps: int = 5            # minimum pseudo-time steps
    max_steps: int = 100          # cap on pseudo-time steps
    kernel_type: str = "diagonal" # "diagonal" | "scalar" (diagonal=matrix-valued, scalar=isotropic)
    lengthscale_mode: str = "std" # "std" | "fixed"
    fixed_lengthscale: float = 1.0
    reg: float = 1e-6             # Tikhonov regularization for inverses
    localization_radius: float = np.inf  # Gaspari–Cohn radius; inf disables
    random_order: bool = True     # Shuffle particle evaluation order per step


@dataclass
class KPFState:
    """Output state returned by ``KernelParticleFilter.analyze``.

    Attributes
    ----------
    particles : ndarray, shape (Np, n)
        Posterior particle ensemble after the flow.
    weights : ndarray, shape (Np,)
        Particle weights (uniform, i.e. ``1/Np``, after the flow).
    s : float
        Final pseudo-time reached (should be ≈ 1.0 on successful completion).
    steps : int
        Number of pseudo-time integration steps taken.
    ds_history : list of float, optional
        Adaptive step sizes used at each pseudo-time step (for diagnostics).
    """
    particles: Array  # (Np, n)
    weights: Array    # (Np,) 
    s: float          # current pseudo-time in [0,1]
    steps: int        # number of pseudo-time steps taken
    ds_history: list = None  # adaptive step sizes (theta) for diagnostics


# Kernel Particle Filter
class KernelParticleFilter:
    """Matrix-kernel Particle Flow Filter.

    Transports an ensemble of ``Np`` particles from a prior distribution to
    an approximate posterior by integrating a deterministic velocity field in
    pseudo-time s ∈ [0, 1].  The velocity field uses a (diagonal or scalar)
    RBF kernel and the prior sample covariance to define the flow direction,
    optionally with Gaspari–Cohn spatial localisation.

    Parameters
    ----------
    model : Model
        User-supplied observation model containing ``H``, ``JH``, and ``R``.
    config : KPFConfig, optional
        Filter hyper-parameters.  Defaults to ``KPFConfig()`` if not supplied.
    """

    def __init__(self, model: Model, config: Optional[KPFConfig] = None):
        self.model = model
        self.cfg = config or KPFConfig()

    # helpers 
    @staticmethod
    def mean_and_cov(X: Array, reg: float = 0.0) -> Tuple[Array, Array]:
        """Compute ensemble mean and sample covariance with optional regularisation.

        Parameters
        ----------
        X : ndarray, shape (Np, n)
            Ensemble matrix.
        reg : float, optional
            Ridge regularisation added to the diagonal.  Default is ``0.0``.

        Returns
        -------
        mu : ndarray, shape (n,)
            Ensemble mean.
        B : ndarray, shape (n, n)
            Sample covariance (optionally regularised).
        """
        mu = X.mean(axis=0)
        A = X - mu
        B = (A.T @ A) / max(1, X.shape[0] - 1)
        if reg > 0:
            B = B + reg * np.eye(B.shape[0])
        return mu, B

    @staticmethod
    def schur(A: Array, B: Array) -> Array:
        """Return the elementwise (Hadamard) product of two arrays.

        Parameters
        ----------
        A : ndarray
            First array.
        B : ndarray
            Second array, same shape as ``A``.

        Returns
        -------
        ndarray
            Elementwise product ``A ⊙ B``.
        """
        return np.multiply(A, B)

    def _prior_stats(self, X: Array) -> Tuple[Array, Array]:
        """Compute prior mean and localised sample covariance.

        Parameters
        ----------
        X : ndarray, shape (Np, n)
            Current particle ensemble.

        Returns
        -------
        x0 : ndarray, shape (n,)
            Ensemble mean.
        B_loc : ndarray, shape (n, n)
            Gaspari–Cohn-localised sample covariance (regularised).
        """
        x0, B = self.mean_and_cov(X, reg=self.cfg.reg)
        n = B.shape[0]
        L = build_localization_matrix(n, self.cfg.localization_radius)
        B_loc = self.schur(B, L)
        return x0, B_loc

    def _lengthscales(self, X: Array, mode: str) -> Array:
        """Choose per-dimension RBF lengthscales for the kernel.

        Parameters
        ----------
        X : ndarray, shape (Np, n)
            Current particle ensemble.
        mode : str
            ``'fixed'`` returns ``cfg.fixed_lengthscale`` for every dimension;
            ``'std'`` uses the per-dimension ensemble standard deviation
            (with a small floor of ``1e-12``).

        Returns
        -------
        ndarray, shape (n,)
            Per-dimension lengthscales.
        """
        if mode == "fixed":
            return np.full(X.shape[1], self.cfg.fixed_lengthscale, dtype=float)
        # std mode: per-dimension ensemble std with floor
        std = X.std(axis=0) + 1e-12
        return std

    def _score(self, x: Array, x0: Array, B_inv: Array, y: Array) -> Array:
        """Compute the log-posterior gradient ∇ log p(x | y).

        Uses a Gaussian prior ``p(x) = N(x0, B)`` and a nonlinear likelihood
        with observation function ``H`` and noise covariance ``R``::

            ∇ log p(x | y) = JH(x)ᵀ R⁻¹ (y − H(x)) − B⁻¹ (x − x0)

        Parameters
        ----------
        x : ndarray, shape (n,)
            State at which to evaluate the score.
        x0 : ndarray, shape (n,)
            Prior mean.
        B_inv : ndarray, shape (n, n)
            Inverse prior covariance.
        y : ndarray, shape (m,)
            Observation vector.

        Returns
        -------
        ndarray, shape (n,)
            Score vector ∇ log p(x | y).
        """
        Hx = self.model.H(x)              # (m,)
        J = self.model.JH(x)              # (m, n)
        # Solve R^{-1} (y - Hx)
        R = self.model.R
        r = y - Hx
        # Use solve for numerical stability
        z = np.linalg.solve(R, r)
        g = J.T @ z - (B_inv @ (x - x0))
        return g

    def _mahalanobis(self, dx: Array, B_inv: Array) -> float:
        """Return the Mahalanobis norm ``sqrt(dxᵀ B⁻¹ dx)``.

        Parameters
        ----------
        dx : ndarray, shape (n,)
            Displacement vector.
        B_inv : ndarray, shape (n, n)
            Inverse covariance matrix.

        Returns
        -------
        float
            Mahalanobis norm of ``dx``.
        """
        return float(np.sqrt(dx @ (B_inv @ dx)))


    def analyze(
        self, 
        X: Array, 
        y: Array, 
        lengthscales: Optional[Array] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> KPFState:
        """Run the particle flow from prior ensemble X to posterior ensemble.
        
        Parameters
        ----------
        X : np.ndarray, shape (Np, n)
            Prior ensemble (particles).
        y : np.ndarray, shape (m,)
            Observation.
        lengthscales : Optional[np.ndarray], shape (n,), optional
            Per-dimension RBF lengthscales. If None, chosen from ensemble std.
        rng : Optional[np.random.Generator], optional
            RNG for shuffling evaluation order.
        
        Returns
        -------
        KPFState
            Final particle set and diagnostics.
        """
        X = np.asarray(X, dtype=float).copy()
        y = np.asarray(y, dtype=float)
        Np, n = X.shape
        x0, B = self._prior_stats(X)
        B_inv = np.linalg.inv(B + self.cfg.reg * np.eye(n))
        
        # Determine kernel type and lengthscale
        cfg = self.cfg
        use_scalar_kernel = (cfg.kernel_type == "scalar")
        
        if use_scalar_kernel:
            # Scalar kernel uses a single lengthscale
            if lengthscales is not None:
                ell_scalar = float(lengthscales) if np.isscalar(lengthscales) else float(lengthscales[0])
            elif cfg.lengthscale_mode == "fixed":
                ell_scalar = cfg.fixed_lengthscale
            else:  # "std" mode - use mean of ensemble std
                ell_scalar = float(X.std(axis=0).mean())
            ell = ell_scalar  # store for compatibility
        else:
            # Diagonal/matrix kernel uses per-dimension lengthscales
            ell = lengthscales if lengthscales is not None else self._lengthscales(X, cfg.lengthscale_mode)
        
        s = 0.0
        steps = 0
        ds = cfg.ds_init
        weights = np.full(Np, 1.0 / Np)  
        
        # Track adaptive step sizes for diagnostics
        ds_history = []

        # Precompute scores at particle positions for efficiency 
        def compute_scores(Xp: Array) -> Array:
            G = np.empty_like(Xp)
            for i in range(Np):
                G[i] = self._score(Xp[i], x0, B_inv, y)
            return G

        G = compute_scores(X)

        # Pseudo-time integration loop
        while (s < 1.0 and steps < cfg.max_steps) or (steps < cfg.min_steps):
            steps += 1
            # Adjust last step to hit s=1
            if s + ds > 1.0:
                ds = 1.0 - s
            
            # Track step size
            ds_history.append(float(ds))

            order = np.arange(Np)
            if cfg.random_order:
                (rng or np.random.default_rng()).shuffle(order)

            X_new = X.copy()
            # Evaluate velocity f_s(x_i) at each particle using current scores
            for idx in order:
                xi = X[idx]
                # Kernel blocks and divergence wrt xi with all ensemble
                if use_scalar_kernel:
                    # Full-matrix scalar kernel: K has all entries = k(r)
                    k_vals, grad_k, divK = scalar_kernel_full_matrix(xi, X, ell_scalar)  # (Np,), (Np,n), (n,)
                    # For full matrix K with all entries k(r):
                    # (K @ G)_i = k(r) * Σ_j G_j  (sum over all dimensions for each particle)
                    # Average over particles: (1/N) Σ_m k_m * (Σ_j G_m,j) * e_i
                    # where e_i is i-th unit vector, but since K is full, all dims get same contribution
                    # Actually: (K @ G) = k * (1^T @ G) where 1 is all-ones vector
                    # So each component i gets: k * sum(G)
                    G_sum_per_particle = G.sum(axis=1)  # (Np,) - sum over dimensions for each particle
                    term1 = (k_vals * G_sum_per_particle).mean() * np.ones(n)  # broadcast to all dims
                    term2 = divK / float(Np)  # (n,)
                else:
                    K_blocks, divK = matrix_kernel_and_divergence(xi, X, ell)  # (Np,n), (n,)
                    # Diagonal kernel: K * G is elementwise per-dim product
                    term1 = (K_blocks * G).mean(axis=0)  # (n,)
                    term2 = divK / float(Np)  # (n,)
                
                v = B @ (term1 + term2)  # (n,)  <- Using B as per algorithm
                # Adaptive step to control Mahalanobis move
                move = self._mahalanobis(ds * v, B_inv)
                if move > cfg.c_move_max:
                    # reduce step for this particle; keep global ds for others
                    scale = cfg.c_move_max / max(move, 1e-12)
                    X_new[idx] = xi + (ds * scale) * v
                else:
                    X_new[idx] = xi + ds * v

            # Accept step; update s and recompute scores
            X = X_new
            s += ds
            # Optional backtracking if step was too aggressive globally (rare if per-particle guard works)
            # Here we simply recompute scores for next iteration.
            G = compute_scores(X)

            # Heuristic: if little movement and near s=1, break
            if ds <= cfg.ds_min and (1.0 - s) < 1e-6:
                break

        return KPFState(particles=X, weights=weights, s=s, steps=steps, ds_history=ds_history)