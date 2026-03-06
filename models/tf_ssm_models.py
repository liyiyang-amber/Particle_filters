"""
TensorFlow / TensorFlow Probability state-space model definitions.

Provides ready-to-use SSM classes whose log-likelihoods are differentiable
with respect to model parameters (via ``tf.GradientTape``), together with
HMC/NUTS target builders and a convenience sampler for Bayesian parameter
estimation.

Classes
-------
LGSSM_TFP
    Linear Gaussian SSM with exact Kalman-filter log-likelihood.
NonlinSSM_TFP
    Scalar nonlinear SSM (SMC benchmark) with EKF-approximated
    log-likelihood.

Functions
---------
make_lgssm_hmc_target
    Build a TFP ``target_log_prob_fn`` for learning the process-noise
    covariance :math:`Q` of an LGSSM via HMC.
make_nonlinssm_hmc_target
    Build a TFP ``target_log_prob_fn`` for learning :math:`\\sigma_v` and
    :math:`\\sigma_w` of the nonlinear SSM via HMC.
run_hmc
    Run TFP HMC or NUTS and return posterior samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

try:
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    _HAS_TFP = True
except ImportError:
    _HAS_TFP = False

try:
    from models.tf_core import lgssm_log_likelihood, kalman_filter_tf, _to_f32, _symmetrise
except ModuleNotFoundError:
    from tf_core import lgssm_log_likelihood, kalman_filter_tf, _to_f32, _symmetrise

F32    = tf.float32
Tensor = tf.Tensor


# 1.  LGSSM_TFP
class LGSSM_TFP:
    """Linear Gaussian SSM with exact Kalman-filter log-likelihood.

    The model is:

    .. math::

        x_k &= F x_{k-1} + v_k, \\quad v_k \\sim \\mathcal{N}(0, Q) \\\\
        y_k &= H x_k + w_k,     \\quad w_k \\sim \\mathcal{N}(0, R)

    with initial distribution
    :math:`x_0 \\sim \\mathcal{N}(m_0, P_0)`.

    Parameters
    ----------
    F : ndarray, shape (nx, nx)
        State transition matrix.
    H : ndarray, shape (ny, nx)
        Observation matrix.
    Q : ndarray, shape (nx, nx)
        Process noise covariance.
    R : ndarray, shape (ny, ny)
        Observation noise covariance.
    m0 : ndarray, shape (nx,)
        Initial state mean.
    P0 : ndarray, shape (nx, nx)
        Initial state covariance.

    Notes
    -----
    Assumes ``F``, ``H``, ``Q``, ``R``, ``m0``, and ``P0`` are dimensionally
    compatible. Cholesky-based computations further assume ``Q``, ``R``, and
    ``P0`` are at least numerically positive semidefinite after internal
    diagonal regularisation.
    """

    def __init__(
        self,
        F:  np.ndarray,
        H:  np.ndarray,
        Q:  np.ndarray,
        R:  np.ndarray,
        m0: np.ndarray,
        P0: np.ndarray,
    ) -> None:
        self.F  = _to_f32(F)
        self.H  = _to_f32(H)
        self.Q  = _to_f32(Q)
        self.R  = _to_f32(R)
        self.m0 = _to_f32(m0)
        self.P0 = _to_f32(P0)

    # @tf.function is applied on the static helper; calling log_prob() is
    # gradient-safe (GradientTape can differentiate through lgssm_log_likelihood).
    def log_prob(self, Y: Tensor) -> Tensor:
        """Compute the exact marginal log-likelihood :math:`\\log p(y_{1:T})`.

        Parameters
        ----------
        Y : Tensor or ndarray, shape (T, ny)
            Sequence of observations.

        Returns
        -------
        log_lik : Tensor, scalar float32
            Kalman-filter marginal log-likelihood.
        """
        return lgssm_log_likelihood(
            _to_f32(Y), self.F, self.H, self.Q, self.R, self.m0, self.P0
        )

    def filter(self, Y: Tensor):
        """Run the full Kalman filter and return filtering results.

        Parameters
        ----------
        Y : Tensor or ndarray, shape (T, ny)
            Sequence of observations.

        Returns
        -------
        results : KFResultsTF
            Container returned by :func:`models.tf_core.kalman_filter_tf`
            containing predicted and filtered moments together with
            likelihood diagnostics.

        Notes
        -----
        Assumes ``Y`` is a time-ordered observation tensor with shape
        ``(T, ny)``.
        """
        return kalman_filter_tf(
            _to_f32(Y), self.F, self.H, self.Q, self.R, self.m0, self.P0
        )

    # TFP JointDistribution (optional – requires TFP)
    def as_joint_distribution(self):
        """Return a TFP joint distribution representation of the LGSSM.

        Returns
        -------
        joint : tfd.JointDistributionSequential
            Joint distribution over the initial state, one transition, and
            one observation kernel, suitable for TFP-based posterior methods.

        Notes
        -----
        Assumes TensorFlow Probability is installed and the stored covariance
        matrices admit numerically stable Cholesky factors after adding a
        small diagonal jitter.

        Raises
        ------
        ImportError
            If ``tensorflow_probability`` is not installed.
        """
        if not _HAS_TFP:
            raise ImportError("tensorflow_probability is required.")

        F, H, Q, R, m0, P0 = self.F, self.H, self.Q, self.R, self.m0, self.P0
        nx = int(m0.shape[0])
        ny = int(H.shape[0])

        def make_transition(x_prev):
            return tfd.MultivariateNormalTriL(
                loc         = tf.linalg.matvec(F, x_prev),
                scale_tril  = tf.linalg.cholesky(Q + 1e-9 * tf.eye(nx, dtype=F32)),
            )

        def make_observation(x_k):
            return tfd.MultivariateNormalTriL(
                loc         = tf.linalg.matvec(H, x_k),
                scale_tril  = tf.linalg.cholesky(R + 1e-9 * tf.eye(ny, dtype=F32)),
            )

        return tfd.JointDistributionSequential([
            tfd.MultivariateNormalTriL(
                loc        = m0,
                scale_tril = tf.linalg.cholesky(P0 + 1e-9 * tf.eye(nx, dtype=F32)),
            ),
            make_transition,
            make_observation,
        ])


# 2.  NonlinSSM_TFP  (scalar benchmark)
class NonlinSSM_TFP:
    """Scalar nonlinear SSM from the SMC benchmark literature.

        x_k = x_{k-1}/2 + 25*x_{k-1}/(1+x_{k-1}^2) + 8*cos(1.2*k) + v_k
        y_k = x_k^2 / 20 + w_k

    with v_k ~ N(0, sigma_v^2), w_k ~ N(0, sigma_w^2).

    Parameters
    ----------
    sigma_v : float, optional
        Process noise standard deviation.  Default ``1.0``.
    sigma_w : float, optional
        Measurement noise standard deviation.  Default ``1.0``.
    T : int, optional
        Sequence length.  Default ``100``.
    x0_mean : float, optional
        Initial state mean.  Default ``0.0``.
    x0_std : float, optional
        Initial state standard deviation.  Default ``1.0``.

    Notes
    -----
    Assumes a scalar latent state and scalar observations. The public
    log-probability methods in this class use Gaussian approximations rather
    than exact nonlinear marginalisation.
    """

    def __init__(
        self,
        sigma_v:  float = 1.0,
        sigma_w:  float = 1.0,
        T:        int   = 100,
        x0_mean:  float = 0.0,
        x0_std:   float = 1.0,
    ) -> None:
        self.sigma_v  = tf.cast(sigma_v,  F32)
        self.sigma_w  = tf.cast(sigma_w,  F32)
        self.T        = T
        self.x0_mean  = tf.constant(x0_mean, dtype=F32)
        self.x0_std   = tf.constant(x0_std,  dtype=F32)

    # ---- dynamics ----

    @staticmethod
    def _f(x: Tensor, k: Tensor) -> Tensor:
        """Compute the deterministic part of the state transition.

        Parameters
        ----------
        x : Tensor, scalar float32
            Current state value.
        k : Tensor, scalar int32
            Current time index (1-based).

        Returns
        -------
        f_x : Tensor, scalar float32
            Deterministic state prediction at time ``k``.

        Notes
        -----
        Assumes ``k`` follows the one-based indexing convention from the
        standard benchmark model definition.
        """
        k_f = tf.cast(k, F32)
        return x / 2.0 + 25.0 * x / (1.0 + x * x) + 8.0 * tf.cos(1.2 * k_f)

    @staticmethod
    def _h(x: Tensor) -> Tensor:
        """Compute the observation function :math:`h(x) = x^2 / 20`.

        Parameters
        ----------
        x : Tensor, scalar float32
            State value.

        Returns
        -------
        h_x : Tensor, scalar float32
            Predicted observation.

        Notes
        -----
        Assumes scalar latent state input.
        """
        return x * x / 20.0

    # ---- exact log-likelihood via @tf.function scan ----

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=F32),
                                   tf.TensorSpec(shape=[], dtype=F32),
                                   tf.TensorSpec(shape=[], dtype=F32)])
    def log_prob_tf(self, Y: Tensor, sigma_v: Tensor, sigma_w: Tensor) -> Tensor:
        """Compute the EKF-approximated marginal log-likelihood.

        Uses a single-particle EKF approximation for gradient-based
        optimisation and HMC warm-up.  For exact marginal likelihoods
        use :meth:`log_prob_pf` with a particle filter.

        Parameters
        ----------
        Y : Tensor, shape (T,)
            Scalar observation sequence.
        sigma_v : Tensor, scalar float32
            Process noise standard deviation.
        sigma_w : Tensor, scalar float32
            Measurement noise standard deviation.

        Returns
        -------
        log_lik : Tensor, scalar float32
            Sum of per-step EKF log-likelihood contributions.
        """
        T   = tf.shape(Y)[0]
        log2pi = tf.constant(math.log(2.0 * math.pi), dtype=F32)

        def step(carry, inputs):
            m, P, k, _ = carry
            y_k      = inputs

            # EKF prediction
            f_m = self._f(m, k)
            # Jacobian of f w.r.t. m:  df/dm = 1/2 + 25*(1 - m^2)/(1+m^2)^2
            dfdm = 0.5 + 25.0 * (1.0 - m * m) / tf.square(1.0 + m * m)
            P_pred = dfdm * P * dfdm + sigma_v * sigma_v

            # EKF update
            h_m  = self._h(f_m)
            dhdm = f_m / 10.0                         # d(x^2/20)/dx = x/10
            S    = dhdm * P_pred * dhdm + sigma_w * sigma_w + 1e-9
            K    = P_pred * dhdm / S
            innov= y_k - h_m
            m_up = f_m + K * innov
            P_up = (1.0 - K * dhdm) * P_pred

            # Per-step log-likelihood
            ll_k = -0.5 * (innov * innov / S + tf.math.log(S) + log2pi)
            return (m_up, P_up, k + 1, ll_k)

        _, _, _, ll_steps = tf.scan(
            fn          = step,
            elems       = Y,
            initializer = (self.x0_mean, self.x0_std ** 2,
                           tf.constant(1, dtype=tf.int32),
                           tf.constant(0.0, dtype=F32)),
        )
        return tf.reduce_sum(ll_steps)

    def log_prob(self, Y: Tensor) -> Tensor:
        """Compute the EKF-approximated log-likelihood using stored noise parameters.

        Parameters
        ----------
        Y : Tensor or ndarray, shape (T,)
            Scalar observation sequence.

        Returns
        -------
        log_lik : Tensor, scalar float32
            Approximate marginal log-likelihood.
        """
        return self.log_prob_tf(_to_f32(Y), self.sigma_v, self.sigma_w)


# 3.  HMC target builders (TFP)
def make_lgssm_hmc_target(
    Y:   np.ndarray,
    F:   np.ndarray,
    H:   np.ndarray,
    R:   np.ndarray,
    m0:  np.ndarray,
    P0:  np.ndarray,
    nx:  int,
) -> Callable:
    """Build a TFP ``target_log_prob_fn`` for learning the process-noise covariance.

    The free parameter is ``log_L_flat`` — the flattened lower-triangular
    entries of the log-Cholesky factor of :math:`Q`.  Diagonal entries are
    stored in log-space so the optimiser works in an unconstrained space.

    Parameters
    ----------
    Y : ndarray, shape (T, ny)
        Observation sequence.
    F : ndarray, shape (nx, nx)
        State transition matrix.
    H : ndarray, shape (ny, nx)
        Observation matrix.
    R : ndarray, shape (ny, ny)
        Observation noise covariance (fixed).
    m0 : ndarray, shape (nx,)
        Initial state mean.
    P0 : ndarray, shape (nx, nx)
        Initial state covariance.
    nx : int
        State dimension.

    Returns
    -------
    target : callable
        ``target(log_L_flat) -> scalar Tensor`` — the unnormalised log
        posterior used as the HMC/NUTS target.

    Raises
    ------
    ImportError
        If ``tensorflow_probability`` is not installed.
    """
    if not _HAS_TFP:
        raise ImportError("tensorflow_probability is required.")

    Y_tf  = _to_f32(Y)
    F_tf  = _to_f32(F)
    H_tf  = _to_f32(H)
    R_tf  = _to_f32(R)
    m0_tf = _to_f32(m0)
    P0_tf = _to_f32(P0)
    n_lower = nx * (nx + 1) // 2

    @tf.function
    def target(log_L_flat: Tensor) -> Tensor:
        # Reconstruct lower-triangular L from flat vector.
        # Diagonal entries are exp(log_L_flat[diag_idx]) > 0;
        # off-diagonal entries enter directly.
        L = tfb.FillScaleTriL().forward(log_L_flat)  # (nx, nx) lower-triangular, diag > 0
        Q = L @ tf.linalg.matrix_transpose(L)        # (nx, nx) PSD

        log_lik = lgssm_log_likelihood(Y_tf, F_tf, H_tf, Q, R_tf, m0_tf, P0_tf)
        # Weakly informative log-normal prior on diagonal entries
        log_diag = tf.math.log(tf.linalg.diag_part(L) + 1e-30)
        log_prior = tf.reduce_sum(
            tfd.Normal(0.0, 2.0).log_prob(log_diag)
        )
        return log_lik + log_prior

    return target


def make_nonlinssm_hmc_target(
    Y:         np.ndarray,
    x0_mean:   float = 0.0,
    x0_std:    float = 1.0,
) -> Callable:
    """Build a TFP ``target_log_prob_fn`` for learning the noise parameters.

    The free parameters are ``(log_sigma_v, log_sigma_w)`` — log-space
    representations of the process and measurement noise standard deviations
    for the scalar nonlinear SSM.

    Parameters
    ----------
    Y : ndarray, shape (T,)
        Scalar observation sequence.
    x0_mean : float, optional
        Initial state mean for the EKF.  Default ``0.0``.
    x0_std : float, optional
        Initial state standard deviation for the EKF.  Default ``1.0``.

    Returns
    -------
    target : callable
        ``target(log_sigma_v, log_sigma_w) -> scalar Tensor`` — the
        unnormalised log posterior used as the HMC/NUTS target.

    Raises
    ------
    ImportError
        If ``tensorflow_probability`` is not installed.
    """
    if not _HAS_TFP:
        raise ImportError("tensorflow_probability is required.")

    Y_tf  = _to_f32(Y)
    model = NonlinSSM_TFP(T=len(Y), x0_mean=x0_mean, x0_std=x0_std)

    @tf.function
    def target(log_sigma_v: Tensor, log_sigma_w: Tensor) -> Tensor:
        sv = tf.exp(log_sigma_v)
        sw = tf.exp(log_sigma_w)
        log_lik   = model.log_prob_tf(Y_tf, sv, sw)
        # Half-Normal(1) priors on sigma_v, sigma_w
        log_prior = (tfd.HalfNormal(1.0).log_prob(sv)
                   + tfd.HalfNormal(1.0).log_prob(sw))
        # Jacobian of exp transform (change-of-variables)
        log_jac   = log_sigma_v + log_sigma_w
        return log_lik + log_prior + log_jac

    return target


# ---------------------------------------------------------------------------
# 4.  Convenience: run TFP HMC / NUTS sampler
# ---------------------------------------------------------------------------

def run_hmc(
    target_log_prob_fn: Callable,
    init_state:         Sequence[Tensor],
    num_results:        int   = 1000,
    num_burnin:         int   = 500,
    step_size:          float = 0.01,
    num_leapfrog:       int   = 10,
    use_nuts:           bool  = False,
) -> Tuple:
    """Run TFP HMC or NUTS and return posterior samples.

    Dual-averaging step-size adaptation is applied during the burn-in
    phase.  The adapted kernel is used unchanged during the sampling phase.

    Parameters
    ----------
    target_log_prob_fn : callable
        Function mapping a list of parameter tensors to a scalar log
        unnormalised posterior density.
    init_state : sequence of Tensor
        Starting point for the Markov chain; one tensor per free parameter.
    num_results : int, optional
        Number of posterior samples to collect after burn-in.
        Default ``1000``.
    num_burnin : int, optional
        Number of warm-up steps (discarded).  Default ``500``.
    step_size : float, optional
        Initial HMC step size (or NUTS initial step size).  Default ``0.01``.
    num_leapfrog : int, optional
        Number of leapfrog steps per HMC transition.  Ignored when
        ``use_nuts=True``.  Default ``10``.
    use_nuts : bool, optional
        If ``True``, use :class:`tfp.mcmc.NoUTurnSampler`; otherwise use
        :class:`tfp.mcmc.HamiltonianMonteCarlo`.  Default ``False``.

    Returns
    -------
    samples : list of Tensor, each shape (num_results, ...)
        Posterior samples for each free parameter.
    is_accepted : Tensor of bool, shape (num_results,)
        Per-step acceptance indicator.

    Raises
    ------
    ImportError
        If ``tensorflow_probability`` is not installed.
    """
    if not _HAS_TFP:
        raise ImportError("tensorflow_probability is required.")

    if use_nuts:
        kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn = target_log_prob_fn,
            step_size          = step_size,
        )
    else:
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn = target_log_prob_fn,
            step_size          = step_size,
            num_leapfrog_steps = num_leapfrog,
        )

    # Dual-average step-size adaptation during burn-in
    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel       = kernel,
        num_adaptation_steps= int(0.8 * num_burnin),
    )

    @tf.function
    def run_chain():
        return tfp.mcmc.sample_chain(
            num_results         = num_results,
            num_burnin_steps    = num_burnin,
            current_state       = init_state,
            kernel              = adaptive_kernel,
            trace_fn            = lambda _, pkr: pkr.inner_results.is_accepted
                                  if not use_nuts
                                  else pkr.inner_results.leapfrogs_taken > 0,
        )

    samples, is_accepted = run_chain()
    return samples, is_accepted
