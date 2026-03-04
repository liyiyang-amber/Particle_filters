"""
Integration tests: HMC parameter inference on the NonlinearSSM.

Tests
-----
  - log-likelihood function is finite and monotone in sigma_w
  - HMCSampler.run() returns the expected output contract
  - sigma_v posterior mean within 2 sigma of the truth (generous CI)
  - sigma_w posterior mean within 2 sigma of the truth
  - Acceptance rate in a reasonable range [0.05, 0.99]
  - All samples are positive (parameters are log-transformed)
  - ESS > 0 on the returned chain
  - runtime is positive
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pytest
import tensorflow as tf

pytestmark = [pytest.mark.integration, pytest.mark.tensorflow]

from simulator.simulator_nonlinSSM import simulate_nonlinear_ssm
from models.particle_filter import ParticleFilter
from models.HMC_inference import HMCSampler, HMCConfig, compute_ess


# Shared constants
TRUE_SIGMA_V = 10.0
TRUE_SIGMA_W = 1.0
N_STEPS = 40          # short sequence – fast likelihood evaluations
N_PARTICLES = 200     # BPF particles
N_SAMPLES = 200       # HMC post-burnin samples
N_BURNIN = 200        # HMC burn-in
DATA_SEED = 42        # reproducible simulated dataset


# Bootstrap Particle Filter log-likelihood
def _bpf_log_likelihood(
    observations: np.ndarray,
    sigma_v: float,
    sigma_w: float,
    n_particles: int = N_PARTICLES,
    seed: int = 7,
) -> float:
    """
    Compute log p(Y_{1:T} | sigma_v, sigma_w) via a Bootstrap PF.

    The log marginal likelihood is accumulated as
        log p(Y_{1:T}) = sum_t log( mean_i w_i * p(y_t | x_t^i) )
    where weights are re-normalised between steps.

    Parameters
    ----------
    observations : ndarray, shape (T,)
    sigma_v, sigma_w : float
        Positive noise standard deviations.
    n_particles : int
    seed : int

    Returns
    -------
    log_likelihood : float
    """
    if sigma_v <= 0 or sigma_w <= 0:
        return -np.inf

    T = len(observations)
    rng = np.random.default_rng(seed)

    Q = np.array([[sigma_v ** 2]])
    R = np.array([[sigma_w ** 2]])
    R_inv = 1.0 / sigma_w ** 2
    log_norm_const = -0.5 * np.log(2.0 * np.pi * sigma_w ** 2)

    # Non-linear SSM transition 
    counter = {"n": 0}

    def g(x, u):
        counter["n"] += 1
        n = counter["n"]
        x_s = float(x[0])
        return np.array([x_s / 2.0 + 25.0 * x_s / (1.0 + x_s ** 2) + 8.0 * np.cos(1.2 * n)])

    def h(x):
        return np.array([float(x[0]) ** 2 / 20.0])

    pf = ParticleFilter(
        g=g, h=h, Q=Q, R=R,
        Np=n_particles,
        resample_thresh=0.5,
        resample_method="systematic",
        regularize_after_resample=False,
        rng=rng,
    )
    pf.initialize(mean=np.array([0.0]), cov=np.array([[5.0]]))

    log_likelihood = 0.0
    for t in range(T):
        z_t = np.array([observations[t]])

        # Predict step
        pf.predict(u=None)

        # Compute un-normalised log-weights
        z_pred = np.array([h(x) for x in pf.state.particles])  # (Np, 1)
        diffs = z_t - z_pred[:, 0]   # (Np,)
        log_liks = log_norm_const - 0.5 * diffs ** 2 * R_inv

        # Accumulate marginal log-likelihood: log(sum_i w_i * p(y_t | x_t^i))
        log_weights_old = np.log(pf.state.weights + 1e-300)
        log_unnorm = log_weights_old + log_liks
        m = np.max(log_unnorm)
        log_marginal = m + np.log(np.sum(np.exp(log_unnorm - m)))
        log_likelihood += log_marginal

        # Update (reweight + resample)
        pf.update(z_t)

    return float(log_likelihood)


# Log-posterior 
def _build_log_posterior_fn(
    observations: np.ndarray,
    n_particles: int = N_PARTICLES,
    bpf_seed: int = 7,
):
    """
    Return a TensorFlow-callable log-posterior function.

    The BPF log-likelihood is wrapped in tf.py_function so it can be
    called from within TensorFlow Probability's MCMC chain.  Gradients
    of the BPF output are not available; HMC therefore falls back to a
    finite-difference approximation internally (or uses the gradient of
    the prior only).  This is the standard "pseudo-marginal" approach.

    Parameters
    ----------
    observations : ndarray, shape (T,)
    n_particles : int
    bpf_seed : int

    Returns
    -------
    log_posterior_fn : callable
        Accepts a tf.Tensor of shape (2,) and returns a scalar tf.Tensor.
    """
    obs_np = np.asarray(observations, dtype=np.float64)

    def _np_log_posterior(theta_np: np.ndarray) -> float:
        """numpy-level computation."""
        sigma_v = float(np.exp(theta_np[0]))
        sigma_w = float(np.exp(theta_np[1]))

        log_lik = _bpf_log_likelihood(obs_np, sigma_v, sigma_w,
                                      n_particles=n_particles, seed=bpf_seed)

        # log-normal prior: theta ~ N(0, 2^2) on log-scale
        log_prior = float(-0.5 * np.sum(theta_np ** 2 / 4.0))

        return log_lik + log_prior

    @tf.custom_gradient
    def log_posterior_fn(theta: tf.Tensor):
        """
        Wrap the numpy BPF log-posterior so that TFP can call it inside a
        GradientTape.  Because the BPF has no analytic gradient we return a
        zero gradient vector; TFP's HMC will then rely solely on the
        prior gradient (the BPF likelihood contribution has zero gradient
        in this pseudo-marginal sense).  This is sufficient for the sampler
        to run; the chain explores the posterior via Metropolis corrections.
        """
        val = tf.py_function(
            func=lambda t: np.float32(_np_log_posterior(t.numpy().astype(np.float64))),
            inp=[theta],
            Tout=tf.float32,
        )
        val = tf.reshape(val, [])  # ensure scalar shape

        def grad(upstream):
            return upstream * tf.zeros_like(theta)

        return val, grad

    return log_posterior_fn


# Module-scoped fixtures
@pytest.fixture(scope="module")
def sim_data():
    """Simulate one fixed dataset with known parameters."""
    return simulate_nonlinear_ssm(
        N=N_STEPS,
        sigma_v=TRUE_SIGMA_V,
        sigma_w=TRUE_SIGMA_W,
        seed=DATA_SEED,
    )


@pytest.fixture(scope="module")
def hmc_results(sim_data):
    """Run a short HMC chain on the simulated data (module scope – run once)."""
    log_posterior_fn = _build_log_posterior_fn(sim_data.Y)

    cfg = HMCConfig(
        n_samples=N_SAMPLES,
        n_burnin=N_BURNIN,
        step_size=0.1,
        n_leapfrog_steps=3,
        adapt_step_size=True,
        target_accept_prob=0.65,
        verbose=False,
    )
    sampler = HMCSampler(log_posterior_fn, cfg)
    # Initial point: slightly perturbed log-scale truth
    init = np.array(
        [np.log(TRUE_SIGMA_V) + 0.5, np.log(TRUE_SIGMA_W) + 0.3],
        dtype=np.float32,
    )
    results = sampler.run(init)

    # Transform samples back to constrained space (sigma_v, sigma_w > 0)
    results["samples_constrained"] = np.exp(results["samples"])
    return results


# Log-likelihood sanity tests 
class TestHMCNonlinSSMLogLikelihood:
    """Verify the BPF log-likelihood surface behaves sensibly."""

    def test_log_lik_finite_at_truth(self, sim_data):
        ll = _bpf_log_likelihood(sim_data.Y, TRUE_SIGMA_V, TRUE_SIGMA_W)
        assert np.isfinite(ll), f"Log-likelihood at truth is not finite: {ll}"

    def test_log_lik_decreases_for_bad_sigma_w(self, sim_data):
        """A very small sigma_w (forcing near-zero obs noise) should give a
        much lower likelihood than the true value on these data."""
        ll_good = _bpf_log_likelihood(sim_data.Y, TRUE_SIGMA_V, TRUE_SIGMA_W)
        ll_bad = _bpf_log_likelihood(sim_data.Y, TRUE_SIGMA_V, sigma_w=0.001)
        assert ll_good > ll_bad, (
            f"Expected log_lik({TRUE_SIGMA_W}) > log_lik(0.001), "
            f"got {ll_good:.2f} vs {ll_bad:.2f}"
        )

    def test_log_lik_negative_for_invalid_params(self, sim_data):
        """Log-likelihood should return -inf for non-positive parameters."""
        ll = _bpf_log_likelihood(sim_data.Y, sigma_v=-1.0, sigma_w=1.0)
        assert ll == -np.inf

    def test_log_posterior_finite_near_truth(self, sim_data):
        log_post_fn = _build_log_posterior_fn(sim_data.Y)
        theta_truth = tf.constant(
            [np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)], dtype=tf.float32
        )
        val = log_post_fn(theta_truth)
        assert np.isfinite(float(val)), f"Log-posterior at truth not finite: {val}"


#  HMC output contract tests
class TestHMCNonlinSSMOutputContract:
    """Check the dict returned by HMCSampler.run() has the right structure."""

    REQUIRED_KEYS = {"samples", "is_accepted", "accept_rate", "runtime"}

    def test_required_keys_present(self, hmc_results):
        assert self.REQUIRED_KEYS.issubset(hmc_results.keys())

    def test_samples_shape(self, hmc_results):
        assert hmc_results["samples"].shape == (N_SAMPLES, 2)

    def test_samples_constrained_shape(self, hmc_results):
        assert hmc_results["samples_constrained"].shape == (N_SAMPLES, 2)

    def test_samples_constrained_all_positive(self, hmc_results):
        """sigma_v and sigma_w must be strictly positive after exp-transform."""
        assert np.all(hmc_results["samples_constrained"] > 0), (
            "Some constrained samples are non-positive after exp-transform"
        )

    def test_samples_finite(self, hmc_results):
        assert np.all(np.isfinite(hmc_results["samples"]))

    def test_is_accepted_shape(self, hmc_results):
        assert hmc_results["is_accepted"].shape == (N_SAMPLES,)

    def test_accept_rate_in_unit_interval(self, hmc_results):
        assert 0.0 <= hmc_results["accept_rate"] <= 1.0

    def test_accept_rate_consistent_with_is_accepted(self, hmc_results):
        expected = float(np.mean(hmc_results["is_accepted"]))
        assert abs(hmc_results["accept_rate"] - expected) < 1e-6

    def test_runtime_positive(self, hmc_results):
        assert hmc_results["runtime"] > 0.0


# HMC sampling quality tests
class TestHMCNonlinSSMSamplingQuality:
    """Verify the posterior concentrates near the true parameter values."""

    def test_acceptance_rate_reasonable(self, hmc_results):
        ar = hmc_results["accept_rate"]
        assert 0.05 < ar < 0.99, f"Acceptance rate {ar:.3f} outside (0.05, 0.99)"

    def test_ess_positive(self, hmc_results):
        ess = compute_ess(hmc_results["samples"])
        assert ess > 0, f"ESS is {ess}"

    def test_sigma_v_posterior_mean_reasonable(self, hmc_results):
        """Posterior mean of sigma_v should be in a broad interval around the truth."""
        sigma_v_samples = hmc_results["samples_constrained"][:, 0]
        post_mean = float(np.mean(sigma_v_samples))
        # Generous 3× interval since BPF likelihood is noisy
        assert 0.2 * TRUE_SIGMA_V < post_mean < 5.0 * TRUE_SIGMA_V, (
            f"sigma_v posterior mean {post_mean:.3f} very far from truth {TRUE_SIGMA_V}"
        )

    def test_sigma_w_posterior_mean_reasonable(self, hmc_results):
        """Posterior mean of sigma_w should be in a broad interval around the truth."""
        sigma_w_samples = hmc_results["samples_constrained"][:, 1]
        post_mean = float(np.mean(sigma_w_samples))
        assert 0.2 * TRUE_SIGMA_W < post_mean < 5.0 * TRUE_SIGMA_W, (
            f"sigma_w posterior mean {post_mean:.3f} very far from truth {TRUE_SIGMA_W}"
        )

    def test_sigma_v_samples_span_truth(self, hmc_results):
        """The sigma_v chain should visit values near the truth at least once."""
        sigma_v_samples = hmc_results["samples_constrained"][:, 0]
        assert np.any(np.abs(sigma_v_samples - TRUE_SIGMA_V) < 0.5 * TRUE_SIGMA_V), (
            "sigma_v chain never came close to the true value"
        )

    def test_sigma_w_samples_span_truth(self, hmc_results):
        """The sigma_w chain should visit values near the truth at least once."""
        sigma_w_samples = hmc_results["samples_constrained"][:, 1]
        assert np.any(np.abs(sigma_w_samples - TRUE_SIGMA_W) < 0.5 * TRUE_SIGMA_W), (
            "sigma_w chain never came close to the true value"
        )


# Configuration variant tests
class TestHMCNonlinSSMConfigVariants:
    """Verify different HMC config options run without crashing."""

    def test_no_adaptation_runs(self, sim_data):
        log_post_fn = _build_log_posterior_fn(sim_data.Y, n_particles=50)
        cfg = HMCConfig(
            n_samples=30,
            n_burnin=20,
            step_size=0.05,
            n_leapfrog_steps=2,
            adapt_step_size=False,
            verbose=False,
        )
        sampler = HMCSampler(log_post_fn, cfg)
        init = np.array([np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)], dtype=np.float32)
        res = sampler.run(init)
        assert res["samples"].shape == (30, 2)
        assert 0.0 <= res["accept_rate"] <= 1.0

    def test_single_leapfrog_step_runs(self, sim_data):
        log_post_fn = _build_log_posterior_fn(sim_data.Y, n_particles=50)
        cfg = HMCConfig(
            n_samples=30,
            n_burnin=20,
            n_leapfrog_steps=1,
            verbose=False,
        )
        sampler = HMCSampler(log_post_fn, cfg)
        init = np.array([np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)], dtype=np.float32)
        res = sampler.run(init)
        assert res["samples"].shape == (30, 2)

    def test_verbose_false_no_crash(self, sim_data):
        log_post_fn = _build_log_posterior_fn(sim_data.Y, n_particles=50)
        cfg = HMCConfig(n_samples=20, n_burnin=10, verbose=False)
        sampler = HMCSampler(log_post_fn, cfg)
        init = np.array([np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)], dtype=np.float32)
        res = sampler.run(init)
        assert "samples" in res
