"""
Integration tests: PMMH parameter inference on the NonlinearSSM.

Tests
-----
  - log-likelihood function is finite and monotone in sigma_w
  - PMMHSampler.run() returns the expected output contract
  - sigma_v posterior mean within generous CI of the truth
  - sigma_w posterior mean within generous CI of the truth
  - Acceptance rate in a reasonable range [0.01, 0.99]
  - All samples are positive (parameters are log-transformed)
  - ESS > 0 on the returned chain
  - runtime is positive
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pytest

pytestmark = [pytest.mark.integration]

from simulator.simulator_nonlinSSM import simulate_nonlinear_ssm
from models.particle_filter import ParticleFilter
from models.PMMH_inference import PMMHSampler, PMMHConfig, compute_ess


# Shared constants
TRUE_SIGMA_V = 10.0
TRUE_SIGMA_W = 1.0
N_STEPS = 40          # short sequence – fast likelihood evaluations
N_PARTICLES = 200     # BPF particles
N_SAMPLES = 200       # PMMH post-burnin samples
N_BURNIN = 200        # PMMH burn-in
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


# Log-likelihood and log-prior factories 
def _build_log_likelihood_fn(
    observations: np.ndarray,
    n_particles: int = N_PARTICLES,
    bpf_seed: int = 7,
):
    """
    Return a numpy-callable log-likelihood function for PMMH.

    The parameters are supplied in log-space:
        theta[0] = log(sigma_v),  theta[1] = log(sigma_w).

    Parameters
    ----------
    observations : ndarray, shape (T,)
    n_particles : int
    bpf_seed : int

    Returns
    -------
    log_likelihood_fn : callable
        Accepts theta (ndarray, shape (2,)) and returns a float.
    """
    obs_np = np.asarray(observations, dtype=np.float64)

    def log_likelihood_fn(theta: np.ndarray) -> float:
        sigma_v = float(np.exp(theta[0]))
        sigma_w = float(np.exp(theta[1]))
        return _bpf_log_likelihood(obs_np, sigma_v, sigma_w,
                                   n_particles=n_particles, seed=bpf_seed)

    return log_likelihood_fn


def _build_log_prior_fn():
    """
    Return a log-prior function for PMMH.

    Prior: theta ~ N(0, 2^2) on the log-scale (log-normal prior on sigma).

    Returns
    -------
    log_prior_fn : callable
        Accepts theta (ndarray) and returns a float.
    """
    def log_prior_fn(theta: np.ndarray) -> float:
        return float(-0.5 * np.sum(theta ** 2 / 4.0))

    return log_prior_fn


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
def pmmh_results(sim_data):
    """Run a short PMMH chain on the simulated data (module scope – run once)."""
    log_likelihood_fn = _build_log_likelihood_fn(sim_data.Y)
    log_prior_fn = _build_log_prior_fn()

    cfg = PMMHConfig(
        n_samples=N_SAMPLES,
        n_burnin=N_BURNIN,
        proposal_std=np.array([0.3, 0.3], dtype=np.float64),
        adapt_proposal=True,
        adapt_interval=50,
        target_accept_rate=0.234,
        verbose=False,
    )
    sampler = PMMHSampler(log_likelihood_fn, log_prior_fn, cfg)

    # Initial point: slightly perturbed log-scale truth
    init = np.array(
        [np.log(TRUE_SIGMA_V) + 0.5, np.log(TRUE_SIGMA_W) + 0.3],
        dtype=np.float64,
    )
    results = sampler.run(init)

    # Transform samples back to constrained space (sigma_v, sigma_w > 0)
    results["samples_constrained"] = np.exp(results["samples"])
    return results


# Log-likelihood sanity tests (do not depend on the PMMH chain)
class TestPMMHNonlinSSMLogLikelihood:
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
        log_lik_fn = _build_log_likelihood_fn(sim_data.Y)
        log_prior_fn = _build_log_prior_fn()
        theta_truth = np.array(
            [np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)], dtype=np.float64
        )
        val = log_lik_fn(theta_truth) + log_prior_fn(theta_truth)
        assert np.isfinite(val), f"Log-posterior at truth not finite: {val}"

    def test_log_likelihood_fn_returns_float(self, sim_data):
        log_lik_fn = _build_log_likelihood_fn(sim_data.Y, n_particles=50)
        theta = np.array([np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)])
        val = log_lik_fn(theta)
        assert isinstance(val, float), f"Expected float, got {type(val)}"

    def test_log_prior_fn_returns_float(self):
        log_prior_fn = _build_log_prior_fn()
        theta = np.array([np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)])
        val = log_prior_fn(theta)
        assert isinstance(val, float), f"Expected float, got {type(val)}"


# PMMH output contract tests
class TestPMMHNonlinSSMOutputContract:
    """Check the dict returned by PMMHSampler.run() has the right structure."""

    REQUIRED_KEYS = {
        "samples", "accept_trace", "accept_rate",
        "log_likelihood_trace", "log_prior_trace", "runtime",
    }

    def test_required_keys_present(self, pmmh_results):
        assert self.REQUIRED_KEYS.issubset(pmmh_results.keys())

    def test_samples_shape(self, pmmh_results):
        assert pmmh_results["samples"].shape == (N_SAMPLES, 2)

    def test_samples_constrained_shape(self, pmmh_results):
        assert pmmh_results["samples_constrained"].shape == (N_SAMPLES, 2)

    def test_samples_constrained_all_positive(self, pmmh_results):
        """sigma_v and sigma_w must be strictly positive after exp-transform."""
        assert np.all(pmmh_results["samples_constrained"] > 0), (
            "Some constrained samples are non-positive after exp-transform"
        )

    def test_samples_finite(self, pmmh_results):
        assert np.all(np.isfinite(pmmh_results["samples"]))

    def test_accept_trace_shape(self, pmmh_results):
        assert pmmh_results["accept_trace"].shape == (N_SAMPLES + N_BURNIN,)

    def test_accept_rate_in_unit_interval(self, pmmh_results):
        assert 0.0 <= pmmh_results["accept_rate"] <= 1.0

    def test_accept_rate_consistent_with_accept_trace(self, pmmh_results):
        """Overall accept_rate is computed over the full chain (incl. burnin)."""
        expected = float(np.mean(pmmh_results["accept_trace"]))
        assert abs(pmmh_results["accept_rate"] - expected) < 1e-6

    def test_log_likelihood_trace_shape(self, pmmh_results):
        assert pmmh_results["log_likelihood_trace"].shape == (N_SAMPLES + N_BURNIN,)

    def test_log_prior_trace_shape(self, pmmh_results):
        assert pmmh_results["log_prior_trace"].shape == (N_SAMPLES + N_BURNIN,)

    def test_log_likelihood_trace_finite_at_accepted(self, pmmh_results):
        """Wherever PMMH accepted a proposal the stored log-likelihood is finite."""
        accepted_idx = np.where(pmmh_results["accept_trace"])[0]
        if len(accepted_idx) == 0:
            pytest.skip("No accepted proposals in chain – cannot validate")
        finite = np.isfinite(
            pmmh_results["log_likelihood_trace"][accepted_idx]
        )
        assert np.all(finite), "Non-finite log-likelihood stored at accepted steps"

    def test_runtime_positive(self, pmmh_results):
        assert pmmh_results["runtime"] > 0.0

    def test_proposal_std_present_and_positive(self, pmmh_results):
        std = pmmh_results["proposal_std"]
        assert std is not None
        assert np.all(std > 0), f"Proposal std not positive: {std}"


# PMMH sampling quality tests
class TestPMMHNonlinSSMSamplingQuality:
    """Verify the posterior concentrates near the true parameter values."""

    def test_acceptance_rate_reasonable(self, pmmh_results):
        ar = pmmh_results["accept_rate"]
        assert 0.01 < ar < 0.99, f"Acceptance rate {ar:.3f} outside (0.01, 0.99)"

    def test_ess_positive(self, pmmh_results):
        ess = compute_ess(pmmh_results["samples"])
        assert ess > 0, f"ESS is {ess}"

    def test_sigma_v_posterior_mean_reasonable(self, pmmh_results):
        """Posterior mean of sigma_v should be in a broad interval around the truth."""
        sigma_v_samples = pmmh_results["samples_constrained"][:, 0]
        post_mean = float(np.mean(sigma_v_samples))
        # Generous 5× interval since BPF likelihood is noisy
        assert 0.2 * TRUE_SIGMA_V < post_mean < 5.0 * TRUE_SIGMA_V, (
            f"sigma_v posterior mean {post_mean:.3f} very far from truth {TRUE_SIGMA_V}"
        )

    def test_sigma_w_posterior_mean_reasonable(self, pmmh_results):
        """Posterior mean of sigma_w should be in a broad interval around the truth."""
        sigma_w_samples = pmmh_results["samples_constrained"][:, 1]
        post_mean = float(np.mean(sigma_w_samples))
        assert 0.2 * TRUE_SIGMA_W < post_mean < 5.0 * TRUE_SIGMA_W, (
            f"sigma_w posterior mean {post_mean:.3f} very far from truth {TRUE_SIGMA_W}"
        )

    def test_sigma_v_samples_span_truth(self, pmmh_results):
        """The sigma_v chain should visit values near the truth at least once."""
        sigma_v_samples = pmmh_results["samples_constrained"][:, 0]
        assert np.any(np.abs(sigma_v_samples - TRUE_SIGMA_V) < 0.5 * TRUE_SIGMA_V), (
            "sigma_v chain never came close to the true value"
        )

    def test_sigma_w_samples_span_truth(self, pmmh_results):
        """The sigma_w chain should visit values near the truth at least once."""
        sigma_w_samples = pmmh_results["samples_constrained"][:, 1]
        assert np.any(np.abs(sigma_w_samples - TRUE_SIGMA_W) < 0.5 * TRUE_SIGMA_W), (
            "sigma_w chain never came close to the true value"
        )

    def test_log_likelihood_trace_not_all_same(self, pmmh_results):
        """Chain should have moved at least once during burn-in + sampling."""
        ll = pmmh_results["log_likelihood_trace"]
        assert not np.all(ll == ll[0]), (
            "Log-likelihood trace is constant – chain appears stuck"
        )

    def test_post_burnin_accept_rate_reasonable(self, pmmh_results):
        """Post-burnin acceptance rate should be in a sensible range."""
        ar_post = pmmh_results["accept_rate_post_burnin"]
        assert 0.01 < ar_post < 0.99, (
            f"Post-burnin acceptance rate {ar_post:.3f} outside (0.01, 0.99)"
        )


# Configuration variant tests
class TestPMMHNonlinSSMConfigVariants:
    """Verify different PMMH config options run without crashing."""

    def test_no_adaptation_runs(self, sim_data):
        log_lik_fn = _build_log_likelihood_fn(sim_data.Y, n_particles=50)
        log_prior_fn = _build_log_prior_fn()
        cfg = PMMHConfig(
            n_samples=30,
            n_burnin=20,
            proposal_std=np.array([0.2, 0.2]),
            adapt_proposal=False,
            verbose=False,
        )
        sampler = PMMHSampler(log_lik_fn, log_prior_fn, cfg)
        init = np.array([np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)], dtype=np.float64)
        res = sampler.run(init)
        assert res["samples"].shape == (30, 2)
        assert 0.0 <= res["accept_rate"] <= 1.0

    def test_small_proposal_std_runs(self, sim_data):
        """Very small proposal std should still run (low acceptance rate expected)."""
        log_lik_fn = _build_log_likelihood_fn(sim_data.Y, n_particles=50)
        log_prior_fn = _build_log_prior_fn()
        cfg = PMMHConfig(
            n_samples=30,
            n_burnin=20,
            proposal_std=np.array([0.01, 0.01]),
            adapt_proposal=False,
            verbose=False,
        )
        sampler = PMMHSampler(log_lik_fn, log_prior_fn, cfg)
        init = np.array([np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)], dtype=np.float64)
        res = sampler.run(init)
        assert res["samples"].shape == (30, 2)

    def test_large_proposal_std_runs(self, sim_data):
        """Very large proposal std should still run (low acceptance rate expected)."""
        log_lik_fn = _build_log_likelihood_fn(sim_data.Y, n_particles=50)
        log_prior_fn = _build_log_prior_fn()
        cfg = PMMHConfig(
            n_samples=30,
            n_burnin=20,
            proposal_std=np.array([2.0, 2.0]),
            adapt_proposal=False,
            verbose=False,
        )
        sampler = PMMHSampler(log_lik_fn, log_prior_fn, cfg)
        init = np.array([np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)], dtype=np.float64)
        res = sampler.run(init)
        assert res["samples"].shape == (30, 2)
        assert 0.0 <= res["accept_rate"] <= 1.0

    def test_default_proposal_std_runs(self, sim_data):
        """Running with proposal_std=None should use the auto-computed default."""
        log_lik_fn = _build_log_likelihood_fn(sim_data.Y, n_particles=50)
        log_prior_fn = _build_log_prior_fn()
        cfg = PMMHConfig(
            n_samples=20,
            n_burnin=10,
            proposal_std=None,
            verbose=False,
        )
        sampler = PMMHSampler(log_lik_fn, log_prior_fn, cfg)
        init = np.array([np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)], dtype=np.float64)
        res = sampler.run(init)
        assert "samples" in res
        assert res["samples"].shape == (20, 2)

    def test_verbose_false_no_crash(self, sim_data):
        log_lik_fn = _build_log_likelihood_fn(sim_data.Y, n_particles=50)
        log_prior_fn = _build_log_prior_fn()
        cfg = PMMHConfig(n_samples=20, n_burnin=10, verbose=False)
        sampler = PMMHSampler(log_lik_fn, log_prior_fn, cfg)
        init = np.array([np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)], dtype=np.float64)
        res = sampler.run(init)
        assert "samples" in res

    def test_samples_full_contains_burnin(self, sim_data):
        """samples_full should contain both burnin and post-burnin samples."""
        log_lik_fn = _build_log_likelihood_fn(sim_data.Y, n_particles=50)
        log_prior_fn = _build_log_prior_fn()
        n_s, n_b = 30, 20
        cfg = PMMHConfig(
            n_samples=n_s,
            n_burnin=n_b,
            proposal_std=np.array([0.3, 0.3]),
            verbose=False,
        )
        sampler = PMMHSampler(log_lik_fn, log_prior_fn, cfg)
        init = np.array([np.log(TRUE_SIGMA_V), np.log(TRUE_SIGMA_W)], dtype=np.float64)
        res = sampler.run(init)
        assert res["samples_full"].shape == (n_s + n_b, 2), (
            f"Expected samples_full shape ({n_s + n_b}, 2), "
            f"got {res['samples_full'].shape}"
        )
