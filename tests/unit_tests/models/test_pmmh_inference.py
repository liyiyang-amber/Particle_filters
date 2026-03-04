"""
Unit tests for PMMH
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from models.PMMH_inference import PMMHSampler, PMMHConfig, compute_ess, compute_ess_per_second

pytestmark = pytest.mark.tensorflow   # keep consistent marker with rest of suite


# Shared helpers / fixtures
def _flat_log_lik(theta):
    """Uninformative likelihood — posterior = prior."""
    return 0.0


def _gaussian_log_prior(theta, mu, sigma):
    return float(-0.5 * np.sum(((theta - mu) / sigma) ** 2))


@pytest.fixture(scope="module")
def gaussian_1d_prior_params():
    return {"mu": np.array([2.0]), "sigma": np.array([1.5])}


@pytest.fixture(scope="module")
def gaussian_2d_prior_params():
    """Independent 2-D Gaussian with known mean and std."""
    return {
        "mu": np.array([1.0, -2.0]),
        "sigma": np.array([1.0, 2.0]),
    }


@pytest.fixture(scope="module")
def pmmh_results_1d(gaussian_1d_prior_params):
    p = gaussian_1d_prior_params
    cfg = PMMHConfig(
        n_samples=600,
        n_burnin=300,
        proposal_std=np.array([0.4]),
        adapt_proposal=True,
        verbose=False,
    )
    def log_prior(theta):
        return _gaussian_log_prior(theta, p["mu"], p["sigma"])
    sampler = PMMHSampler(_flat_log_lik, log_prior, cfg)
    return sampler.run(p["mu"].copy())


@pytest.fixture(scope="module")
def pmmh_results_2d(gaussian_2d_prior_params):
    p = gaussian_2d_prior_params
    cfg = PMMHConfig(
        n_samples=800,
        n_burnin=400,
        proposal_std=np.array([0.5, 1.0]),
        adapt_proposal=True,
        verbose=False,
    )
    def log_prior(theta):
        return _gaussian_log_prior(theta, p["mu"], p["sigma"])
    sampler = PMMHSampler(_flat_log_lik, log_prior, cfg)
    return sampler.run(p["mu"].copy())


# PMMHConfig tests
class TestPMMHConfig:
    def test_default_n_samples(self):
        assert PMMHConfig().n_samples == 1000

    def test_default_n_burnin(self):
        assert PMMHConfig().n_burnin == 500

    def test_default_proposal_std_is_none(self):
        assert PMMHConfig().proposal_std is None

    def test_default_adapt_proposal(self):
        assert PMMHConfig().adapt_proposal is True

    def test_default_adapt_interval_positive(self):
        assert PMMHConfig().adapt_interval > 0

    def test_default_target_accept_rate_in_unit_interval(self):
        assert 0.0 < PMMHConfig().target_accept_rate < 1.0

    def test_custom_values_stored(self):
        std = np.array([0.1, 0.2])
        cfg = PMMHConfig(
            n_samples=200,
            n_burnin=100,
            proposal_std=std,
            adapt_proposal=False,
            adapt_interval=25,
            target_accept_rate=0.3,
            verbose=False,
        )
        assert cfg.n_samples == 200
        assert cfg.n_burnin == 100
        assert np.allclose(cfg.proposal_std, std)
        assert cfg.adapt_proposal is False
        assert cfg.adapt_interval == 25
        assert abs(cfg.target_accept_rate - 0.3) < 1e-9
        assert cfg.verbose is False


# PMMHSampler output contract tests
class TestPMMHSamplerOutputContract:
    REQUIRED_KEYS = {
        "samples",
        "samples_full",
        "log_likelihood_trace",
        "log_prior_trace",
        "accept_trace",
        "accept_rate",
        "accept_rate_post_burnin",
        "runtime",
        "proposal_std",
    }

    def test_result_keys_present(self, pmmh_results_1d):
        assert self.REQUIRED_KEYS.issubset(pmmh_results_1d.keys())

    def test_samples_is_ndarray(self, pmmh_results_1d):
        assert isinstance(pmmh_results_1d["samples"], np.ndarray)

    def test_samples_full_is_ndarray(self, pmmh_results_1d):
        assert isinstance(pmmh_results_1d["samples_full"], np.ndarray)

    def test_samples_shape(self, pmmh_results_1d):
        # post-burnin only: 600 samples, 1 parameter
        assert pmmh_results_1d["samples"].shape == (600, 1)

    def test_samples_2d_shape(self, pmmh_results_2d):
        assert pmmh_results_2d["samples"].shape == (800, 2)

    def test_samples_full_length(self, pmmh_results_1d):
        # samples_full covers burnin + post-burnin
        assert pmmh_results_1d["samples_full"].shape[0] == 600 + 300

    def test_log_likelihood_trace_length(self, pmmh_results_1d):
        assert len(pmmh_results_1d["log_likelihood_trace"]) == 600 + 300

    def test_log_prior_trace_length(self, pmmh_results_1d):
        assert len(pmmh_results_1d["log_prior_trace"]) == 600 + 300

    def test_accept_trace_length(self, pmmh_results_1d):
        assert len(pmmh_results_1d["accept_trace"]) == 600 + 300

    def test_accept_rate_is_float(self, pmmh_results_1d):
        assert isinstance(pmmh_results_1d["accept_rate"], float)

    def test_accept_rate_post_burnin_is_float(self, pmmh_results_1d):
        assert isinstance(pmmh_results_1d["accept_rate_post_burnin"], float)

    def test_runtime_positive(self, pmmh_results_1d):
        assert pmmh_results_1d["runtime"] > 0.0

    def test_proposal_std_positive(self, pmmh_results_1d):
        assert np.all(pmmh_results_1d["proposal_std"] > 0)

    def test_accept_rate_in_unit_interval(self, pmmh_results_1d):
        assert 0.0 <= pmmh_results_1d["accept_rate"] <= 1.0

    def test_accept_rate_post_burnin_in_unit_interval(self, pmmh_results_1d):
        assert 0.0 <= pmmh_results_1d["accept_rate_post_burnin"] <= 1.0

    def test_accept_rate_post_burnin_consistent_with_trace(self, pmmh_results_1d):
        """accept_rate_post_burnin must match the mean of accept_trace[n_burnin:]."""
        trace = pmmh_results_1d["accept_trace"]
        n_burnin = 300
        expected = float(np.mean(trace[n_burnin:]))
        actual = pmmh_results_1d["accept_rate_post_burnin"]
        assert abs(actual - expected) < 1e-6

    def test_samples_are_finite(self, pmmh_results_2d):
        assert np.all(np.isfinite(pmmh_results_2d["samples"]))


# PMMH sampling quality tests
class TestPMMHSamplingQuality:
    """Verify PMMH recovers the prior when the likelihood is flat."""

    def test_1d_posterior_mean_within_3_se(self, pmmh_results_1d, gaussian_1d_prior_params):
        samples = pmmh_results_1d["samples"][:, 0]
        ess = compute_ess(pmmh_results_1d["samples"])
        se = gaussian_1d_prior_params["sigma"][0] / np.sqrt(max(ess, 1.0))
        error = abs(np.mean(samples) - gaussian_1d_prior_params["mu"][0])
        assert error < 3.0 * se, (
            f"PMMH mean {np.mean(samples):.4f} not within 3 SE "
            f"({3*se:.4f}) of true mean {gaussian_1d_prior_params['mu'][0]}"
        )

    def test_1d_posterior_std_within_20_percent(self, pmmh_results_1d, gaussian_1d_prior_params):
        std_est = float(np.std(pmmh_results_1d["samples"][:, 0]))
        std_true = float(gaussian_1d_prior_params["sigma"][0])
        assert abs(std_est - std_true) / std_true < 0.20, (
            f"PMMH std {std_est:.3f} deviates >20% from true {std_true:.3f}"
        )

    def test_2d_posterior_means_within_3_se(self, pmmh_results_2d, gaussian_2d_prior_params):
        samples = pmmh_results_2d["samples"]
        ess = compute_ess(samples)
        sigma = gaussian_2d_prior_params["sigma"]
        mu = gaussian_2d_prior_params["mu"]
        se = sigma / np.sqrt(max(ess, 1.0))
        errors = np.abs(np.mean(samples, axis=0) - mu)
        for j, (err, s) in enumerate(zip(errors, se)):
            assert err < 3.0 * s, (
                f"Parameter {j}: PMMH mean error {err:.4f} > 3 SE ({3*s:.4f})"
            )

    def test_acceptance_rate_reasonable(self, pmmh_results_1d):
        # Gaussian target with adaptation → expect something in [0.05, 0.99]
        ar = pmmh_results_1d["accept_rate_post_burnin"]
        assert 0.05 < ar < 0.99, f"Acceptance rate {ar:.3f} outside (0.05, 0.99)"


# Proposal adaptation tests
class TestPMMHProposalAdaptation:
    def test_adaptation_changes_proposal_std(self, gaussian_1d_prior_params):
        """With adapt_proposal=True, the final proposal_std should differ from initial."""
        p = gaussian_1d_prior_params
        initial_std = np.array([0.01])   # deliberately too small → will be scaled up

        def log_prior(theta):
            return _gaussian_log_prior(theta, p["mu"], p["sigma"])

        cfg = PMMHConfig(
            n_samples=200,
            n_burnin=200,
            proposal_std=initial_std.copy(),
            adapt_proposal=True,
            adapt_interval=20,
            verbose=False,
        )
        sampler = PMMHSampler(_flat_log_lik, log_prior, cfg)
        res = sampler.run(p["mu"].copy())
        assert not np.allclose(res["proposal_std"], initial_std), (
            "Proposal std was not adapted despite adapt_proposal=True"
        )

    def test_no_adaptation_preserves_proposal_std(self, gaussian_1d_prior_params):
        """With adapt_proposal=False, proposal_std should stay at its initial value."""
        p = gaussian_1d_prior_params
        initial_std = np.array([0.5])

        def log_prior(theta):
            return _gaussian_log_prior(theta, p["mu"], p["sigma"])

        cfg = PMMHConfig(
            n_samples=100,
            n_burnin=100,
            proposal_std=initial_std.copy(),
            adapt_proposal=False,
            verbose=False,
        )
        sampler = PMMHSampler(_flat_log_lik, log_prior, cfg)
        res = sampler.run(p["mu"].copy())
        assert np.allclose(res["proposal_std"], initial_std), (
            "Proposal std changed despite adapt_proposal=False"
        )

    def test_default_proposal_std_set_from_initial_params(self, gaussian_1d_prior_params):
        """When proposal_std is None, it should be initialised from initial_params."""
        p = gaussian_1d_prior_params

        def log_prior(theta):
            return _gaussian_log_prior(theta, p["mu"], p["sigma"])

        cfg = PMMHConfig(
            n_samples=100,
            n_burnin=50,
            proposal_std=None,   # ← let the sampler set it
            adapt_proposal=False,
            verbose=False,
        )
        sampler = PMMHSampler(_flat_log_lik, log_prior, cfg)
        res = sampler.run(p["mu"].copy())
        # Should be an array of positive values, not None
        assert res["proposal_std"] is not None
        assert np.all(res["proposal_std"] > 0)


# PMMH prior-rejection test
class TestPMMHPriorRejection:
    def test_invalid_prior_proposals_rejected(self):
        """When the prior returns -inf, proposals should be auto-rejected."""

        def log_lik(theta):
            return 0.0

        def log_prior(theta):
            # Only accept theta[0] > 0
            if theta[0] <= 0.0:
                return -np.inf
            return -0.5 * (theta[0] - 2.0) ** 2

        cfg = PMMHConfig(
            n_samples=200,
            n_burnin=100,
            proposal_std=np.array([5.0]),   # large proposals → many out-of-support
            adapt_proposal=False,
            verbose=False,
        )
        sampler = PMMHSampler(log_lik, log_prior, cfg)
        res = sampler.run(np.array([2.0]))
        # All stored samples must satisfy the prior support
        assert np.all(res["samples"][:, 0] > 0), (
            "PMMH kept samples outside prior support"
        )


# compute_ess  
class TestPMMHComputeESS:
    def test_iid_samples_ess_close_to_n(self):
        rng = np.random.default_rng(10)
        n = 2000
        samples = rng.standard_normal((n, 2))
        ess = compute_ess(samples)
        assert 0.70 * n < ess < 1.30 * n

    def test_correlated_samples_ess_less_than_n(self):
        rng = np.random.default_rng(11)
        n = 1000
        rho = 0.97
        x = np.zeros((n, 1))
        x[0] = rng.standard_normal()
        for i in range(1, n):
            x[i] = rho * x[i - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
        ess = compute_ess(x)
        assert ess < 0.15 * n

    def test_1d_array_input(self):
        rng = np.random.default_rng(12)
        x = rng.standard_normal(300)
        ess = compute_ess(x)
        assert ess > 0

    def test_ess_positive_on_pmmh_chain(self, pmmh_results_2d):
        ess = compute_ess(pmmh_results_2d["samples"])
        assert ess > 0


# compute_ess_per_second 
class TestPMMHComputeESSPerSecond:
    def test_equals_ess_over_runtime(self):
        rng = np.random.default_rng(13)
        samples = rng.standard_normal((300, 2))
        rt = 3.0
        ess = compute_ess(samples)
        ess_s = compute_ess_per_second(samples, rt)
        assert abs(ess_s - ess / rt) < 1e-6

    def test_positive(self):
        rng = np.random.default_rng(14)
        samples = rng.standard_normal((300, 2))
        assert compute_ess_per_second(samples, 2.0) > 0
