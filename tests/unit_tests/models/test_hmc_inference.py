"""
Unit tests for HMC
"""

import numpy as np
import pytest
import tensorflow as tf

tfp = pytest.importorskip(
    "tensorflow_probability",
    exc_type=ImportError,
    reason="TensorFlow Probability is unavailable or incompatible with the installed TensorFlow version.",
)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from models.HMC_inference import HMCSampler, HMCConfig, compute_ess, compute_ess_per_second

pytestmark = pytest.mark.tensorflow


# Shared fixtures
@pytest.fixture(scope="module")
def gaussian_1d_target():
    """1-D Gaussian N(3, 2²).  Analytically known mean and std."""
    mu, sigma = 3.0, 2.0

    @tf.function
    def log_posterior(theta):
        return -0.5 * ((theta[0] - mu) / sigma) ** 2

    return {"log_posterior": log_posterior, "mu": mu, "sigma": sigma}


@pytest.fixture(scope="module")
def gaussian_2d_target():
    """2-D Gaussian N([1, -1], diag([1, 4])).  Independent components."""
    mu = tf.constant([1.0, -1.0], dtype=tf.float32)
    sigma = tf.constant([1.0, 2.0], dtype=tf.float32)

    @tf.function
    def log_posterior(theta):
        return -0.5 * tf.reduce_sum(((theta - mu) / sigma) ** 2)

    return {
        "log_posterior": log_posterior,
        "mu": mu.numpy(),
        "sigma": sigma.numpy(),
    }


@pytest.fixture(scope="module")
def hmc_results_1d(gaussian_1d_target):
    """Run a short HMC chain on the 1-D target once; reuse across tests."""
    cfg = HMCConfig(
        n_samples=600,
        n_burnin=300,
        step_size=0.5,
        n_leapfrog_steps=5,
        adapt_step_size=True,
        verbose=False,
    )
    sampler = HMCSampler(gaussian_1d_target["log_posterior"], cfg)
    return sampler.run(np.array([0.0], dtype=np.float32))


@pytest.fixture(scope="module")
def hmc_results_2d(gaussian_2d_target):
    """Run a short HMC chain on the 2-D target once; reuse across tests."""
    cfg = HMCConfig(
        n_samples=800,
        n_burnin=400,
        step_size=0.3,
        n_leapfrog_steps=5,
        adapt_step_size=True,
        verbose=False,
    )
    sampler = HMCSampler(gaussian_2d_target["log_posterior"], cfg)
    return sampler.run(np.array([0.0, 0.0], dtype=np.float32))


# HMCConfig tests
class TestHMCConfig:
    def test_default_n_samples(self):
        assert HMCConfig().n_samples == 1000

    def test_default_n_burnin(self):
        assert HMCConfig().n_burnin == 500

    def test_default_n_leapfrog_steps(self):
        assert HMCConfig().n_leapfrog_steps == 10

    def test_default_adapt_step_size(self):
        assert HMCConfig().adapt_step_size is True

    def test_default_target_accept_prob(self):
        assert 0.0 < HMCConfig().target_accept_prob < 1.0

    def test_default_verbose(self):
        assert isinstance(HMCConfig().verbose, bool)

    def test_custom_values_stored(self):
        cfg = HMCConfig(n_samples=42, n_burnin=7, step_size=0.123,
                        n_leapfrog_steps=3, adapt_step_size=False,
                        target_accept_prob=0.8, verbose=False)
        assert cfg.n_samples == 42
        assert cfg.n_burnin == 7
        assert abs(cfg.step_size - 0.123) < 1e-9
        assert cfg.n_leapfrog_steps == 3
        assert cfg.adapt_step_size is False
        assert abs(cfg.target_accept_prob - 0.8) < 1e-9
        assert cfg.verbose is False


# HMCSampler output contract tests
class TestHMCSamplerOutputContract:
    """Verify the dict returned by HMCSampler.run() has the right structure."""

    REQUIRED_KEYS = {"samples", "is_accepted", "accept_rate", "runtime"}

    def test_result_keys_present(self, hmc_results_1d):
        assert self.REQUIRED_KEYS.issubset(hmc_results_1d.keys())

    def test_samples_is_ndarray(self, hmc_results_1d):
        assert isinstance(hmc_results_1d["samples"], np.ndarray)

    def test_is_accepted_is_ndarray(self, hmc_results_1d):
        assert isinstance(hmc_results_1d["is_accepted"], np.ndarray)

    def test_accept_rate_is_float(self, hmc_results_1d):
        assert isinstance(hmc_results_1d["accept_rate"], float)

    def test_runtime_is_positive(self, hmc_results_1d):
        assert hmc_results_1d["runtime"] > 0.0

    def test_samples_shape_1d(self, hmc_results_1d):
        # 600 samples, 1 parameter
        assert hmc_results_1d["samples"].shape == (600, 1)

    def test_samples_shape_2d(self, hmc_results_2d):
        # 800 samples, 2 parameters
        assert hmc_results_2d["samples"].shape == (800, 2)

    def test_is_accepted_shape_matches_n_samples(self, hmc_results_1d):
        assert hmc_results_1d["is_accepted"].shape == (600,)

    def test_is_accepted_dtype_bool_or_int(self, hmc_results_1d):
        dtype = hmc_results_1d["is_accepted"].dtype
        assert np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_)

    def test_accept_rate_in_unit_interval(self, hmc_results_1d):
        ar = hmc_results_1d["accept_rate"]
        assert 0.0 <= ar <= 1.0

    def test_accept_rate_consistent_with_is_accepted(self, hmc_results_1d):
        expected = float(np.mean(hmc_results_1d["is_accepted"]))
        assert abs(hmc_results_1d["accept_rate"] - expected) < 1e-6

# HMC sampling quality tests
class TestHMCSamplingQuality:
    """Verify HMC recovers the correct posterior on analytically known targets."""

    def test_1d_posterior_mean_within_3_se(self, hmc_results_1d, gaussian_1d_target):
        samples = hmc_results_1d["samples"][:, 0]
        ess = compute_ess(hmc_results_1d["samples"])
        se = gaussian_1d_target["sigma"] / np.sqrt(max(ess, 1.0))
        error = abs(np.mean(samples) - gaussian_1d_target["mu"])
        assert error < 3.0 * se, (
            f"HMC mean {np.mean(samples):.4f} not within 3 SE "
            f"({3*se:.4f}) of true mean {gaussian_1d_target['mu']}"
        )

    def test_1d_posterior_std_within_20_percent(self, hmc_results_1d, gaussian_1d_target):
        std_est = float(np.std(hmc_results_1d["samples"][:, 0]))
        std_true = gaussian_1d_target["sigma"]
        assert abs(std_est - std_true) / std_true < 0.20, (
            f"HMC std {std_est:.3f} deviates >20% from true {std_true:.3f}"
        )

    def test_2d_posterior_means_within_3_se(self, hmc_results_2d, gaussian_2d_target):
        samples = hmc_results_2d["samples"]
        ess = compute_ess(samples)
        se = gaussian_2d_target["sigma"] / np.sqrt(max(ess, 1.0))
        errors = np.abs(np.mean(samples, axis=0) - gaussian_2d_target["mu"])
        for j, (err, s) in enumerate(zip(errors, se)):
            assert err < 3.0 * s, (
                f"Parameter {j}: HMC mean error {err:.4f} > 3 SE ({3*s:.4f})"
            )

    def test_acceptance_rate_reasonable(self, hmc_results_1d):
        # With step-size adaptation, acceptance should converge near target
        # Allow a wide band [0.1, 0.99] to avoid flakiness
        assert 0.10 < hmc_results_1d["accept_rate"] < 0.99

    def test_samples_are_finite(self, hmc_results_2d):
        assert np.all(np.isfinite(hmc_results_2d["samples"]))


# HMC configuration variant tests
class TestHMCSamplerConfigVariants:
    """Check that different config options run without error."""

    def test_no_adaptation(self, gaussian_1d_target):
        """adapt_step_size=False should still run and return valid results."""
        cfg = HMCConfig(
            n_samples=100,
            n_burnin=50,
            step_size=0.3,
            n_leapfrog_steps=3,
            adapt_step_size=False,
            verbose=False,
        )
        sampler = HMCSampler(gaussian_1d_target["log_posterior"], cfg)
        res = sampler.run(np.array([0.0], dtype=np.float32))
        assert res["samples"].shape == (100, 1)
        assert 0.0 <= res["accept_rate"] <= 1.0

    def test_single_leapfrog_step(self, gaussian_1d_target):
        """n_leapfrog_steps=1 (Metropolis-adjusted Langevin-like) should work."""
        cfg = HMCConfig(
            n_samples=100,
            n_burnin=50,
            n_leapfrog_steps=1,
            verbose=False,
        )
        sampler = HMCSampler(gaussian_1d_target["log_posterior"], cfg)
        res = sampler.run(np.array([0.0], dtype=np.float32))
        assert res["samples"].shape == (100, 1)

    def test_verbose_false_no_error(self, gaussian_1d_target):
        cfg = HMCConfig(n_samples=50, n_burnin=25, verbose=False)
        sampler = HMCSampler(gaussian_1d_target["log_posterior"], cfg)
        res = sampler.run(np.array([0.0], dtype=np.float32))
        assert "samples" in res

    def test_default_config_used_when_none_passed(self, gaussian_1d_target):
        """Passing config=None should use HMCConfig() defaults without error."""
        sampler = HMCSampler(gaussian_1d_target["log_posterior"], config=None)
        # Just verify the sampler is constructed and config is set
        assert sampler.config.n_samples == 1000


# compute_ess tests
class TestComputeESS:
    def test_iid_samples_ess_close_to_n(self):
        rng = np.random.default_rng(0)
        n = 2000
        samples = rng.standard_normal((n, 2))
        ess = compute_ess(samples)
        # For iid samples ESS ≈ n; allow 30% tolerance
        assert 0.70 * n < ess < 1.30 * n, f"iid ESS {ess:.1f} not close to {n}"

    def test_autocorrelated_samples_ess_much_less_than_n(self):
        rng = np.random.default_rng(1)
        n = 1000
        rho = 0.97          # very high autocorrelation
        x = np.zeros((n, 1))
        x[0] = rng.standard_normal()
        for i in range(1, n):
            x[i] = rho * x[i - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
        ess = compute_ess(x)
        # Theoretical ESS ≈ n*(1-rho)/(1+rho) ≈ 15; must be << n
        assert ess < 0.15 * n, f"autocorrelated ESS {ess:.1f} should be << {n}"

    def test_1d_input_handled(self):
        rng = np.random.default_rng(2)
        samples_1d = rng.standard_normal(500)
        ess = compute_ess(samples_1d)
        assert ess > 0

    def test_2d_input_shape(self):
        rng = np.random.default_rng(3)
        samples = rng.standard_normal((300, 3))
        ess = compute_ess(samples)
        assert isinstance(ess, (float, np.floating))
        assert ess > 0

    def test_ess_positive(self, hmc_results_2d):
        ess = compute_ess(hmc_results_2d["samples"])
        assert ess > 0

    def test_ess_at_most_n(self, hmc_results_2d):
        n = hmc_results_2d["samples"].shape[0]
        ess = compute_ess(hmc_results_2d["samples"])
        # ESS should not exceed N (with small floating-point tolerance)
        assert ess <= n * 1.05


# compute_ess_per_second tests
class TestComputeESSPerSecond:
    def test_equals_ess_divided_by_runtime(self):
        rng = np.random.default_rng(4)
        samples = rng.standard_normal((200, 2))
        runtime = 5.0
        ess = compute_ess(samples)
        ess_per_s = compute_ess_per_second(samples, runtime)
        assert abs(ess_per_s - ess / runtime) < 1e-6

    def test_positive(self):
        rng = np.random.default_rng(5)
        samples = rng.standard_normal((200, 2))
        assert compute_ess_per_second(samples, 1.0) > 0

    def test_scales_with_runtime(self):
        rng = np.random.default_rng(6)
        samples = rng.standard_normal((200, 2))
        ess1 = compute_ess_per_second(samples, 1.0)
        ess10 = compute_ess_per_second(samples, 10.0)
        assert abs(ess1 / ess10 - 10.0) < 1e-6
