"""Unit tests for models/tf_ssm_models.py.

Covers
------
* LGSSM_TFP.log_prob     – matches lgssm_log_likelihood, finite
* LGSSM_TFP.filter       – returns KFResultsTF with correct shapes
* NonlinSSM_TFP.log_prob  – finite, gradient exists
* NonlinSSM_TFP.log_prob_tf – @tf.function call works
* make_lgssm_hmc_target   – callable, finite value, gradient exists
* make_nonlinssm_hmc_target – callable, finite, gradient exists
* run_hmc                 – correct sample shapes, acceptance dtype
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import tensorflow as tf

from models.tf_core import KFResultsTF, lgssm_log_likelihood
from models.tf_ssm_models import (
    LGSSM_TFP,
    NonlinSSM_TFP,
    make_lgssm_hmc_target,
    make_nonlinssm_hmc_target,
    run_hmc,
)

F32 = tf.float32


def _np2tf(x):
    return tf.constant(x, dtype=F32)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lgssm_mats():
    nx, ny = 2, 1
    F   = np.array([[0.9, 0.2], [0.0, 0.7]], dtype=np.float32)
    H   = np.array([[1.0, 0.0]], dtype=np.float32)
    Q   = np.diag([0.05, 0.02]).astype(np.float32)
    R   = np.array([[0.10]], dtype=np.float32)
    m0  = np.zeros(nx, dtype=np.float32)
    P0  = np.eye(nx, dtype=np.float32)
    return dict(nx=nx, ny=ny, F=F, H=H, Q=Q, R=R, m0=m0, P0=P0)


@pytest.fixture(scope="module")
def obs_lgssm(lgssm_mats):
    rng = np.random.default_rng(7)
    T   = 25
    return rng.normal(size=(T, lgssm_mats["ny"])).astype(np.float32)


@pytest.fixture(scope="module")
def obs_nonlin():
    rng = np.random.default_rng(9)
    T   = 30
    return rng.normal(scale=2.0, size=T).astype(np.float32)


# ---------------------------------------------------------------------------
# LGSSM_TFP
# ---------------------------------------------------------------------------

class TestLGSSMTFP:

    def test_log_prob_scalar(self, lgssm_mats, obs_lgssm):
        p = lgssm_mats
        model = LGSSM_TFP(p["F"], p["H"], p["Q"], p["R"], p["m0"], p["P0"])
        ll = model.log_prob(_np2tf(obs_lgssm))
        assert ll.shape == (), f"Expected scalar, got {ll.shape}"

    def test_log_prob_finite(self, lgssm_mats, obs_lgssm):
        p = lgssm_mats
        model = LGSSM_TFP(p["F"], p["H"], p["Q"], p["R"], p["m0"], p["P0"])
        ll = float(model.log_prob(_np2tf(obs_lgssm)))
        assert math.isfinite(ll), f"log_prob is not finite: {ll}"

    def test_log_prob_matches_lgssm_log_likelihood(self, lgssm_mats, obs_lgssm):
        """LGSSM_TFP.log_prob must equal lgssm_log_likelihood exactly."""
        p = lgssm_mats
        model = LGSSM_TFP(p["F"], p["H"], p["Q"], p["R"], p["m0"], p["P0"])

        Y_tf = _np2tf(obs_lgssm)
        ll_model = float(model.log_prob(Y_tf))
        ll_fn    = float(lgssm_log_likelihood(
            Y_tf,
            _np2tf(p["F"]), _np2tf(p["H"]),
            _np2tf(p["Q"]), _np2tf(p["R"]),
            _np2tf(p["m0"]), _np2tf(p["P0"]),
        ))
        assert abs(ll_model - ll_fn) < 1e-4, (
            f"LGSSM_TFP.log_prob={ll_model:.6f} vs lgssm_log_likelihood={ll_fn:.6f}"
        )

    def test_filter_returns_kfresultstf(self, lgssm_mats, obs_lgssm):
        p = lgssm_mats
        model = LGSSM_TFP(p["F"], p["H"], p["Q"], p["R"], p["m0"], p["P0"])
        res = model.filter(_np2tf(obs_lgssm))
        assert isinstance(res, KFResultsTF)

    def test_filter_x_filt_shape(self, lgssm_mats, obs_lgssm):
        p = lgssm_mats
        T  = obs_lgssm.shape[0]
        model = LGSSM_TFP(p["F"], p["H"], p["Q"], p["R"], p["m0"], p["P0"])
        res = model.filter(_np2tf(obs_lgssm))
        assert res.x_filt.shape == (T, p["nx"]), f"x_filt: {res.x_filt.shape}"

    def test_filter_P_filt_shape(self, lgssm_mats, obs_lgssm):
        p = lgssm_mats
        T  = obs_lgssm.shape[0]
        model = LGSSM_TFP(p["F"], p["H"], p["Q"], p["R"], p["m0"], p["P0"])
        res = model.filter(_np2tf(obs_lgssm))
        assert res.P_filt.shape == (T, p["nx"], p["nx"])

    def test_filter_loglik_matches_log_prob(self, lgssm_mats, obs_lgssm):
        """filter().loglik must equal log_prob()."""
        p = lgssm_mats
        model = LGSSM_TFP(p["F"], p["H"], p["Q"], p["R"], p["m0"], p["P0"])
        Y_tf = _np2tf(obs_lgssm)
        ll_filter = float(model.filter(Y_tf).loglik)
        ll_prob   = float(model.log_prob(Y_tf))
        assert abs(ll_filter - ll_prob) < 1e-4

    def test_log_prob_gradient_exists(self, lgssm_mats, obs_lgssm):
        """Gradient of LGSSM_TFP.log_prob w.r.t. Q must be finite."""
        p = lgssm_mats
        Q_var = tf.Variable(_np2tf(p["Q"]))
        with tf.GradientTape() as tape:
            model = LGSSM_TFP(p["F"], p["H"], Q_var, p["R"], p["m0"], p["P0"])
            ll    = model.log_prob(_np2tf(obs_lgssm))
        grad = tape.gradient(ll, Q_var)
        assert grad is not None, "Gradient w.r.t. Q is None"
        assert tf.reduce_all(tf.math.is_finite(grad)), f"Non-finite gradient: {grad}"

    def test_stores_tensors_as_f32(self, lgssm_mats):
        p = lgssm_mats
        model = LGSSM_TFP(p["F"], p["H"], p["Q"], p["R"], p["m0"], p["P0"])
        for attr in ["F", "H", "Q", "R", "m0", "P0"]:
            t = getattr(model, attr)
            assert t.dtype == F32, f"{attr}.dtype={t.dtype}, expected float32"


# ---------------------------------------------------------------------------
# NonlinSSM_TFP
# ---------------------------------------------------------------------------

class TestNonlinSSMTFP:

    def test_log_prob_scalar(self, obs_nonlin):
        model = NonlinSSM_TFP(sigma_v=1.0, sigma_w=1.0, T=len(obs_nonlin))
        ll = model.log_prob(_np2tf(obs_nonlin))
        assert ll.shape == (), f"Expected scalar, got {ll.shape}"

    def test_log_prob_finite(self, obs_nonlin):
        model = NonlinSSM_TFP(sigma_v=1.0, sigma_w=1.0, T=len(obs_nonlin))
        ll = float(model.log_prob(_np2tf(obs_nonlin)))
        assert math.isfinite(ll), f"log_prob not finite: {ll}"

    def test_log_prob_tf_callable(self, obs_nonlin):
        """@tf.function decorated log_prob_tf should be callable directly."""
        model = NonlinSSM_TFP(sigma_v=1.0, sigma_w=1.0, T=len(obs_nonlin))
        sv = tf.constant(1.0, dtype=F32)
        sw = tf.constant(1.0, dtype=F32)
        ll = model.log_prob_tf(_np2tf(obs_nonlin), sv, sw)
        assert ll.shape == ()

    def test_log_prob_tf_varies_with_params(self, obs_nonlin):
        """LL should change when sigma_v is changed."""
        model = NonlinSSM_TFP(T=len(obs_nonlin))
        Y_tf = _np2tf(obs_nonlin)
        sw = tf.constant(1.0, dtype=F32)
        ll1 = float(model.log_prob_tf(Y_tf, tf.constant(0.5, dtype=F32), sw))
        ll2 = float(model.log_prob_tf(Y_tf, tf.constant(2.0, dtype=F32), sw))
        assert ll1 != ll2, "LL should vary with sigma_v"

    def test_log_prob_gradient_wrt_sigma_v(self, obs_nonlin):
        """Gradient w.r.t. log_sigma_v must exist and be finite."""
        model = NonlinSSM_TFP(T=len(obs_nonlin))
        Y_tf  = _np2tf(obs_nonlin)
        log_sv = tf.Variable(tf.constant(0.0, dtype=F32))
        with tf.GradientTape() as tape:
            sv = tf.exp(log_sv)
            sw = tf.constant(1.0, dtype=F32)
            ll = model.log_prob_tf(Y_tf, sv, sw)
        grad = tape.gradient(ll, log_sv)
        assert grad is not None, "Gradient w.r.t. log_sigma_v is None"
        assert math.isfinite(float(grad)), f"Non-finite gradient: {grad}"


# ---------------------------------------------------------------------------
# make_lgssm_hmc_target
# ---------------------------------------------------------------------------

class TestMakeLGSSMHMCTarget:

    def test_returns_callable(self, lgssm_mats, obs_lgssm):
        p = lgssm_mats
        target = make_lgssm_hmc_target(
            obs_lgssm, p["F"], p["H"], p["R"], p["m0"], p["P0"], p["nx"]
        )
        assert callable(target)

    def test_callable_returns_scalar(self, lgssm_mats, obs_lgssm):
        p = lgssm_mats
        target = make_lgssm_hmc_target(
            obs_lgssm, p["F"], p["H"], p["R"], p["m0"], p["P0"], p["nx"]
        )
        nx = p["nx"]
        n_lower = nx * (nx + 1) // 2
        log_L_flat = tf.zeros(n_lower, dtype=F32)
        val = target(log_L_flat)
        assert val.shape == (), f"Expected scalar, got {val.shape}"

    def test_target_value_finite(self, lgssm_mats, obs_lgssm):
        p = lgssm_mats
        target = make_lgssm_hmc_target(
            obs_lgssm, p["F"], p["H"], p["R"], p["m0"], p["P0"], p["nx"]
        )
        n_lower = p["nx"] * (p["nx"] + 1) // 2
        log_L_flat = tf.zeros(n_lower, dtype=F32)
        val = float(target(log_L_flat))
        assert math.isfinite(val), f"Target not finite: {val}"

    def test_gradient_exists(self, lgssm_mats, obs_lgssm):
        p = lgssm_mats
        target = make_lgssm_hmc_target(
            obs_lgssm, p["F"], p["H"], p["R"], p["m0"], p["P0"], p["nx"]
        )
        n_lower = p["nx"] * (p["nx"] + 1) // 2
        theta = tf.Variable(tf.zeros(n_lower, dtype=F32))
        with tf.GradientTape() as tape:
            val = target(theta)
        grad = tape.gradient(val, theta)
        assert grad is not None, "Gradient is None"
        assert tf.reduce_all(tf.math.is_finite(grad)), f"Non-finite gradient: {grad}"


# ---------------------------------------------------------------------------
# make_nonlinssm_hmc_target
# ---------------------------------------------------------------------------

class TestMakeNonlinSSMHMCTarget:

    def test_returns_callable(self, obs_nonlin):
        target = make_nonlinssm_hmc_target(obs_nonlin)
        assert callable(target)

    def test_callable_returns_scalar(self, obs_nonlin):
        target = make_nonlinssm_hmc_target(obs_nonlin)
        val = target(tf.constant(0.0, F32), tf.constant(0.0, F32))
        assert val.shape == ()

    def test_target_finite(self, obs_nonlin):
        target = make_nonlinssm_hmc_target(obs_nonlin)
        val = float(target(tf.constant(0.0, F32), tf.constant(0.0, F32)))
        assert math.isfinite(val)

    def test_gradient_exists(self, obs_nonlin):
        target = make_nonlinssm_hmc_target(obs_nonlin)
        lsv = tf.Variable(tf.constant(0.0, dtype=F32))
        lsw = tf.Variable(tf.constant(0.0, dtype=F32))
        with tf.GradientTape() as tape:
            val = target(lsv, lsw)
        grads = tape.gradient(val, [lsv, lsw])
        for i, g in enumerate(grads):
            assert g is not None, f"Gradient {i} is None"
            assert math.isfinite(float(g)), f"Gradient {i} not finite: {g}"


# ---------------------------------------------------------------------------
# run_hmc  (short chain for speed)
# ---------------------------------------------------------------------------

class TestRunHMC:
    """Smoke tests for run_hmc with a trivial target (Normal(0,1))."""

    @pytest.fixture(scope="class")
    def hmc_results(self):
        """Run a very short HMC chain on a 1-D standard Normal target."""
        def target(x):
            return -0.5 * x * x  # log p(x) ∝ -x²/2

        init = [tf.constant(0.0, dtype=F32)]
        samples, is_accepted = run_hmc(
            target_log_prob_fn = target,
            init_state         = init,
            num_results        = 20,
            num_burnin         = 10,
            step_size          = 0.3,
            num_leapfrog       = 3,
        )
        return samples, is_accepted

    def test_samples_shape(self, hmc_results):
        samples, _ = hmc_results
        # samples is a list with one element: shape (num_results,)
        assert samples[0].shape == (20,), f"samples shape: {samples[0].shape}"

    def test_acceptance_shape(self, hmc_results):
        _, is_accepted = hmc_results
        assert is_accepted.shape == (20,), f"is_accepted shape: {is_accepted.shape}"

    def test_acceptance_dtype_bool(self, hmc_results):
        _, is_accepted = hmc_results
        assert is_accepted.dtype == tf.bool, f"dtype: {is_accepted.dtype}"

    def test_samples_finite(self, hmc_results):
        samples, _ = hmc_results
        assert tf.reduce_all(tf.math.is_finite(samples[0])), "HMC samples contain non-finite values"

    def test_acceptance_rate_positive(self, hmc_results):
        _, is_accepted = hmc_results
        acc_rate = float(tf.reduce_mean(tf.cast(is_accepted, F32)))
        assert acc_rate > 0.0, f"Zero acceptance rate: {acc_rate}"

    def test_lgssm_hmc_runs_without_error(self, lgssm_mats, obs_lgssm):
        """Integration: HMC target for LGSSM runs a short chain without NaN."""
        p = lgssm_mats
        target = make_lgssm_hmc_target(
            obs_lgssm, p["F"], p["H"], p["R"], p["m0"], p["P0"], p["nx"]
        )
        n_lower = p["nx"] * (p["nx"] + 1) // 2
        init = [tf.zeros(n_lower, dtype=F32)]
        samples, is_accepted = run_hmc(
            target_log_prob_fn = target,
            init_state         = init,
            num_results        = 10,
            num_burnin         = 5,
            step_size          = 0.05,
            num_leapfrog       = 3,
        )
        assert tf.reduce_all(tf.math.is_finite(samples[0])), "LGSSM HMC samples contain NaN/Inf"
