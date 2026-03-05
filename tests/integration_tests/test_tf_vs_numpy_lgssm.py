"""Integration tests for the TF filter implementations.

These tests use a *simulated* LGSSM trajectory (not just random noise) so we
can check that the numerical outputs are physically sensible.

Covers
------
1. NumPy KF vs TF KF end-to-end (log-likelihood, x_filt, P_filt agreement)
2. XLA vs standard lgssm_log_likelihood agreement
3. Gradient of lgssm_log_likelihood w.r.t. Q is finite and non-zero
4. ParticleFilterTF mean tracks KF mean on simulated LGSSM data
5. LGSSM_TFP agrees with lgssm_log_likelihood on simulated data
6. HMC chain on NonlinSSM target runs without NaN and has positive acceptance
7. run_hmc posterior mean for standard Normal target is near 0
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import tensorflow as tf

from models.kalman_filter import kalman_filter_general
from models.tf_core import KFResultsTF, kalman_filter_tf, lgssm_log_likelihood
from models.tf_particle_filter import ParticleFilterTF, run_particle_filter_tf
from models.tf_ssm_models import (
    LGSSM_TFP,
    NonlinSSM_TFP,
    make_lgssm_hmc_target,
    make_nonlinssm_hmc_target,
    run_hmc,
)

F32 = tf.float32


def _np2tf(x):
    return tf.constant(np.asarray(x, dtype=np.float32), dtype=F32)


# ---------------------------------------------------------------------------
# Shared fixtures: simulate a proper LGSSM trajectory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lgssm_system():
    nx, ny = 2, 1
    Phi = np.array([[0.9, 0.2], [0.0, 0.7]], dtype=np.float32)
    H   = np.array([[1.0, 0.0]], dtype=np.float32)
    Q   = np.diag([0.05, 0.02]).astype(np.float32)
    R   = np.array([[0.10]], dtype=np.float32)
    x0  = np.zeros(nx, dtype=np.float32)
    P0  = np.eye(nx, dtype=np.float32)
    return dict(nx=nx, ny=ny, Phi=Phi, H=H, Q=Q, R=R, x0=x0, P0=P0)


@pytest.fixture(scope="module")
def simulated_lgssm(lgssm_system):
    """Simulate T=50 steps from the LGSSM using the system matrices."""
    p   = lgssm_system
    rng = np.random.default_rng(2024)
    T   = 50

    L_Q = np.linalg.cholesky(p["Q"])
    L_R = np.linalg.cholesky(p["R"])

    X = np.zeros((T, p["nx"]), dtype=np.float32)
    Y = np.zeros((T, p["ny"]), dtype=np.float32)

    x = p["x0"].copy()
    for t in range(T):
        x = p["Phi"] @ x + L_Q @ rng.standard_normal(p["nx"])
        y = p["H"] @ x   + L_R @ rng.standard_normal(p["ny"])
        X[t] = x
        Y[t] = y

    return dict(X=X, Y=Y, T=T)


@pytest.fixture(scope="module")
def numpy_kf_result(lgssm_system, simulated_lgssm):
    p = lgssm_system
    d = simulated_lgssm
    Gamma = np.eye(p["nx"], dtype=np.float32)
    return kalman_filter_general(
        Y=d["Y"], Phi=p["Phi"], H=p["H"],
        Gamma=Gamma, Q=p["Q"], R=p["R"],
        x0=p["x0"], P0=p["P0"],
    )


@pytest.fixture(scope="module")
def tf_kf_result(lgssm_system, simulated_lgssm):
    p = lgssm_system
    d = simulated_lgssm
    return kalman_filter_tf(
        _np2tf(d["Y"]),
        _np2tf(p["Phi"]), _np2tf(p["H"]),
        _np2tf(p["Q"]),   _np2tf(p["R"]),
        _np2tf(p["x0"]),  _np2tf(p["P0"]),
    )


# ---------------------------------------------------------------------------
# 1. NumPy KF vs TF KF end-to-end
# ---------------------------------------------------------------------------

class TestNumpyVsTFKalmanFilter:

    def test_loglik_agreement(self, numpy_kf_result, tf_kf_result):
        """Log-likelihoods must agree to < 0.05 nats."""
        ll_np = float(numpy_kf_result.loglik)
        ll_tf = float(tf_kf_result.loglik)
        assert abs(ll_np - ll_tf) < 0.05, (
            f"NumPy LL={ll_np:.4f}, TF LL={ll_tf:.4f}, diff={abs(ll_np-ll_tf):.4e}"
        )

    def test_x_filt_agreement(self, numpy_kf_result, tf_kf_result):
        """Filtered means must agree to < 5e-3 (float32 budget)."""
        max_diff = np.max(np.abs(numpy_kf_result.x_filt - tf_kf_result.x_filt.numpy()))
        assert max_diff < 5e-3, f"x_filt max diff={max_diff:.2e}"

    def test_P_filt_agreement(self, numpy_kf_result, tf_kf_result):
        """Filtered covariances must agree to < 5e-3."""
        max_diff = np.max(np.abs(numpy_kf_result.P_filt - tf_kf_result.P_filt.numpy()))
        assert max_diff < 5e-3, f"P_filt max diff={max_diff:.2e}"

    def test_innov_agreement(self, numpy_kf_result, tf_kf_result):
        max_diff = np.max(np.abs(numpy_kf_result.innov - tf_kf_result.innov.numpy()))
        assert max_diff < 5e-3, f"innov max diff={max_diff:.2e}"

    def test_tf_kf_rmse_lower_than_prior(self, lgssm_system, simulated_lgssm, tf_kf_result):
        """TF KF filtered state RMSE must be lower than the prediction RMSE."""
        p = lgssm_system
        X = simulated_lgssm["X"]
        rmse_filt = float(np.sqrt(np.mean((X - tf_kf_result.x_filt.numpy()) ** 2)))
        rmse_pred = float(np.sqrt(np.mean((X - tf_kf_result.x_pred.numpy()) ** 2)))
        assert rmse_filt < rmse_pred, (
            f"RMSE filt={rmse_filt:.4f} should be < RMSE pred={rmse_pred:.4f}"
        )

    def test_kf_loglik_matches_lgssm_ll_fn(self, lgssm_system, simulated_lgssm):
        """kalman_filter_tf.loglik must equal lgssm_log_likelihood."""
        p = lgssm_system
        d = simulated_lgssm
        args = (
            _np2tf(d["Y"]),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),   _np2tf(p["R"]),
            _np2tf(p["x0"]),  _np2tf(p["P0"]),
        )
        ll_fn = float(lgssm_log_likelihood(*args))
        ll_kf = float(kalman_filter_tf(*args).loglik)
        assert abs(ll_fn - ll_kf) < 1e-4

    def test_lgssm_tfp_log_prob_matches_kf(self, lgssm_system, simulated_lgssm):
        """LGSSM_TFP.log_prob must equal kalman_filter_tf.loglik."""
        p = lgssm_system
        d = simulated_lgssm
        model = LGSSM_TFP(p["Phi"], p["H"], p["Q"], p["R"], p["x0"], p["P0"])
        ll_tfp = float(model.log_prob(_np2tf(d["Y"])))
        ll_kf  = float(kalman_filter_tf(
            _np2tf(d["Y"]),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),   _np2tf(p["R"]),
            _np2tf(p["x0"]),  _np2tf(p["P0"]),
        ).loglik)
        assert abs(ll_tfp - ll_kf) < 1e-4


# ---------------------------------------------------------------------------
# 2. XLA vs standard lgssm_log_likelihood
# ---------------------------------------------------------------------------

class TestXLAConsistency:

    @pytest.fixture(scope="class")
    def ll_xla(self):
        @tf.function(jit_compile=True)
        def _ll(Y, F, H, Q, R, m0, P0):
            return lgssm_log_likelihood(Y, F, H, Q, R, m0, P0)
        return _ll

    def test_xla_matches_standard(self, lgssm_system, simulated_lgssm, ll_xla):
        p = lgssm_system
        d = simulated_lgssm
        args = (
            _np2tf(d["Y"]),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),   _np2tf(p["R"]),
            _np2tf(p["x0"]),  _np2tf(p["P0"]),
        )
        ll_std = float(lgssm_log_likelihood(*args))
        ll_x   = float(ll_xla(*args))
        assert abs(ll_std - ll_x) < 1e-3, (
            f"XLA={ll_x:.6f} vs standard={ll_std:.6f}"
        )

    def test_xla_output_finite(self, lgssm_system, simulated_lgssm, ll_xla):
        p = lgssm_system
        d = simulated_lgssm
        ll = float(ll_xla(
            _np2tf(d["Y"]),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),   _np2tf(p["R"]),
            _np2tf(p["x0"]),  _np2tf(p["P0"]),
        ))
        assert math.isfinite(ll)


# ---------------------------------------------------------------------------
# 3. Gradient of lgssm_log_likelihood w.r.t. Q
# ---------------------------------------------------------------------------

class TestGradientWRTQ:

    def test_gradient_finite_and_nonzero(self, lgssm_system, simulated_lgssm):
        p = lgssm_system
        d = simulated_lgssm
        Q_var = tf.Variable(_np2tf(p["Q"]))
        with tf.GradientTape() as tape:
            ll = lgssm_log_likelihood(
                _np2tf(d["Y"]),
                _np2tf(p["Phi"]), _np2tf(p["H"]),
                Q_var, _np2tf(p["R"]),
                _np2tf(p["x0"]),  _np2tf(p["P0"]),
            )
        grad = tape.gradient(ll, Q_var)
        assert grad is not None, "Gradient w.r.t. Q is None"
        assert tf.reduce_all(tf.math.is_finite(grad)), "Gradient has non-finite values"
        # At least one element must be non-zero
        assert tf.reduce_any(tf.abs(grad) > 1e-10), "All gradient elements are zero"

    def test_gradient_wrt_F(self, lgssm_system, simulated_lgssm):
        p = lgssm_system
        d = simulated_lgssm
        F_var = tf.Variable(_np2tf(p["Phi"]))
        with tf.GradientTape() as tape:
            ll = lgssm_log_likelihood(
                _np2tf(d["Y"]),
                F_var, _np2tf(p["H"]),
                _np2tf(p["Q"]), _np2tf(p["R"]),
                _np2tf(p["x0"]), _np2tf(p["P0"]),
            )
        grad = tape.gradient(ll, F_var)
        assert grad is not None, "Gradient w.r.t. F is None"
        assert tf.reduce_all(tf.math.is_finite(grad)), "Gradient has non-finite values"


# ---------------------------------------------------------------------------
# 4. ParticleFilterTF tracks KF mean on simulated LGSSM
# ---------------------------------------------------------------------------

class TestParticleFilterTrackKF:

    def _build_pf(self, lgssm_system, N=500):
        p = lgssm_system
        Phi = p["Phi"]
        H   = p["H"]

        def g(x, u):
            return tf.linalg.matvec(tf.constant(Phi, dtype=F32), x)

        def h_vec(X):   # (N, nx) -> (N, ny)
            return X @ tf.constant(H.T, dtype=F32)

        return ParticleFilterTF(
            g, h_vec, p["Q"], p["R"],
            n_particles=N, resample_thresh=0.5, vectorised_g=False,
        )

    def test_pf_mean_close_to_kf_mean(
        self, lgssm_system, simulated_lgssm, numpy_kf_result
    ):
        """PF mean must be within 3× the KF filtered RMSE of the true state."""
        p     = lgssm_system
        d     = simulated_lgssm
        X     = d["X"]
        Y     = d["Y"]
        T     = d["T"]

        pf    = self._build_pf(lgssm_system, N=800)
        means_pf, _, ess = run_particle_filter_tf(pf, Y, p["x0"], p["P0"])

        # Reference: KF mean
        kf_mean = numpy_kf_result.x_filt

        rmse_kf = float(np.sqrt(np.mean((kf_mean - X) ** 2)))
        rmse_pf = float(np.sqrt(np.mean((means_pf - X) ** 2)))

        # PF should not be more than 3× worse than KF
        assert rmse_pf < rmse_kf * 3.0, (
            f"PF RMSE={rmse_pf:.4f} exceeds 3× KF RMSE={rmse_kf:.4f}"
        )

    def test_pf_ess_never_collapses(self, lgssm_system, simulated_lgssm):
        """ESS should stay above 5% of N throughout (resampling enabled)."""
        p  = lgssm_system
        d  = simulated_lgssm
        N  = 300
        pf = self._build_pf(lgssm_system, N=N)
        _, _, ess = run_particle_filter_tf(pf, d["Y"], p["x0"], p["P0"])
        assert np.all(ess > N * 0.05), (
            f"ESS below 5%N at steps: {np.where(ess <= N*0.05)[0]}, ess={ess}"
        )


# ---------------------------------------------------------------------------
# 5. NonlinSSM_TFP gradient flow
# ---------------------------------------------------------------------------

class TestNonlinSSMGradient:

    def test_gradient_chain_through_nonlin_model(self):
        """Verify gradient flows through the nonlinear scan (no stop-gradients)."""
        rng = np.random.default_rng(55)
        T   = 20
        Y   = rng.standard_normal(T).astype(np.float32)
        model = NonlinSSM_TFP(T=T)
        Y_tf  = _np2tf(Y)

        log_sv = tf.Variable(tf.constant(0.0, dtype=F32))
        log_sw = tf.Variable(tf.constant(0.0, dtype=F32))
        with tf.GradientTape() as tape:
            sv = tf.exp(log_sv)
            sw = tf.exp(log_sw)
            ll = model.log_prob_tf(Y_tf, sv, sw)
        grads = tape.gradient(ll, [log_sv, log_sw])
        for i, g in enumerate(grads):
            assert g is not None and math.isfinite(float(g)), (
                f"grad[{i}] = {g}"
            )


# ---------------------------------------------------------------------------
# 6. HMC chain on NonlinSSM target – basic sanity
# ---------------------------------------------------------------------------

class TestHMCNonlinSSM:

    @pytest.fixture(scope="class")
    def nonlin_chain(self):
        rng = np.random.default_rng(42)
        T   = 30
        Y   = rng.standard_normal(T).astype(np.float32)
        target = make_nonlinssm_hmc_target(Y)
        init = [tf.constant(0.0, F32), tf.constant(0.0, F32)]
        samples, is_accepted = run_hmc(
            target_log_prob_fn = target,
            init_state         = init,
            num_results        = 30,
            num_burnin         = 20,
            step_size          = 0.05,
            num_leapfrog       = 5,
        )
        return samples, is_accepted

    def test_samples_no_nan(self, nonlin_chain):
        samples, _ = nonlin_chain
        for s in samples:
            assert tf.reduce_all(tf.math.is_finite(s)), "HMC samples contain NaN/Inf"

    def test_positive_acceptance(self, nonlin_chain):
        _, is_accepted = nonlin_chain
        acc = float(tf.reduce_mean(tf.cast(is_accepted, F32)))
        assert acc > 0.0, f"Zero acceptance rate: {acc}"

    def test_acceptance_in_range(self, nonlin_chain):
        _, is_accepted = nonlin_chain
        acc = float(tf.reduce_mean(tf.cast(is_accepted, F32)))
        # Very loose bounds – just ensure the chain is not completely stuck
        assert 0.0 < acc <= 1.0, f"Acceptance rate out of range: {acc}"

    def test_samples_shape(self, nonlin_chain):
        samples, _ = nonlin_chain
        # Two parameters: log_sigma_v, log_sigma_w
        assert len(samples) == 2
        assert samples[0].shape == (30,)
        assert samples[1].shape == (30,)


# ---------------------------------------------------------------------------
# 7. run_hmc posterior mean on standard Normal target
# ---------------------------------------------------------------------------

class TestHMCNormalTarget:

    def test_posterior_mean_near_zero(self):
        """HMC on log p(x) = -x²/2 should give mean ≈ 0."""
        def target(x):
            return -0.5 * x * x

        samples, _ = run_hmc(
            target_log_prob_fn = target,
            init_state         = [tf.constant(3.0, F32)],
            num_results        = 500,
            num_burnin         = 200,
            step_size          = 0.5,
            num_leapfrog       = 5,
        )
        post_mean = float(tf.reduce_mean(samples[0]))
        assert abs(post_mean) < 0.8, (
            f"HMC posterior mean={post_mean:.3f}, expected near 0"
        )

    def test_posterior_std_near_one(self):
        """HMC on standard Normal should give std ≈ 1."""
        def target(x):
            return -0.5 * x * x

        samples, _ = run_hmc(
            target_log_prob_fn = target,
            init_state         = [tf.constant(0.0, F32)],
            num_results        = 500,
            num_burnin         = 200,
            step_size          = 0.5,
            num_leapfrog       = 5,
        )
        post_std = float(tf.math.reduce_std(samples[0]))
        assert 0.5 < post_std < 2.0, (
            f"HMC posterior std={post_std:.3f}, expected near 1"
        )
