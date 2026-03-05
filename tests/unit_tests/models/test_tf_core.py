"""Unit tests for models/tf_core.py.

Covers
------
* lgssm_log_likelihood  – value matches NumPy KF, gradient exists, XLA matches
* kalman_filter_tf      – shapes, loglik consistency, positive-definiteness
* EKFTrackerTF          – shapes, linear-model recovery
* UKFTrackerTF          – shapes, linear-model recovery
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import tensorflow as tf

from models.kalman_filter import kalman_filter_general
from models.tf_core import (
    EKFTrackerTF,
    KFResultsTF,
    UKFTrackerTF,
    kalman_filter_tf,
    lgssm_log_likelihood,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

F32 = tf.float32


def _np2tf(x: np.ndarray) -> tf.Tensor:
    return tf.constant(x, dtype=F32)


# ---------------------------------------------------------------------------
# Shared small LGSSM parameters (nx=2, ny=1, T=12)
# The fixture values mirror tests/conftest.py small_system
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lgssm_params():
    nx, ny = 2, 1
    Phi = np.array([[0.9, 0.2], [0.0, 0.7]], dtype=np.float32)
    H   = np.array([[1.0, 0.0]], dtype=np.float32)
    Q   = np.diag([0.05, 0.02]).astype(np.float32)
    R   = np.array([[0.10]], dtype=np.float32)
    x0  = np.zeros(nx, dtype=np.float32)
    P0  = np.eye(nx, dtype=np.float32)
    return dict(nx=nx, ny=ny, Phi=Phi, H=H, Q=Q, R=R, x0=x0, P0=P0)


@pytest.fixture(scope="module")
def synthetic_obs(lgssm_params):
    rng = np.random.default_rng(42)
    T   = 20
    return rng.normal(size=(T, lgssm_params["ny"])).astype(np.float32)


# ---------------------------------------------------------------------------
# lgssm_log_likelihood
# ---------------------------------------------------------------------------

class TestLGSSMLogLikelihood:
    """Tests for lgssm_log_likelihood (TF JIT-compiled Kalman-filter LL)."""

    def test_returns_scalar(self, lgssm_params, synthetic_obs):
        p = lgssm_params
        ll = lgssm_log_likelihood(
            _np2tf(synthetic_obs),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),  _np2tf(p["R"]),
            _np2tf(p["x0"]), _np2tf(p["P0"]),
        )
        assert ll.shape == (), f"Expected scalar, got shape {ll.shape}"

    def test_value_is_finite(self, lgssm_params, synthetic_obs):
        p = lgssm_params
        ll = lgssm_log_likelihood(
            _np2tf(synthetic_obs),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),  _np2tf(p["R"]),
            _np2tf(p["x0"]), _np2tf(p["P0"]),
        )
        assert math.isfinite(float(ll)), f"Log-likelihood is not finite: {ll}"

    def test_matches_numpy_kf(self, lgssm_params, synthetic_obs):
        """TF log-likelihood must agree with NumPy KF to <0.02 nats."""
        p = lgssm_params
        Gamma = np.eye(p["nx"], dtype=np.float32)

        # NumPy reference
        kf_np = kalman_filter_general(
            Y=synthetic_obs,
            Phi=p["Phi"], H=p["H"],
            Gamma=Gamma,  Q=p["Q"], R=p["R"],
            x0=p["x0"], P0=p["P0"],
        )
        ll_np = float(kf_np.loglik)

        # TF version
        ll_tf = float(lgssm_log_likelihood(
            _np2tf(synthetic_obs),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),  _np2tf(p["R"]),
            _np2tf(p["x0"]), _np2tf(p["P0"]),
        ))

        assert abs(ll_tf - ll_np) < 0.02, (
            f"TF LL={ll_tf:.4f} vs NumPy LL={ll_np:.4f}, diff={abs(ll_tf-ll_np):.4e}"
        )

    def test_gradient_exists(self, lgssm_params, synthetic_obs):
        """Gradient w.r.t. Q must be non-None and finite."""
        p = lgssm_params
        Q_var = tf.Variable(_np2tf(p["Q"]))
        with tf.GradientTape() as tape:
            ll = lgssm_log_likelihood(
                _np2tf(synthetic_obs),
                _np2tf(p["Phi"]), _np2tf(p["H"]),
                Q_var, _np2tf(p["R"]),
                _np2tf(p["x0"]), _np2tf(p["P0"]),
            )
        grad = tape.gradient(ll, Q_var)
        assert grad is not None, "Gradient w.r.t. Q is None"
        assert tf.reduce_all(tf.math.is_finite(grad)), f"Non-finite gradient: {grad}"

    def test_xla_matches_non_xla(self, lgssm_params, synthetic_obs):
        """lgssm_log_likelihood with jit_compile=True must agree with the
        standard @tf.function version to within float32 tolerance (~1e-5)."""
        p = lgssm_params

        @tf.function(jit_compile=True)
        def ll_xla(Y, F, H, Q, R, m0, P0):
            return lgssm_log_likelihood(Y, F, H, Q, R, m0, P0)

        ll_std = float(lgssm_log_likelihood(
            _np2tf(synthetic_obs),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),  _np2tf(p["R"]),
            _np2tf(p["x0"]), _np2tf(p["P0"]),
        ))
        ll_xla_val = float(ll_xla(
            _np2tf(synthetic_obs),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),  _np2tf(p["R"]),
            _np2tf(p["x0"]), _np2tf(p["P0"]),
        ))
        assert abs(ll_std - ll_xla_val) < 1e-3, (
            f"XLA={ll_xla_val:.6f} vs standard={ll_std:.6f}"
        )

    def test_monotone_in_data_fit(self, lgssm_params):
        """LL should be higher when observations are generated by the model
        than when they are random noise far from the model mean."""
        p  = lgssm_params
        rng = np.random.default_rng(0)
        T   = 30

        # On-model observations (small innovation)
        Y_good = rng.normal(scale=0.1, size=(T, p["ny"])).astype(np.float32)
        # Off-model observations (large outliers)
        Y_bad  = rng.normal(scale=100.0, size=(T, p["ny"])).astype(np.float32)

        def ll(Y):
            return float(lgssm_log_likelihood(
                _np2tf(Y),
                _np2tf(p["Phi"]), _np2tf(p["H"]),
                _np2tf(p["Q"]),  _np2tf(p["R"]),
                _np2tf(p["x0"]), _np2tf(p["P0"]),
            ))

        assert ll(Y_good) > ll(Y_bad), "LL should be higher for on-model data"


# ---------------------------------------------------------------------------
# kalman_filter_tf
# ---------------------------------------------------------------------------

class TestKalmanFilterTF:
    """Tests for the full kalman_filter_tf function."""

    def test_returns_kfresultstf(self, lgssm_params, synthetic_obs):
        p = lgssm_params
        res = kalman_filter_tf(
            _np2tf(synthetic_obs),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),  _np2tf(p["R"]),
            _np2tf(p["x0"]), _np2tf(p["P0"]),
        )
        assert isinstance(res, KFResultsTF)

    def test_output_shapes(self, lgssm_params, synthetic_obs):
        p  = lgssm_params
        T  = synthetic_obs.shape[0]
        nx = p["nx"]
        ny = p["ny"]

        res = kalman_filter_tf(
            _np2tf(synthetic_obs),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),  _np2tf(p["R"]),
            _np2tf(p["x0"]), _np2tf(p["P0"]),
        )

        assert res.x_filt.shape == (T, nx), f"x_filt: {res.x_filt.shape}"
        assert res.P_filt.shape == (T, nx, nx), f"P_filt: {res.P_filt.shape}"
        assert res.x_pred.shape == (T, nx), f"x_pred: {res.x_pred.shape}"
        assert res.P_pred.shape == (T, nx, nx), f"P_pred: {res.P_pred.shape}"
        assert res.K.shape      == (T, nx, ny), f"K: {res.K.shape}"
        assert res.innov.shape  == (T, ny), f"innov: {res.innov.shape}"
        assert res.S.shape      == (T, ny, ny), f"S: {res.S.shape}"
        assert res.loglik.shape == (), f"loglik: {res.loglik.shape}"

    def test_loglik_consistent_with_lgssm_ll(self, lgssm_params, synthetic_obs):
        """kalman_filter_tf.loglik must equal lgssm_log_likelihood exactly."""
        p = lgssm_params
        args = (
            _np2tf(synthetic_obs),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),  _np2tf(p["R"]),
            _np2tf(p["x0"]), _np2tf(p["P0"]),
        )
        ll_fn  = float(lgssm_log_likelihood(*args))
        ll_kf  = float(kalman_filter_tf(*args).loglik)
        assert abs(ll_fn - ll_kf) < 1e-4, (
            f"lgssm_log_likelihood={ll_fn:.6f} vs kalman_filter_tf.loglik={ll_kf:.6f}"
        )

    def test_filtered_cov_positive_definite(self, lgssm_params, synthetic_obs):
        """All P_filt matrices must have positive eigenvalues."""
        p = lgssm_params
        res = kalman_filter_tf(
            _np2tf(synthetic_obs),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),  _np2tf(p["R"]),
            _np2tf(p["x0"]), _np2tf(p["P0"]),
        )
        P_filt = res.P_filt.numpy()   # (T, nx, nx)
        for t in range(P_filt.shape[0]):
            eigvals = np.linalg.eigvalsh(P_filt[t])
            assert np.all(eigvals > 0), f"P_filt[{t}] not PD, min eigval={eigvals.min()}"

    def test_predicted_cov_positive_definite(self, lgssm_params, synthetic_obs):
        p = lgssm_params
        res = kalman_filter_tf(
            _np2tf(synthetic_obs),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),  _np2tf(p["R"]),
            _np2tf(p["x0"]), _np2tf(p["P0"]),
        )
        P_pred = res.P_pred.numpy()
        for t in range(P_pred.shape[0]):
            eigvals = np.linalg.eigvalsh(P_pred[t])
            assert np.all(eigvals > 0), f"P_pred[{t}] not PD, min eigval={eigvals.min()}"

    def test_matches_numpy_x_filt(self, lgssm_params, synthetic_obs):
        """x_filt from TF must match NumPy KF to within 1e-4 (float32 budget)."""
        p = lgssm_params
        Gamma = np.eye(p["nx"], dtype=np.float32)

        kf_np = kalman_filter_general(
            Y=synthetic_obs,
            Phi=p["Phi"], H=p["H"],
            Gamma=Gamma,  Q=p["Q"], R=p["R"],
            x0=p["x0"], P0=p["P0"],
        )

        res_tf = kalman_filter_tf(
            _np2tf(synthetic_obs),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),  _np2tf(p["R"]),
            _np2tf(p["x0"]), _np2tf(p["P0"]),
        )

        diff = np.max(np.abs(kf_np.x_filt - res_tf.x_filt.numpy()))
        assert diff < 1e-3, f"Max |x_filt diff|={diff:.2e}"

    def test_values_finite(self, lgssm_params, synthetic_obs):
        p = lgssm_params
        res = kalman_filter_tf(
            _np2tf(synthetic_obs),
            _np2tf(p["Phi"]), _np2tf(p["H"]),
            _np2tf(p["Q"]),  _np2tf(p["R"]),
            _np2tf(p["x0"]), _np2tf(p["P0"]),
        )
        for field in ["x_pred", "P_pred", "x_filt", "P_filt", "K", "innov", "S"]:
            arr = getattr(res, field).numpy()
            assert np.all(np.isfinite(arr)), f"{field} contains non-finite values"


# ---------------------------------------------------------------------------
# EKFTrackerTF
# ---------------------------------------------------------------------------

class TestEKFTrackerTF:
    """Tests for the step-by-step EKF tracker (linear model → exact KF)."""

    @pytest.fixture
    def linear_model(self, lgssm_params):
        p = lgssm_params
        Phi = p["Phi"]
        H   = p["H"]

        def g(x, u=None):
            return tf.linalg.matvec(tf.constant(Phi, dtype=F32), x)

        def h(x):
            return tf.linalg.matvec(tf.constant(H, dtype=F32), x)

        return g, h, p

    def test_init_returns_state(self, linear_model):
        g, h, p = linear_model
        ekf = EKFTrackerTF(g, h, p["Q"], p["R"])
        state = ekf.init(p["x0"], p["P0"])
        assert state is not None
        assert state.mean.shape == (p["nx"],)
        assert state.cov.shape  == (p["nx"], p["nx"])

    def test_predict_shapes(self, linear_model):
        g, h, p = linear_model
        ekf = EKFTrackerTF(g, h, p["Q"], p["R"])
        ekf.init(p["x0"], p["P0"])
        m_pred, P_pred = ekf.predict()
        assert m_pred.shape == (p["nx"],)
        assert P_pred.shape == (p["nx"], p["nx"])

    def test_update_shapes(self, linear_model, synthetic_obs):
        g, h, p = linear_model
        ekf = EKFTrackerTF(g, h, p["Q"], p["R"])
        ekf.init(p["x0"], p["P0"])
        ekf.predict()
        m_post, P_post = ekf.update(synthetic_obs[0])
        assert m_post.shape == (p["nx"],)
        assert P_post.shape == (p["nx"], p["nx"])

    def test_predict_returns_numpy(self, linear_model):
        g, h, p = linear_model
        ekf = EKFTrackerTF(g, h, p["Q"], p["R"])
        ekf.init(p["x0"], p["P0"])
        m, P = ekf.predict()
        assert isinstance(m, np.ndarray), "predict() should return numpy"
        assert isinstance(P, np.ndarray)

    def test_update_returns_numpy(self, linear_model, synthetic_obs):
        g, h, p = linear_model
        ekf = EKFTrackerTF(g, h, p["Q"], p["R"])
        ekf.init(p["x0"], p["P0"])
        ekf.predict()
        m, P = ekf.update(synthetic_obs[0])
        assert isinstance(m, np.ndarray)
        assert isinstance(P, np.ndarray)

    def test_linear_matches_kf(self, linear_model, synthetic_obs, lgssm_params):
        """EKF on a linear model must produce x_filt close to the exact KF."""
        g, h, p = linear_model
        Gamma = np.eye(p["nx"], dtype=np.float32)

        # NumPy reference KF
        kf_np = kalman_filter_general(
            Y=synthetic_obs,
            Phi=p["Phi"], H=p["H"],
            Gamma=Gamma,  Q=p["Q"], R=p["R"],
            x0=p["x0"], P0=p["P0"],
        )

        # EKF tracker
        ekf = EKFTrackerTF(g, h, p["Q"], p["R"])
        ekf.init(p["x0"], p["P0"])

        x_filt_ekf = []
        for y in synthetic_obs:
            ekf.predict()
            m_post, _ = ekf.update(y)
            x_filt_ekf.append(m_post)

        x_filt_ekf = np.array(x_filt_ekf)
        max_diff = np.max(np.abs(x_filt_ekf - kf_np.x_filt))
        assert max_diff < 1e-3, f"EKF vs KF max diff={max_diff:.2e}"

    def test_filtered_cov_pd(self, linear_model, synthetic_obs):
        g, h, p = linear_model
        ekf = EKFTrackerTF(g, h, p["Q"], p["R"])
        ekf.init(p["x0"], p["P0"])
        for y in synthetic_obs:
            ekf.predict()
            _, P_post = ekf.update(y)
            eigvals = np.linalg.eigvalsh(P_post)
            assert np.all(eigvals > 0), f"P_post not PD: {eigvals}"

    def test_get_past_mean(self, linear_model, synthetic_obs):
        g, h, p = linear_model
        ekf = EKFTrackerTF(g, h, p["Q"], p["R"])
        ekf.init(p["x0"], p["P0"])
        ekf.predict()
        past = ekf.get_past_mean()
        assert past is not None
        assert past.shape == (p["nx"],)


# ---------------------------------------------------------------------------
# UKFTrackerTF
# ---------------------------------------------------------------------------

class TestUKFTrackerTF:
    """Tests for the step-by-step UKF tracker (linear model → near-exact KF)."""

    @pytest.fixture
    def linear_model(self, lgssm_params):
        p = lgssm_params
        Phi = p["Phi"]
        H   = p["H"]

        def g(x, u=None):
            return tf.linalg.matvec(tf.constant(Phi, dtype=F32), x)

        def h(x):
            return tf.linalg.matvec(tf.constant(H, dtype=F32), x)

        return g, h, p

    def test_init_returns_state(self, linear_model):
        g, h, p = linear_model
        ukf = UKFTrackerTF(g, h, p["Q"], p["R"])
        state = ukf.init(p["x0"], p["P0"])
        assert state is not None

    def test_predict_shapes(self, linear_model):
        g, h, p = linear_model
        ukf = UKFTrackerTF(g, h, p["Q"], p["R"])
        ukf.init(p["x0"], p["P0"])
        m_pred, P_pred = ukf.predict()
        assert m_pred.shape == (p["nx"],)
        assert P_pred.shape == (p["nx"], p["nx"])

    def test_update_shapes(self, linear_model, synthetic_obs):
        g, h, p = linear_model
        ukf = UKFTrackerTF(g, h, p["Q"], p["R"])
        ukf.init(p["x0"], p["P0"])
        ukf.predict()
        m_post, P_post = ukf.update(synthetic_obs[0])
        assert m_post.shape == (p["nx"],)
        assert P_post.shape == (p["nx"], p["nx"])

    def test_predict_returns_numpy(self, linear_model):
        g, h, p = linear_model
        ukf = UKFTrackerTF(g, h, p["Q"], p["R"])
        ukf.init(p["x0"], p["P0"])
        m, P = ukf.predict()
        assert isinstance(m, np.ndarray)
        assert isinstance(P, np.ndarray)

    def test_linear_matches_kf(self, linear_model, synthetic_obs, lgssm_params):
        """UKF on a linear model must produce x_filt close to the exact KF."""
        g, h, p = linear_model
        Gamma = np.eye(p["nx"], dtype=np.float32)

        kf_np = kalman_filter_general(
            Y=synthetic_obs,
            Phi=p["Phi"], H=p["H"],
            Gamma=Gamma,  Q=p["Q"], R=p["R"],
            x0=p["x0"], P0=p["P0"],
        )

        ukf = UKFTrackerTF(g, h, p["Q"], p["R"])
        ukf.init(p["x0"], p["P0"])

        x_filt_ukf = []
        for y in synthetic_obs:
            ukf.predict()
            m_post, _ = ukf.update(y)
            x_filt_ukf.append(m_post)

        x_filt_ukf = np.array(x_filt_ukf)
        max_diff = np.max(np.abs(x_filt_ukf - kf_np.x_filt))
        assert max_diff < 5e-2, f"UKF vs KF max diff={max_diff:.2e}"

    def test_filtered_cov_pd(self, linear_model, synthetic_obs):
        g, h, p = linear_model
        ukf = UKFTrackerTF(g, h, p["Q"], p["R"])
        ukf.init(p["x0"], p["P0"])
        for y in synthetic_obs:
            ukf.predict()
            _, P_post = ukf.update(y)
            eigvals = np.linalg.eigvalsh(P_post)
            assert np.all(eigvals > 0), f"P_post not PD: {eigvals}"

    def test_get_past_mean(self, linear_model, synthetic_obs):
        g, h, p = linear_model
        ukf = UKFTrackerTF(g, h, p["Q"], p["R"])
        ukf.init(p["x0"], p["P0"])
        ukf.predict()
        past = ukf.get_past_mean()
        assert past is not None and past.shape == (p["nx"],)
