"""Unit tests for Extended Kalman Filter: shapes, API, and basic functionality."""

import numpy as np
import pytest
from models.extended_kalman_filter import (
    ExtendedKalmanFilter,
    EKFState,
    numerical_jacobian_g,
    numerical_jacobian_h,
)


@pytest.fixture
def simple_linear_system():
    """Simple linear system for testing: x' = Ax + w, z = Hx + v."""
    nx, nz = 2, 1
    A = np.array([[0.9, 0.2], [0.0, 0.7]])
    H = np.array([[1.0, 0.5]])
    Q = np.diag([0.05, 0.02])
    R = np.array([[0.10]])

    def g(x, u):
        return A @ x if u is None else A @ x + u

    def h(x):
        return H @ x

    def jac_g(x, u):
        return A

    def jac_h(x):
        return H

    return dict(
        nx=nx, nz=nz, A=A, H=H, Q=Q, R=R, g=g, h=h, jac_g=jac_g, jac_h=jac_h
    )


@pytest.fixture
def nonlinear_system():
    """Nonlinear system: x' = 0.5*x + 25*x/(1+x^2), z = x^2/20."""
    nx, nz = 1, 1
    Q = np.array([[0.1]])
    R = np.array([[1.0]])

    def g(x, u):
        x = np.atleast_1d(x)
        return 0.5 * x + 25 * x / (1 + x**2)

    def h(x):
        x = np.atleast_1d(x)
        return x**2 / 20.0

    def jac_g(x, u):
        x = np.atleast_1d(x)
        denom = (1 + x**2) ** 2
        deriv = 0.5 + 25 * (1 - x**2) / denom
        return np.atleast_2d(deriv)

    def jac_h(x):
        x = np.atleast_1d(x)
        return np.atleast_2d(x / 10.0)

    return dict(nx=nx, nz=nz, Q=Q, R=R, g=g, h=h, jac_g=jac_g, jac_h=jac_h)


def test_ekf_state_creation():
    """Test EKFState dataclass creation."""
    mean = np.array([1.0, 2.0])
    cov = np.eye(2)
    state = EKFState(mean=mean, cov=cov, t=0)

    assert state.t == 0
    np.testing.assert_array_equal(state.mean, mean)
    np.testing.assert_array_equal(state.cov, cov)


def test_ekf_initialization(simple_linear_system):
    """Test EKF initialization with various configurations."""
    s = simple_linear_system

    # With analytic Jacobians
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )
    assert ekf.joseph is False
    assert ekf.jitter == 0.0

    # With Joseph form and jitter
    ekf_joseph = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], joseph=True, jitter=1e-8
    )
    assert ekf_joseph.joseph is True
    assert ekf_joseph.jitter == 1e-8

    # Without Jacobians (should use numerical)
    ekf_numerical = ExtendedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])
    assert ekf_numerical.jac_g is None
    assert ekf_numerical.jac_h is None


def test_predict_shapes(simple_linear_system):
    """Test that predict returns correct shapes."""
    s = simple_linear_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    x0 = np.zeros(s["nx"])
    P0 = np.eye(s["nx"])
    state = EKFState(mean=x0, cov=P0, t=0)

    pred = ekf.predict(state)

    assert pred.mean.shape == (s["nx"],)
    assert pred.cov.shape == (s["nx"], s["nx"])
    assert pred.t == 1
    assert np.allclose(pred.cov, pred.cov.T)  # symmetry


def test_update_shapes(simple_linear_system):
    """Test that update returns correct shapes."""
    s = simple_linear_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    x_pred = np.array([1.0, 0.5])
    P_pred = np.eye(s["nx"])
    pred_state = EKFState(mean=x_pred, cov=P_pred, t=1)

    z = np.array([1.2])
    post = ekf.update(pred_state, z)

    assert post.mean.shape == (s["nx"],)
    assert post.cov.shape == (s["nx"], s["nx"])
    assert post.t == 1
    assert np.allclose(post.cov, post.cov.T)  # symmetry


def test_step_combines_predict_update(simple_linear_system):
    """Test that step() combines predict and update correctly."""
    s = simple_linear_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    state = EKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)
    z = np.array([1.0])

    # Using step
    result_step = ekf.step(state, z)

    # Using predict + update separately
    pred = ekf.predict(state)
    result_manual = ekf.update(pred, z)

    np.testing.assert_allclose(result_step.mean, result_manual.mean, rtol=1e-10)
    np.testing.assert_allclose(result_step.cov, result_manual.cov, rtol=1e-10)
    assert result_step.t == result_manual.t


def test_joseph_form_stability(simple_linear_system):
    """Test that Joseph form produces symmetric positive-definite covariances."""
    s = simple_linear_system
    ekf = ExtendedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        jac_g=s["jac_g"],
        jac_h=s["jac_h"],
        joseph=True,
        jitter=1e-10,
    )

    state = EKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)

    # Run several steps
    for k in range(10):
        z = np.random.randn(s["nz"])
        state = ekf.step(state, z)

        # Check symmetry
        assert np.allclose(state.cov, state.cov.T, atol=1e-10)

        # Check positive-definiteness via eigenvalues
        eigvals = np.linalg.eigvalsh(state.cov)
        assert np.all(eigvals > 0), f"Non-positive eigenvalues at step {k}: {eigvals}"


def test_numerical_jacobian_g():
    """Test numerical Jacobian computation for g."""
    A = np.array([[0.9, 0.1], [0.0, 0.8]])

    def g(x, u):
        return A @ x

    x = np.array([1.0, 2.0])
    J_numerical = numerical_jacobian_g(g, x, u=None)

    np.testing.assert_allclose(J_numerical, A, atol=1e-5)


def test_numerical_jacobian_h():
    """Test numerical Jacobian computation for h."""
    H = np.array([[1.0, 0.5]])

    def h(x):
        return H @ x

    x = np.array([1.0, 2.0])
    J_numerical = numerical_jacobian_h(h, x)

    np.testing.assert_allclose(J_numerical, H, atol=1e-5)


def test_numerical_vs_analytic_jacobians(simple_linear_system):
    """Test that numerical Jacobians match analytic for linear case."""
    s = simple_linear_system
    x = np.array([1.0, 0.5])

    # Jacobian of g
    J_g_analytic = s["jac_g"](x, None)
    J_g_numerical = numerical_jacobian_g(s["g"], x, None)
    np.testing.assert_allclose(J_g_numerical, J_g_analytic, atol=1e-5)

    # Jacobian of h
    J_h_analytic = s["jac_h"](x)
    J_h_numerical = numerical_jacobian_h(s["h"], x)
    np.testing.assert_allclose(J_h_numerical, J_h_analytic, atol=1e-5)


def test_ekf_with_control_input(simple_linear_system):
    """Test EKF with control input."""
    s = simple_linear_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    state = EKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)
    u = np.array([0.1, 0.2])

    pred = ekf.predict(state, u=u)

    # Expected: A @ x0 + u
    expected_mean = s["A"] @ state.mean + u
    np.testing.assert_allclose(pred.mean, expected_mean, rtol=1e-10)


def test_nonlinear_ekf_runs(nonlinear_system):
    """Test EKF on a nonlinear system."""
    s = nonlinear_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    state = EKFState(mean=np.array([1.0]), cov=np.array([[1.0]]), t=0)

    # Run multiple steps
    for _ in range(20):
        z = np.array([np.random.randn()])
        state = ekf.step(state, z)

        assert state.mean.shape == (1,)
        assert state.cov.shape == (1, 1)
        assert np.allclose(state.cov, state.cov.T)


def test_covariance_reduction_after_update(simple_linear_system):
    """Test that update step reduces uncertainty (in general)."""
    s = simple_linear_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    # Start with high uncertainty
    state = EKFState(mean=np.zeros(s["nx"]), cov=10.0 * np.eye(s["nx"]), t=0)
    pred = ekf.predict(state)

    # Measurement with reasonable noise
    z = np.array([0.5])
    post = ekf.update(pred, z)

    # Check that trace (total variance) is reduced
    trace_pred = np.trace(pred.cov)
    trace_post = np.trace(post.cov)

    assert trace_post < trace_pred, "Update should reduce uncertainty"


def test_sequential_filtering(simple_linear_system):
    """Test sequential filtering over multiple time steps."""
    s = simple_linear_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    n = 50
    np.random.seed(42)
    observations = np.random.randn(n, s["nz"])

    state = EKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)
    states = [state]

    for k in range(n):
        state = ekf.step(state, observations[k])
        states.append(state)

        assert state.t == k + 1
        assert state.mean.shape == (s["nx"],)
        assert state.cov.shape == (s["nx"], s["nx"])

    assert len(states) == n + 1


def test_ekf_without_jacobians_uses_numerical(simple_linear_system):
    """Test that EKF without jacobians uses numerical approximation."""
    s = simple_linear_system

    # Create EKF without Jacobians
    ekf = ExtendedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    # Create EKF with Jacobians
    ekf_analytic = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    state = EKFState(mean=np.array([1.0, 0.5]), cov=np.eye(s["nx"]), t=0)
    z = np.array([1.2])

    # Results should be very close for linear system
    result_numerical = ekf.step(state, z)
    result_analytic = ekf_analytic.step(state, z)

    np.testing.assert_allclose(
        result_numerical.mean, result_analytic.mean, rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        result_numerical.cov, result_analytic.cov, rtol=1e-4, atol=1e-6
    )


def test_jitter_prevents_singular_matrix(simple_linear_system):
    """Test that jitter helps with near-singular covariances."""
    s = simple_linear_system
    ekf = ExtendedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        jac_g=s["jac_g"],
        jac_h=s["jac_h"],
        jitter=1e-6,
    )

    # Start with very small covariance
    state = EKFState(mean=np.zeros(s["nx"]), cov=1e-10 * np.eye(s["nx"]), t=0)

    # Should not crash with jitter
    z = np.array([1.0])
    result = ekf.step(state, z)

    assert result.mean.shape == (s["nx"],)
    assert np.all(np.isfinite(result.mean))
    assert np.all(np.isfinite(result.cov))
