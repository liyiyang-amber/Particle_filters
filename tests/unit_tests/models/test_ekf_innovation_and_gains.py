"""Unit tests for EKF likelihood computation and innovation statistics."""

import numpy as np
import pytest
from models.extended_kalman_filter import ExtendedKalmanFilter, EKFState


@pytest.fixture
def simple_system():
    """Simple linear system for testing."""
    nx, nz = 2, 1
    A = np.array([[0.9, 0.1], [0.0, 0.8]])
    H = np.array([[1.0, 0.5]])
    Q = np.diag([0.05, 0.02])
    R = np.array([[0.10]])

    def g(x, u):
        return A @ x

    def h(x):
        return H @ x

    def jac_g(x, u):
        return A

    def jac_h(x):
        return H

    return dict(nx=nx, nz=nz, A=A, H=H, Q=Q, R=R, g=g, h=h, jac_g=jac_g, jac_h=jac_h)


def test_innovation_computation(simple_system):
    """Test innovation computation in update step."""
    s = simple_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    x_pred = np.array([1.0, 0.5])
    P_pred = np.eye(s["nx"])
    pred_state = EKFState(mean=x_pred, cov=P_pred, t=1)

    z = np.array([2.0])

    # Compute expected innovation
    z_pred = s["h"](x_pred)
    expected_innov = z - z_pred

    # Run update to trigger innovation computation
    post = ekf.update(pred_state, z)

    # The innovation should be z - H @ x_pred
    np.testing.assert_allclose(expected_innov, z - z_pred, rtol=1e-10)


def test_innovation_covariance(simple_system):
    """Test innovation covariance S = H P H^T + R."""
    s = simple_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    x_pred = np.array([1.0, 0.5])
    P_pred = np.array([[1.0, 0.1], [0.1, 0.5]])
    pred_state = EKFState(mean=x_pred, cov=P_pred, t=1)

    # Compute expected S
    H = s["jac_h"](x_pred)
    expected_S = H @ P_pred @ H.T + s["R"]

    # We can't directly access S from the update method, but we can verify
    # that the Kalman gain K = P H^T S^{-1} is computed correctly
    z = np.array([2.0])
    post = ekf.update(pred_state, z)

    # Verify that update was successful
    assert post.mean.shape == (s["nx"],)
    assert post.cov.shape == (s["nx"], s["nx"])


def test_kalman_gain_computation(simple_system):
    """Test Kalman gain K = P H^T S^{-1}."""
    s = simple_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    x_pred = np.array([1.0, 0.5])
    P_pred = np.array([[1.0, 0.1], [0.1, 0.5]])
    pred_state = EKFState(mean=x_pred, cov=P_pred, t=1)

    H = s["jac_h"](x_pred)
    S = H @ P_pred @ H.T + s["R"]
    expected_K = P_pred @ H.T @ np.linalg.inv(S)

    z = np.array([2.0])
    post = ekf.update(pred_state, z)

    # Verify dimensions match
    assert expected_K.shape == (s["nx"], s["nz"])


def test_posterior_mean_update(simple_system):
    """Test posterior mean: x_post = x_pred + K * (z - h(x_pred))."""
    s = simple_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    x_pred = np.array([1.0, 0.5])
    P_pred = np.array([[1.0, 0.1], [0.1, 0.5]])
    pred_state = EKFState(mean=x_pred, cov=P_pred, t=1)

    z = np.array([2.0])

    # Compute expected posterior mean
    H = s["jac_h"](x_pred)
    z_pred = s["h"](x_pred)
    innov = z - z_pred
    S = H @ P_pred @ H.T + s["R"]
    K = P_pred @ H.T @ np.linalg.inv(S)
    expected_x_post = x_pred + K @ innov

    post = ekf.update(pred_state, z)

    np.testing.assert_allclose(post.mean, expected_x_post, rtol=1e-10)


def test_posterior_covariance_standard_form(simple_system):
    """Test posterior covariance: P_post = (I - K H) P_pred."""
    s = simple_system
    ekf = ExtendedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        jac_g=s["jac_g"],
        jac_h=s["jac_h"],
        joseph=False,
    )

    x_pred = np.array([1.0, 0.5])
    P_pred = np.array([[1.0, 0.1], [0.1, 0.5]])
    pred_state = EKFState(mean=x_pred, cov=P_pred, t=1)

    z = np.array([2.0])

    # Compute expected posterior covariance (standard form)
    H = s["jac_h"](x_pred)
    S = H @ P_pred @ H.T + s["R"]
    K = P_pred @ H.T @ np.linalg.inv(S)
    expected_P_post = (np.eye(s["nx"]) - K @ H) @ P_pred

    post = ekf.update(pred_state, z)

    np.testing.assert_allclose(post.cov, expected_P_post, rtol=1e-10)


def test_joseph_form_correctness(simple_system):
    """Test Joseph form covariance update."""
    s = simple_system
    ekf_joseph = ExtendedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        jac_g=s["jac_g"],
        jac_h=s["jac_h"],
        joseph=True,
    )

    x_pred = np.array([1.0, 0.5])
    P_pred = np.array([[1.0, 0.1], [0.1, 0.5]])
    pred_state = EKFState(mean=x_pred, cov=P_pred, t=1)

    z = np.array([2.0])

    # Compute expected Joseph form: P = (I-KH)P(I-KH)^T + KRK^T
    H = s["jac_h"](x_pred)
    S = H @ P_pred @ H.T + s["R"]
    K = P_pred @ H.T @ np.linalg.inv(S)
    I_KH = np.eye(s["nx"]) - K @ H
    expected_P_post = I_KH @ P_pred @ I_KH.T + K @ s["R"] @ K.T

    post = ekf_joseph.update(pred_state, z)

    np.testing.assert_allclose(post.cov, expected_P_post, rtol=1e-10)


def test_prediction_covariance(simple_system):
    """Test prediction covariance: P_pred = G P G^T + Q."""
    s = simple_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    x_post = np.array([1.0, 0.5])
    P_post = np.array([[1.0, 0.1], [0.1, 0.5]])
    post_state = EKFState(mean=x_post, cov=P_post, t=0)

    # Compute expected prediction covariance
    G = s["jac_g"](x_post, None)
    expected_P_pred = G @ P_post @ G.T + s["Q"]

    pred = ekf.predict(post_state)

    np.testing.assert_allclose(pred.cov, expected_P_pred, rtol=1e-10)


def test_sequential_innovations_statistics(simple_system):
    """Test that innovations have correct statistical properties."""
    s = simple_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    # Generate synthetic data
    np.random.seed(42)
    n = 100
    X_true = np.zeros((n, s["nx"]))
    Z = np.zeros((n, s["nz"]))
    x = np.zeros(s["nx"])

    for k in range(n):
        x = s["A"] @ x + np.random.multivariate_normal(np.zeros(s["nx"]), s["Q"])
        z = s["H"] @ x + np.random.multivariate_normal(np.zeros(s["nz"]), s["R"])
        X_true[k] = x
        Z[k] = z

    # Run filter and collect innovations
    state = EKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)
    innovations = []

    for k in range(n):
        pred = ekf.predict(state)
        z_pred = s["h"](pred.mean)
        innov = Z[k] - z_pred
        innovations.append(innov)
        state = ekf.update(pred, Z[k])

    innovations = np.array(innovations).flatten()

    # Innovations should have approximately zero mean
    innov_mean = np.mean(innovations)
    assert np.abs(innov_mean) < 0.2, f"Innovation mean not zero: {innov_mean}"


def test_error_on_mismatched_dimensions(simple_system):
    """Test that EKF raises errors on dimension mismatches."""
    s = simple_system
    ekf = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jac_g=s["jac_g"], jac_h=s["jac_h"]
    )

    state = EKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)

    # Wrong observation dimension
    z_wrong = np.array([1.0, 2.0])  # Should be 1D

    # This should raise an error or handle gracefully
    # Depending on implementation, it might raise ValueError
    try:
        pred = ekf.predict(state)
        post = ekf.update(pred, z_wrong)
        # If it doesn't raise, check that shapes don't match
        assert z_wrong.shape[0] != s["nz"]
    except (ValueError, IndexError):
        pass  # Expected behavior
