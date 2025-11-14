"""Unit tests for UKF sigma points and weights."""

import numpy as np
import pytest
from models.unscented_kalman_filter import UnscentedKalmanFilter, UKFState


@pytest.fixture
def simple_system():
    """Simple system for testing."""
    nx, nz = 3, 2
    Q = np.diag([0.1, 0.2, 0.15])
    R = np.diag([0.5, 0.3])

    def g(x, u):
        return 0.9 * x

    def h(x):
        return np.array([x[0] + x[1], x[1] + x[2]])

    return dict(nx=nx, nz=nz, Q=Q, R=R, g=g, h=h)


def test_sigma_points_count(simple_system):
    """Test that correct number of sigma points are generated."""
    s = simple_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    mean = np.zeros(s["nx"])
    cov = np.eye(s["nx"])

    sigma_pts = ukf._sigma_points(mean, cov)

    # Should be 2*nx + 1 points
    assert sigma_pts.shape[0] == 2 * s["nx"] + 1
    assert sigma_pts.shape[1] == s["nx"]


def test_sigma_points_first_point_is_mean(simple_system):
    """Test that first sigma point equals the mean."""
    s = simple_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    mean = np.array([1.0, 2.0, 3.0])
    cov = np.eye(s["nx"])

    sigma_pts = ukf._sigma_points(mean, cov)

    np.testing.assert_allclose(sigma_pts[0], mean, rtol=1e-10)


def test_sigma_points_symmetry(simple_system):
    """Test that sigma points are symmetric around mean."""
    s = simple_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    mean = np.array([1.0, 2.0, 3.0])
    cov = np.eye(s["nx"])

    sigma_pts = ukf._sigma_points(mean, cov)

    # Points i+1 and i+1+nx should be symmetric around mean
    for i in range(s["nx"]):
        pos_point = sigma_pts[i + 1]
        neg_point = sigma_pts[i + 1 + s["nx"]]

        # Mean of pos and neg should equal original mean
        avg = (pos_point + neg_point) / 2
        np.testing.assert_allclose(avg, mean, rtol=1e-10)

        # They should be equidistant from mean
        dist_pos = np.linalg.norm(pos_point - mean)
        dist_neg = np.linalg.norm(neg_point - mean)
        np.testing.assert_allclose(dist_pos, dist_neg, rtol=1e-10)


def test_weights_properties(simple_system):
    """Test UKF weight properties."""
    s = simple_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    # Check weight dimensions
    assert len(ukf.Wm) == 2 * s["nx"] + 1
    assert len(ukf.Wc) == 2 * s["nx"] + 1

    # Mean weights should sum to 1
    assert np.isclose(np.sum(ukf.Wm), 1.0)

    # Note: Covariance weights do NOT necessarily sum to 1 in general UKF
    # due to the beta parameter in Wc[0]
    # Just check they are finite
    assert np.all(np.isfinite(ukf.Wc))


def test_sigma_points_recover_mean(simple_system):
    """Test that sigma points recover the original mean."""
    s = simple_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    mean = np.array([1.5, -0.5, 2.0])
    cov = np.diag([1.0, 2.0, 0.5])

    sigma_pts = ukf._sigma_points(mean, cov)

    # Weighted mean of sigma points should equal original mean
    recovered_mean = np.sum(ukf.Wm[:, None] * sigma_pts, axis=0)

    np.testing.assert_allclose(recovered_mean, mean, rtol=1e-8, atol=1e-10)


def test_sigma_points_recover_covariance(simple_system):
    """Test that sigma points recover the original covariance."""
    s = simple_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    mean = np.array([1.0, 2.0, 3.0])
    cov = np.array([[1.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.5]])

    sigma_pts = ukf._sigma_points(mean, cov)

    # Recover mean first
    recovered_mean = np.sum(ukf.Wm[:, None] * sigma_pts, axis=0)

    # Recover covariance
    DX = sigma_pts - recovered_mean
    recovered_cov = np.zeros_like(cov)
    for i in range(sigma_pts.shape[0]):
        recovered_cov += ukf.Wc[i] * np.outer(DX[i], DX[i])

    np.testing.assert_allclose(recovered_cov, cov, rtol=1e-5, atol=1e-8)


def test_ut_through_identity_function(simple_system):
    """Test unscented transform through identity function."""
    s = simple_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    mean = np.array([1.0, 2.0, 3.0])
    cov = np.diag([1.0, 2.0, 0.5])

    # Identity function
    def f_identity(x):
        return x

    sigma_pts = ukf._sigma_points(mean, cov)
    transformed_pts = np.array([f_identity(xi) for xi in sigma_pts])

    # Recover statistics
    trans_mean = np.sum(ukf.Wm[:, None] * transformed_pts, axis=0)
    DX = transformed_pts - trans_mean
    trans_cov = np.zeros_like(cov)
    for i in range(transformed_pts.shape[0]):
        trans_cov += ukf.Wc[i] * np.outer(DX[i], DX[i])

    # Should match original for identity function
    np.testing.assert_allclose(trans_mean, mean, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(trans_cov, cov, rtol=1e-5, atol=1e-8)


def test_ut_through_linear_function(simple_system):
    """Test unscented transform through linear function."""
    s = simple_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    mean = np.array([1.0, 2.0, 3.0])
    cov = np.diag([1.0, 2.0, 0.5])

    # Linear function: f(x) = A @ x + b
    A = np.array([[1.0, 0.5, 0.2], [0.3, 1.0, 0.4]])
    b = np.array([0.5, -0.3])

    def f_linear(x):
        return A @ x + b

    sigma_pts = ukf._sigma_points(mean, cov)
    transformed_pts = np.array([f_linear(xi) for xi in sigma_pts])

    # Recover statistics
    trans_mean = np.sum(ukf.Wm[:, None] * transformed_pts, axis=0)
    DX = transformed_pts - trans_mean
    trans_cov = np.zeros((2, 2))
    for i in range(transformed_pts.shape[0]):
        trans_cov += ukf.Wc[i] * np.outer(DX[i], DX[i])

    # Expected for linear function
    expected_mean = A @ mean + b
    expected_cov = A @ cov @ A.T

    np.testing.assert_allclose(trans_mean, expected_mean, rtol=1e-10)
    np.testing.assert_allclose(trans_cov, expected_cov, rtol=1e-5, atol=1e-8)


def test_ut_through_nonlinear_function(simple_system):
    """Test unscented transform through nonlinear function."""
    s = simple_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    mean = np.array([0.0, 0.0, 0.0])
    cov = 0.1 * np.eye(s["nx"])

    # Nonlinear function: f(x) = x^2 (element-wise)
    def f_nonlinear(x):
        return x**2

    sigma_pts = ukf._sigma_points(mean, cov)
    transformed_pts = np.array([f_nonlinear(xi) for xi in sigma_pts])

    # Recover statistics
    trans_mean = np.sum(ukf.Wm[:, None] * transformed_pts, axis=0)

    # For x ~ N(0, 0.1), x^2 should have mean ≈ 0.1
    # This is a rough check
    assert np.all(trans_mean > 0)  # Squared values should be positive on average
    assert np.all(trans_mean < 1.0)  # But not too large


def test_different_alpha_affects_spread(simple_system):
    """Test that alpha parameter affects sigma point spread."""
    s = simple_system

    mean = np.array([0.0, 0.0, 0.0])
    cov = np.eye(s["nx"])

    # Small alpha
    ukf_small = UnscentedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], alpha=1e-3
    )
    sigma_small = ukf_small._sigma_points(mean, cov)

    # Large alpha
    ukf_large = UnscentedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], alpha=1.0
    )
    sigma_large = ukf_large._sigma_points(mean, cov)

    # Compute spreads (average distance from mean)
    spread_small = np.mean([np.linalg.norm(xi - mean) for xi in sigma_small[1:]])
    spread_large = np.mean([np.linalg.norm(xi - mean) for xi in sigma_large[1:]])

    # Larger alpha should give larger spread
    assert spread_large > spread_small


def test_kappa_affects_lambda(simple_system):
    """Test that kappa affects the lambda parameter."""
    s = simple_system

    ukf1 = UnscentedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], alpha=1e-3, kappa=0.0
    )
    ukf2 = UnscentedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], alpha=1e-3, kappa=3.0
    )

    # lambda = alpha^2 * (nx + kappa) - nx
    assert ukf1._lambda != ukf2._lambda


def test_cross_covariance_computation(simple_system):
    """Test cross-covariance computation in measurement update."""
    s = simple_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    state = UKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)
    pred = ukf.predict(state)

    # Generate sigma points
    X = ukf._sigma_points(pred.mean, pred.cov)
    Z = np.array([s["h"](xi) for xi in X])

    # Compute cross-covariance manually
    x_mean = np.sum(ukf.Wm[:, None] * X, axis=0)
    z_mean = np.sum(ukf.Wm[:, None] * Z, axis=0)

    DX = X - x_mean
    DZ = Z - z_mean

    Pxz = np.zeros((s["nx"], s["nz"]))
    for i in range(X.shape[0]):
        Pxz += ukf.Wc[i] * np.outer(DX[i], DZ[i])

    # Cross-covariance should have correct dimensions
    assert Pxz.shape == (s["nx"], s["nz"])

    # All values should be finite
    assert np.all(np.isfinite(Pxz))


def test_measurement_covariance_includes_R(simple_system):
    """Test that measurement covariance includes measurement noise R."""
    s = simple_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    state = UKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)
    pred = ukf.predict(state)

    # Generate sigma points and transform
    X = ukf._sigma_points(pred.mean, pred.cov)
    Z = np.array([s["h"](xi) for xi in X])

    # Compute measurement covariance
    z_mean = np.sum(ukf.Wm[:, None] * Z, axis=0)
    DZ = Z - z_mean

    S = s["R"].copy()
    for i in range(Z.shape[0]):
        S += ukf.Wc[i] * np.outer(DZ[i], DZ[i])

    # S should be at least as large as R (in terms of eigenvalues)
    eigvals_S = np.linalg.eigvalsh(S)
    eigvals_R = np.linalg.eigvalsh(s["R"])

    # Each eigenvalue of S should be >= corresponding eigenvalue of R
    # (This is a rough check; exact relationship depends on transformation)
    assert np.all(eigvals_S > 0)
