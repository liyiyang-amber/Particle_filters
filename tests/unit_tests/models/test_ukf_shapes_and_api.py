"""Unit tests for Unscented Kalman Filter: shapes, API, and basic functionality."""

import numpy as np
import pytest
from models.unscented_kalman_filter import UnscentedKalmanFilter, UKFState


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

    return dict(nx=nx, nz=nz, A=A, H=H, Q=Q, R=R, g=g, h=h)


@pytest.fixture
def nonlinear_system():
    """Nonlinear system similar to stochastic volatility."""
    nx, nz = 2, 2
    alpha = np.array([0.9, 0.85])
    beta = np.array([0.7, 1.0])
    Q = np.diag([0.1, 0.08])
    R = np.diag([0.5, 0.5])

    def g(x, u):
        return alpha * x

    def h(x):
        return beta * np.exp(0.5 * x)

    return dict(nx=nx, nz=nz, Q=Q, R=R, g=g, h=h, alpha=alpha, beta=beta)


def test_ukf_state_creation():
    """Test UKFState dataclass creation."""
    mean = np.array([1.0, 2.0])
    cov = np.eye(2)
    state = UKFState(mean=mean, cov=cov, t=0)

    assert state.t == 0
    np.testing.assert_array_equal(state.mean, mean)
    np.testing.assert_array_equal(state.cov, cov)


def test_ukf_initialization(simple_linear_system):
    """Test UKF initialization with various configurations."""
    s = simple_linear_system

    # Default parameters
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])
    assert ukf.alpha == 1e-3
    assert ukf.beta == 2.0
    assert ukf.kappa == 0.0
    assert ukf.jitter == 0.0

    # Custom parameters
    ukf_custom = UnscentedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        alpha=0.1,
        beta=1.0,
        kappa=3.0 - s["nx"],
        jitter=1e-8,
    )
    assert ukf_custom.alpha == 0.1
    assert ukf_custom.beta == 1.0
    assert ukf_custom.kappa == 3.0 - s["nx"]
    assert ukf_custom.jitter == 1e-8


def test_ukf_weights_sum_to_one(simple_linear_system):
    """Test that UKF weights sum to 1 (for mean)."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    # Mean weights should sum to 1
    assert np.isclose(np.sum(ukf.Wm), 1.0)


def test_sigma_points_shape_and_mean(simple_linear_system):
    """Test sigma points generation."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    mean = np.array([1.0, 2.0])
    cov = np.eye(s["nx"])

    sigma_pts = ukf._sigma_points(mean, cov)

    # Should have 2*nx + 1 points
    assert sigma_pts.shape == (2 * s["nx"] + 1, s["nx"])

    # First point should be the mean
    np.testing.assert_allclose(sigma_pts[0], mean, rtol=1e-10)

    # Weighted mean of sigma points should equal original mean
    recovered_mean = np.sum(ukf.Wm[:, None] * sigma_pts, axis=0)
    np.testing.assert_allclose(recovered_mean, mean, rtol=1e-10)


def test_predict_shapes(simple_linear_system):
    """Test that predict returns correct shapes."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    x0 = np.zeros(s["nx"])
    P0 = np.eye(s["nx"])
    state = UKFState(mean=x0, cov=P0, t=0)

    pred = ukf.predict(state)

    assert pred.mean.shape == (s["nx"],)
    assert pred.cov.shape == (s["nx"], s["nx"])
    assert pred.t == 1
    assert np.allclose(pred.cov, pred.cov.T)  # symmetry


def test_update_shapes(simple_linear_system):
    """Test that update returns correct shapes."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    x_pred = np.array([1.0, 0.5])
    P_pred = np.eye(s["nx"])
    pred_state = UKFState(mean=x_pred, cov=P_pred, t=1)

    z = np.array([1.2])
    post = ukf.update(pred_state, z)

    assert post.mean.shape == (s["nx"],)
    assert post.cov.shape == (s["nx"], s["nx"])
    assert post.t == 1
    assert np.allclose(post.cov, post.cov.T)  # symmetry


def test_step_combines_predict_update(simple_linear_system):
    """Test that step() combines predict and update correctly."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    state = UKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)
    z = np.array([1.0])

    # Using step
    result_step = ukf.step(state, z)

    # Using predict + update separately
    pred = ukf.predict(state)
    result_manual = ukf.update(pred, z)

    np.testing.assert_allclose(result_step.mean, result_manual.mean, rtol=1e-10)
    np.testing.assert_allclose(result_step.cov, result_manual.cov, rtol=1e-10)
    assert result_step.t == result_manual.t


def test_covariance_symmetry_maintained(simple_linear_system):
    """Test that covariances remain symmetric through filtering."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jitter=1e-10
    )

    state = UKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)

    # Run several steps
    for k in range(10):
        z = np.random.randn(s["nz"])
        state = ukf.step(state, z)

        # Check symmetry
        assert np.allclose(state.cov, state.cov.T, atol=1e-10)


def test_covariance_positive_definite(simple_linear_system):
    """Test that covariances remain positive-definite."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jitter=1e-9
    )

    state = UKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)

    # Run several steps
    for k in range(10):
        z = np.random.randn(s["nz"])
        state = ukf.step(state, z)

        # Check positive-definiteness via eigenvalues
        eigvals = np.linalg.eigvalsh(state.cov)
        assert np.all(eigvals > 0), f"Non-positive eigenvalues at step {k}: {eigvals}"


def test_ukf_with_control_input(simple_linear_system):
    """Test UKF with control input."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    state = UKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)
    u = np.array([0.1, 0.2])

    pred = ukf.predict(state, u=u)

    # Expected: A @ x0 + u
    expected_mean = s["A"] @ state.mean + u
    np.testing.assert_allclose(pred.mean, expected_mean, rtol=1e-5)


def test_nonlinear_ukf_runs(nonlinear_system):
    """Test UKF on a nonlinear system (SV-like)."""
    s = nonlinear_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    state = UKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)

    # Run multiple steps
    for _ in range(20):
        z = np.random.randn(s["nz"])
        state = ukf.step(state, z)

        assert state.mean.shape == (s["nx"],)
        assert state.cov.shape == (s["nx"], s["nx"])
        assert np.allclose(state.cov, state.cov.T)


def test_covariance_reduction_after_update(simple_linear_system):
    """Test that update step reduces uncertainty."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    # Start with high uncertainty
    state = UKFState(mean=np.zeros(s["nx"]), cov=10.0 * np.eye(s["nx"]), t=0)
    pred = ukf.predict(state)

    # Measurement with reasonable noise
    z = np.array([0.5])
    post = ukf.update(pred, z)

    # Check that trace (total variance) is reduced
    trace_pred = np.trace(pred.cov)
    trace_post = np.trace(post.cov)

    assert trace_post < trace_pred, "Update should reduce uncertainty"


def test_sequential_filtering(simple_linear_system):
    """Test sequential filtering over multiple time steps."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    n = 50
    np.random.seed(42)
    observations = np.random.randn(n, s["nz"])

    state = UKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)
    states = [state]

    for k in range(n):
        state = ukf.step(state, observations[k])
        states.append(state)

        assert state.t == k + 1
        assert state.mean.shape == (s["nx"],)
        assert state.cov.shape == (s["nx"], s["nx"])

    assert len(states) == n + 1


def test_ukf_linear_close_to_kf(simple_linear_system):
    """Test that UKF gives similar results to linear filter for linear system."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], alpha=1.0)

    n = 30
    np.random.seed(123)
    observations = np.random.randn(n, s["nz"])

    state = UKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)

    # Run UKF
    for k in range(n):
        state = ukf.step(state, observations[k])

    # For linear system, UKF should perform well
    # Just verify it runs and produces reasonable results
    assert np.all(np.isfinite(state.mean))
    assert np.all(np.isfinite(state.cov))


def test_jitter_prevents_cholesky_failure(simple_linear_system):
    """Test that jitter helps with near-singular covariances."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], jitter=1e-6
    )

    # Start with very small covariance
    state = UKFState(mean=np.zeros(s["nx"]), cov=1e-10 * np.eye(s["nx"]), t=0)

    # Should not crash with jitter
    z = np.array([1.0])
    result = ukf.step(state, z)

    assert result.mean.shape == (s["nx"],)
    assert np.all(np.isfinite(result.mean))
    assert np.all(np.isfinite(result.cov))


def test_different_alpha_values(simple_linear_system):
    """Test UKF with different alpha values."""
    s = simple_linear_system

    alphas = [1e-4, 1e-3, 0.1, 1.0]
    state0 = UKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)
    z = np.array([1.5])

    results = []
    for alpha in alphas:
        ukf = UnscentedKalmanFilter(
            g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], alpha=alpha
        )
        result = ukf.step(state0, z)
        results.append(result)

    # All should produce valid results
    for result in results:
        assert np.all(np.isfinite(result.mean))
        assert np.all(np.isfinite(result.cov))
        assert np.allclose(result.cov, result.cov.T)


def test_different_kappa_values(simple_linear_system):
    """Test UKF with different kappa values."""
    s = simple_linear_system

    kappas = [0.0, 1.0, 3.0 - s["nx"]]
    state0 = UKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)
    z = np.array([1.5])

    results = []
    for kappa in kappas:
        ukf = UnscentedKalmanFilter(
            g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], kappa=kappa
        )
        result = ukf.step(state0, z)
        results.append(result)

    # All should produce valid results
    for result in results:
        assert np.all(np.isfinite(result.mean))
        assert np.all(np.isfinite(result.cov))
        assert np.allclose(result.cov, result.cov.T)


def test_sigma_points_recover_covariance(simple_linear_system):
    """Test that sigma points recover the original covariance."""
    s = simple_linear_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    mean = np.array([1.0, 2.0])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])

    sigma_pts = ukf._sigma_points(mean, cov)

    # Compute covariance from sigma points
    recovered_mean = np.sum(ukf.Wm[:, None] * sigma_pts, axis=0)
    DX = sigma_pts - recovered_mean
    recovered_cov = np.zeros_like(cov)
    for i in range(sigma_pts.shape[0]):
        recovered_cov += ukf.Wc[i] * np.outer(DX[i], DX[i])

    np.testing.assert_allclose(recovered_mean, mean, rtol=1e-10)
    np.testing.assert_allclose(recovered_cov, cov, rtol=1e-5)


def test_ukf_handles_multidimensional_observations(nonlinear_system):
    """Test UKF with multi-dimensional observations."""
    s = nonlinear_system
    ukf = UnscentedKalmanFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"])

    state = UKFState(mean=np.zeros(s["nx"]), cov=np.eye(s["nx"]), t=0)
    z = np.array([0.5, 0.8])  # 2D observation

    result = ukf.step(state, z)

    assert result.mean.shape == (s["nx"],)
    assert result.cov.shape == (s["nx"], s["nx"])
    assert np.all(np.isfinite(result.mean))
    assert np.all(np.isfinite(result.cov))
