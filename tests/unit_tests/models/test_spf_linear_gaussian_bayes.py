"""Unit tests for SPF LinearGaussianBayes helper class."""

import numpy as np
import pytest
from models.Stochastic_particle_filter import LinearGaussianBayes


@pytest.fixture
def simple_lg_model():
    """Simple 2D linear-Gaussian model."""
    n, d = 2, 1
    m0 = np.array([1.0, 2.0])
    P0 = np.array([[2.0, 0.5], [0.5, 1.0]])
    H = np.array([[1.0, 0.5]])
    R = np.array([[0.5]])
    z = np.array([3.0])
    return dict(m0=m0, P0=P0, H=H, R=R, z=z, n=n, d=d)


@pytest.fixture
def lg_model_instance(simple_lg_model):
    """Create LinearGaussianBayes instance."""
    s = simple_lg_model
    return LinearGaussianBayes(m0=s["m0"], P0=s["P0"], H=s["H"], R=s["R"], z=s["z"])


def test_lg_initialization(simple_lg_model):
    """Test LinearGaussianBayes initialization."""
    s = simple_lg_model
    model = LinearGaussianBayes(m0=s["m0"], P0=s["P0"], H=s["H"], R=s["R"], z=s["z"])
    
    assert model.n == s["n"]
    assert model.d == s["d"]
    assert model.m0.shape == (s["n"],)
    assert model.z.shape == (s["d"],)
    assert model.P0.shape == (s["n"], s["n"])
    assert model.H.shape == (s["d"], s["n"])
    assert model.R.shape == (s["d"], s["d"])


def test_lg_initialization_with_lists():
    """Test that lists are correctly converted to arrays."""
    model = LinearGaussianBayes(
        m0=[1.0, 2.0],
        P0=[[2.0, 0.5], [0.5, 1.0]],
        H=[[1.0, 0.5]],
        R=[[0.5]],
        z=[3.0]
    )
    
    assert isinstance(model.m0, np.ndarray)
    assert isinstance(model.P0, np.ndarray)
    assert isinstance(model.H, np.ndarray)
    assert isinstance(model.R, np.ndarray)
    assert isinstance(model.z, np.ndarray)


def test_lg_precomputed_matrices(lg_model_instance):
    """Test that precomputed matrices are calculated correctly."""
    model = lg_model_instance
    
    # Check that P0_inv is actually the inverse
    P0_inv_check = np.linalg.inv(model.P0)
    np.testing.assert_allclose(model.P0_inv, P0_inv_check, rtol=1e-10)
    
    # Check that R_inv is actually the inverse
    R_inv_check = np.linalg.inv(model.R)
    np.testing.assert_allclose(model.R_inv, R_inv_check, rtol=1e-10)
    
    # Check Hessian of log p0
    expected_Hess_log_p0 = -model.P0_inv
    np.testing.assert_allclose(model.Hess_log_p0, expected_Hess_log_p0, rtol=1e-10)
    
    # Check Hessian of log h
    expected_Hess_log_h = -(model.H.T @ model.R_inv @ model.H)
    np.testing.assert_allclose(model.Hess_log_h, expected_Hess_log_h, rtol=1e-10)


def test_lg_M_matrices_spd(lg_model_instance):
    """Test that M0 and Mh are symmetric and positive (semi)definite."""
    model = lg_model_instance
    
    # M0 should be SPD (symmetric positive definite)
    np.testing.assert_allclose(model.M0, model.M0.T, rtol=1e-10)
    eigvals_M0 = np.linalg.eigvalsh(model.M0)
    assert np.all(eigvals_M0 > 0), "M0 should be positive definite"
    
    # Mh should be PSD (symmetric positive semi-definite)
    np.testing.assert_allclose(model.Mh, model.Mh.T, rtol=1e-10)
    eigvals_Mh = np.linalg.eigvalsh(model.Mh)
    assert np.all(eigvals_Mh >= -1e-10), "Mh should be positive semi-definite"


def test_lg_grad_log_p0(lg_model_instance):
    """Test gradient of log prior."""
    model = lg_model_instance
    
    # At the mean, gradient should be zero
    grad_at_mean = model.grad_log_p0(model.m0)
    np.testing.assert_allclose(grad_at_mean, np.zeros(model.n), atol=1e-10)
    
    # Test at a different point
    x = np.array([2.0, 3.0])
    grad = model.grad_log_p0(x)
    
    # Manually compute expected gradient
    expected_grad = -model.P0_inv @ (x - model.m0)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-10)
    
    # Check shape
    assert grad.shape == (model.n,)


def test_lg_grad_log_h(lg_model_instance):
    """Test gradient of log likelihood."""
    model = lg_model_instance
    
    x = np.array([2.0, 3.0])
    grad = model.grad_log_h(x)
    
    # Manually compute expected gradient
    expected_grad = model.H.T @ (model.R_inv @ (model.z - model.H @ x))
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-10)
    
    # Check shape
    assert grad.shape == (model.n,)


def test_lg_kalman_posterior(lg_model_instance):
    """Test analytic Kalman posterior calculation."""
    model = lg_model_instance
    
    m_post, P_post = model.kalman_posterior()
    
    # Check shapes
    assert m_post.shape == (model.n,)
    assert P_post.shape == (model.n, model.n)
    
    # Check that P_post is symmetric
    np.testing.assert_allclose(P_post, P_post.T, rtol=1e-10)
    
    # Check that P_post is positive definite
    eigvals = np.linalg.eigvalsh(P_post)
    assert np.all(eigvals > 0), "Posterior covariance should be positive definite"
    
    # Manually verify Kalman update
    S = model.H @ model.P0 @ model.H.T + model.R
    K = model.P0 @ model.H.T @ np.linalg.solve(S, np.eye(model.d))
    m_post_expected = model.m0 + K @ (model.z - model.H @ model.m0)
    P_post_expected = (np.eye(model.n) - K @ model.H) @ model.P0
    
    np.testing.assert_allclose(m_post, m_post_expected, rtol=1e-10)
    np.testing.assert_allclose(P_post, P_post_expected, rtol=1e-10)


def test_lg_kalman_posterior_reduces_uncertainty(lg_model_instance):
    """Test that Kalman posterior has smaller covariance than prior."""
    model = lg_model_instance
    
    _, P_post = model.kalman_posterior()
    
    # Posterior covariance should be smaller (in terms of trace)
    trace_prior = np.trace(model.P0)
    trace_post = np.trace(P_post)
    assert trace_post < trace_prior, "Posterior should reduce uncertainty"


def test_lg_1d_case():
    """Test 1D scalar case."""
    model = LinearGaussianBayes(
        m0=np.array([5.0]),
        P0=np.array([[2.0]]),
        H=np.array([[1.0]]),
        R=np.array([[1.0]]),
        z=np.array([7.0])
    )
    
    assert model.n == 1
    assert model.d == 1
    
    # Verify posterior calculation for 1D case
    m_post, P_post = model.kalman_posterior()
    
    # Analytical solution for 1D
    # P_post = 1/(1/P0 + H^2/R) = 1/(1/2 + 1/1) = 2/3
    # m_post = P_post * (m0/P0 + Hz/R) = (2/3) * (5/2 + 7/1) = (2/3) * (2.5 + 7) = 6.333...
    expected_P_post = 2.0 / 3.0
    expected_m_post = expected_P_post * (5.0 / 2.0 + 7.0 / 1.0)
    
    np.testing.assert_allclose(P_post[0, 0], expected_P_post, rtol=1e-10)
    np.testing.assert_allclose(m_post[0], expected_m_post, rtol=1e-10)


def test_lg_high_dimensional():
    """Test with higher dimensional state (n=5, d=3)."""
    n, d = 5, 3
    m0 = np.random.randn(n)
    P0 = np.eye(n) * 2.0  # Simple diagonal for stability
    H = np.random.randn(d, n)
    R = np.eye(d) * 0.5
    z = np.random.randn(d)
    
    model = LinearGaussianBayes(m0=m0, P0=P0, H=H, R=R, z=z)
    
    assert model.n == n
    assert model.d == d
    
    # Test that all operations work
    x = np.random.randn(n)
    grad_p0 = model.grad_log_p0(x)
    grad_h = model.grad_log_h(x)
    m_post, P_post = model.kalman_posterior()
    
    assert grad_p0.shape == (n,)
    assert grad_h.shape == (n,)
    assert m_post.shape == (n,)
    assert P_post.shape == (n, n)


def test_lg_symmetrization():
    """Test that matrices are properly symmetrized."""
    # Create slightly asymmetric matrix (might happen due to numerical errors)
    P0 = np.array([[2.0, 0.5 + 1e-15], [0.5, 1.0]])
    
    model = LinearGaussianBayes(
        m0=np.array([1.0, 2.0]),
        P0=P0,
        H=np.array([[1.0, 0.5]]),
        R=np.array([[0.5]]),
        z=np.array([3.0])
    )
    
    # All Hessian and M matrices should be symmetric
    np.testing.assert_allclose(model.Hess_log_p0, model.Hess_log_p0.T, rtol=1e-10)
    np.testing.assert_allclose(model.Hess_log_h, model.Hess_log_h.T, rtol=1e-10)
    np.testing.assert_allclose(model.M0, model.M0.T, rtol=1e-10)
    np.testing.assert_allclose(model.Mh, model.Mh.T, rtol=1e-10)


def test_lg_invalid_dimensions():
    """Test that dimension mismatches raise errors."""
    with pytest.raises(AssertionError):
        # P0 wrong shape
        LinearGaussianBayes(
            m0=np.array([1.0, 2.0]),
            P0=np.array([[2.0]]),  # Should be 2x2
            H=np.array([[1.0, 0.5]]),
            R=np.array([[0.5]]),
            z=np.array([3.0])
        )
    
    with pytest.raises(AssertionError):
        # H wrong shape
        LinearGaussianBayes(
            m0=np.array([1.0, 2.0]),
            P0=np.eye(2),
            H=np.array([[1.0]]),  # Should have 2 columns
            R=np.array([[0.5]]),
            z=np.array([3.0])
        )
    
    with pytest.raises(AssertionError):
        # R wrong shape
        LinearGaussianBayes(
            m0=np.array([1.0, 2.0]),
            P0=np.eye(2),
            H=np.array([[1.0, 0.5]]),
            R=np.array([[0.5, 0.1], [0.1, 0.5]]),  # Should be 1x1
            z=np.array([3.0])
        )
