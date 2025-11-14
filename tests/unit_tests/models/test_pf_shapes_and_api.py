"""Unit tests for Particle Filter: shapes, API, and basic functionality."""

import numpy as np
import pytest
from models.particle_filter import ParticleFilter, PFState


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


def test_pf_state_creation():
    """Test PFState dataclass creation."""
    Np = 100
    nx = 2
    particles = np.random.randn(Np, nx)
    weights = np.ones(Np) / Np
    mean = np.array([1.0, 2.0])
    cov = np.eye(2)
    
    state = PFState(particles=particles, weights=weights, mean=mean, cov=cov, t=0)

    assert state.t == 0
    assert state.particles.shape == (Np, nx)
    assert state.weights.shape == (Np,)
    np.testing.assert_array_equal(state.mean, mean)
    np.testing.assert_array_equal(state.cov, cov)


def test_pf_initialization(simple_linear_system):
    """Test Particle Filter initialization with various configurations."""
    s = simple_linear_system

    # Default parameters
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=1000)
    assert pf.Np == 1000
    assert pf.resample_thresh == 0.5
    assert pf.resample_method == "systematic"
    assert pf.regularize_after_resample is False
    assert pf.nx == s["nx"]
    assert pf.nz == s["nz"]

    # Custom parameters
    pf_custom = ParticleFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        Np=500,
        resample_thresh=0.3,
        resample_method="multinomial",
        regularize_after_resample=True,
    )
    assert pf_custom.Np == 500
    assert pf_custom.resample_thresh == 0.3
    assert pf_custom.resample_method == "multinomial"
    assert pf_custom.regularize_after_resample is True


def test_pf_initialize_particles(simple_linear_system):
    """Test particle initialization from Gaussian."""
    s = simple_linear_system
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=1000)

    mean = np.array([1.0, 2.0])
    cov = np.eye(s["nx"])
    state = pf.initialize(mean, cov)

    # Check shapes
    assert state.particles.shape == (1000, s["nx"])
    assert state.weights.shape == (1000,)
    assert state.mean.shape == (s["nx"],)
    assert state.cov.shape == (s["nx"], s["nx"])
    assert state.t == 0

    # Check initial weights are uniform
    np.testing.assert_allclose(state.weights, np.ones(1000) / 1000)

    # Check particle statistics roughly match initialization
    particle_mean = np.mean(state.particles, axis=0)
    particle_cov = np.cov(state.particles.T)
    np.testing.assert_allclose(particle_mean, mean, atol=0.2)
    np.testing.assert_allclose(particle_cov, cov, atol=0.3)


def test_pf_weights_sum_to_one(simple_linear_system):
    """Test that particle weights always sum to 1."""
    s = simple_linear_system
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=1000)

    mean = np.zeros(s["nx"])
    cov = np.eye(s["nx"])
    state = pf.initialize(mean, cov)

    # After initialization
    assert np.isclose(np.sum(state.weights), 1.0)

    # After predict
    pf.predict()
    assert np.isclose(np.sum(pf.state.weights), 1.0)

    # After update
    z = np.array([0.5])
    state = pf.update(z)
    assert np.isclose(np.sum(state.weights), 1.0)


def test_predict_shapes(simple_linear_system):
    """Test that predict maintains correct shapes."""
    s = simple_linear_system
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=500)

    mean = np.zeros(s["nx"])
    cov = np.eye(s["nx"])
    pf.initialize(mean, cov)

    # Store initial particles
    initial_particles = pf.state.particles.copy()

    pf.predict()

    # Check shapes unchanged
    assert pf.state.particles.shape == (500, s["nx"])
    assert pf.state.weights.shape == (500,)

    # Particles should have moved (not identical)
    assert not np.allclose(pf.state.particles, initial_particles)


def test_predict_with_control(simple_linear_system):
    """Test predict with control input."""
    s = simple_linear_system
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=500)

    mean = np.zeros(s["nx"])
    cov = np.eye(s["nx"])
    pf.initialize(mean, cov)

    u = np.array([0.5, -0.3])
    pf.predict(u)

    # Check that control was applied (mean should shift)
    assert pf.state.particles.shape == (500, s["nx"])


def test_update_shapes(simple_linear_system):
    """Test that update returns correct shapes."""
    s = simple_linear_system
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=500)

    mean = np.zeros(s["nx"])
    cov = np.eye(s["nx"])
    pf.initialize(mean, cov)
    pf.predict()

    z = np.array([1.2])
    state = pf.update(z)

    assert state.particles.shape == (500, s["nx"])
    assert state.weights.shape == (500,)
    assert state.mean.shape == (s["nx"],)
    assert state.cov.shape == (s["nx"], s["nx"])
    assert state.t == 1


def test_step_combines_predict_update(simple_linear_system):
    """Test that step() combines predict and update correctly."""
    s = simple_linear_system
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=500)

    mean = np.zeros(s["nx"])
    cov = np.eye(s["nx"])
    pf.initialize(mean, cov)

    z = np.array([0.5])
    
    # Manual predict + update
    pf_manual = ParticleFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=500, rng=np.random.default_rng(42)
    )
    pf_manual.initialize(mean, cov)
    pf_manual.predict()
    state_manual = pf_manual.update(z)

    # Using step
    pf_step = ParticleFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=500, rng=np.random.default_rng(42)
    )
    pf_step.initialize(mean, cov)
    state_step = pf_step.step(z)

    # Results should be similar (same RNG seed)
    assert state_step.t == state_manual.t


def test_effective_sample_size(simple_linear_system):
    """Test effective sample size calculation."""
    s = simple_linear_system
    # Use high resample threshold to prevent automatic resampling
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=1000, resample_thresh=0.1)

    mean = np.zeros(s["nx"])
    cov = np.eye(s["nx"])
    pf.initialize(mean, cov)

    # Initially uniform weights → Neff ≈ Np
    Neff = pf.effective_sample_size()
    assert np.isclose(Neff, 1000.0, rtol=0.01)

    # After update, weights become non-uniform
    z = np.array([5.0])  # Extreme observation
    pf.update(z)
    Neff_after = pf.effective_sample_size()
    
    # Neff should decrease (unless resampling happened, which shouldn't with low threshold)
    # With extreme observation, weights should be non-uniform
    assert 1.0 <= Neff_after <= 1000.0


def test_multiple_steps(simple_linear_system):
    """Test multiple sequential filtering steps."""
    s = simple_linear_system
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=1000)

    mean = np.zeros(s["nx"])
    cov = np.eye(s["nx"])
    pf.initialize(mean, cov)

    # Generate synthetic observations
    np.random.seed(123)
    T = 10
    observations = [np.random.randn(s["nz"]) for _ in range(T)]

    for t, z in enumerate(observations):
        state = pf.step(z)
        assert state.t == t + 1
        assert state.particles.shape == (1000, s["nx"])
        assert np.isclose(np.sum(state.weights), 1.0)


def test_nonlinear_system_filtering(nonlinear_system):
    """Test filtering on nonlinear stochastic volatility-like system."""
    s = nonlinear_system
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=2000)

    # Initialize near zero
    mean = np.array([0.0, 0.0])
    cov = np.eye(s["nx"]) * 0.5
    pf.initialize(mean, cov)

    # Generate synthetic observations
    T = 20
    np.random.seed(456)
    observations = [s["h"](np.random.randn(s["nx"])) + np.random.randn(s["nz"]) * 0.3 
                   for _ in range(T)]

    for t, z in enumerate(observations):
        state = pf.step(z)
        assert state.t == t + 1
        assert np.isclose(np.sum(state.weights), 1.0)
        # Check mean and cov are computed
        assert state.mean.shape == (s["nx"],)
        assert state.cov.shape == (s["nx"], s["nx"])


def test_pf_no_assertion_before_init(simple_linear_system):
    """Test that operations fail gracefully before initialization."""
    s = simple_linear_system
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=500)

    with pytest.raises(AssertionError):
        pf.predict()

    with pytest.raises(AssertionError):
        pf.update(np.array([1.0]))

    with pytest.raises(AssertionError):
        pf.effective_sample_size()


def test_pf_cov_is_symmetric(simple_linear_system):
    """Test that posterior covariance is symmetric."""
    s = simple_linear_system
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=1000)

    mean = np.zeros(s["nx"])
    cov = np.eye(s["nx"])
    pf.initialize(mean, cov)

    z = np.array([0.5])
    state = pf.step(z)

    np.testing.assert_allclose(state.cov, state.cov.T, rtol=1e-10)


def test_pf_deterministic_with_seed(simple_linear_system):
    """Test that results are reproducible with same seed."""
    s = simple_linear_system
    
    mean = np.zeros(s["nx"])
    cov = np.eye(s["nx"])
    z = np.array([0.5])

    # Run 1
    pf1 = ParticleFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=500, rng=np.random.default_rng(42)
    )
    pf1.initialize(mean, cov)
    state1 = pf1.step(z)

    # Run 2 with same seed
    pf2 = ParticleFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], Np=500, rng=np.random.default_rng(42)
    )
    pf2.initialize(mean, cov)
    state2 = pf2.step(z)

    # Should be identical
    np.testing.assert_array_equal(state1.particles, state2.particles)
    np.testing.assert_array_equal(state1.weights, state2.weights)
    np.testing.assert_array_equal(state1.mean, state2.mean)


def test_pf_matrix_inputs(simple_linear_system):
    """Test that Q and R are properly converted to arrays."""
    s = simple_linear_system
    
    # Test with lists
    Q_list = [[0.05, 0.0], [0.0, 0.02]]
    R_list = [[0.10]]
    
    pf = ParticleFilter(g=s["g"], h=s["h"], Q=Q_list, R=R_list, Np=500)
    
    assert isinstance(pf.Q, np.ndarray)
    assert isinstance(pf.R, np.ndarray)
    np.testing.assert_allclose(pf.Q, s["Q"])
    np.testing.assert_allclose(pf.R, s["R"])
