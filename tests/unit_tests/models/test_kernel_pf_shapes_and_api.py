"""Unit tests for Kernel Particle Filter: shapes, API, and basic functionality."""

import numpy as np
import pytest
from models.kernel_particle_filter import (
    KernelParticleFilter,
    Model,
    KPFConfig,
    KPFState,
)


@pytest.fixture
def simple_linear_system():
    """Simple linear system for testing: y = Hx + v."""
    nx, ny = 2, 1
    H = np.array([[1.0, 0.5]])
    R = np.array([[0.10]])

    def h(x):
        return H @ x

    def jh(x):
        return H

    return dict(nx=nx, ny=ny, H=H, R=R, h=h, jh=jh)


@pytest.fixture
def nonlinear_system():
    """Nonlinear observation system."""
    nx, ny = 2, 2
    R = np.diag([0.5, 0.5])

    def h(x):
        return np.array([x[0]**2, np.sin(x[1])])

    def jh(x):
        return np.array([
            [2*x[0], 0.0],
            [0.0, np.cos(x[1])],
        ])

    return dict(nx=nx, ny=ny, R=R, h=h, jh=jh)


def test_kpf_config_creation():
    """Test KPFConfig dataclass creation with defaults."""
    config = KPFConfig()

    assert config.ds_init == 0.2
    assert config.ds_min == 1e-3
    assert config.c_move_max == 2.0
    assert config.min_steps == 5
    assert config.max_steps == 100
    assert config.kernel_type == "diagonal"
    assert config.lengthscale_mode == "std"
    assert config.fixed_lengthscale == 1.0
    assert config.reg == 1e-6
    assert config.localization_radius == np.inf
    assert config.random_order is True


def test_kpf_config_custom_values():
    """Test KPFConfig with custom values."""
    config = KPFConfig(
        ds_init=0.1,
        max_steps=50,
        kernel_type="scalar",
        lengthscale_mode="fixed",
        fixed_lengthscale=2.0,
        localization_radius=5.0,
        random_order=False,
    )

    assert config.ds_init == 0.1
    assert config.max_steps == 50
    assert config.kernel_type == "scalar"
    assert config.lengthscale_mode == "fixed"
    assert config.fixed_lengthscale == 2.0
    assert config.localization_radius == 5.0
    assert config.random_order is False


def test_kpf_state_creation():
    """Test KPFState dataclass creation."""
    Np = 100
    nx = 2
    particles = np.random.randn(Np, nx)
    weights = np.ones(Np) / Np
    s = 0.5
    steps = 10

    state = KPFState(particles=particles, weights=weights, s=s, steps=steps)

    assert state.particles.shape == (Np, nx)
    assert state.weights.shape == (Np,)
    assert state.s == 0.5
    assert state.steps == 10


def test_model_creation(simple_linear_system):
    """Test Model dataclass creation."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])

    # Test observation function
    x = np.array([1.0, 2.0])
    y = model.H(x)
    assert y.shape == (s["ny"],)
    np.testing.assert_allclose(y, [2.0])  # 1*1 + 0.5*2 = 2

    # Test Jacobian
    J = model.JH(x)
    assert J.shape == (s["ny"], s["nx"])
    np.testing.assert_allclose(J, s["H"])

    # Test R
    assert model.R.shape == (s["ny"], s["ny"])


def test_kpf_initialization(simple_linear_system):
    """Test Kernel Particle Filter initialization."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])

    # Default config
    kpf = KernelParticleFilter(model=model)
    assert kpf.model == model
    assert isinstance(kpf.cfg, KPFConfig)

    # Custom config
    config = KPFConfig(max_steps=50, kernel_type="scalar")
    kpf_custom = KernelParticleFilter(model=model, config=config)
    assert kpf_custom.cfg.max_steps == 50
    assert kpf_custom.cfg.kernel_type == "scalar"


def test_mean_and_cov_computation():
    """Test mean and covariance computation."""
    kpf = KernelParticleFilter(
        model=Model(H=lambda x: x, JH=lambda x: np.eye(2), R=np.eye(2))
    )

    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ])

    mu, B = kpf.mean_and_cov(X)

    expected_mu = np.array([2.0, 3.0])
    np.testing.assert_allclose(mu, expected_mu)

    # Covariance should be positive semi-definite
    assert np.all(np.linalg.eigvals(B) >= -1e-10)


def test_mean_and_cov_with_regularization():
    """Test mean and covariance with regularization."""
    kpf = KernelParticleFilter(
        model=Model(H=lambda x: x, JH=lambda x: np.eye(2), R=np.eye(2))
    )

    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ])

    reg = 0.1
    mu, B = kpf.mean_and_cov(X, reg=reg)

    # Diagonal should have regularization added
    # The regularization ensures positive definiteness
    eigvals = np.linalg.eigvals(B)
    assert np.all(eigvals > 0)


def test_analyze_shapes(simple_linear_system):
    """Test that analyze returns correct shapes."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    kpf = KernelParticleFilter(model=model)

    Np = 100
    nx = s["nx"]
    X = np.random.randn(Np, nx)
    y = np.array([0.5])

    result = kpf.analyze(X, y)

    # Check result is KPFState
    assert isinstance(result, KPFState)

    # Check shapes
    assert result.particles.shape == (Np, nx)
    assert result.weights.shape == (Np,)

    # Check pseudo-time reached 1.0 or close
    assert 0.99 <= result.s <= 1.0

    # Check steps taken
    assert result.steps >= 1


def test_analyze_reaches_pseudo_time_one(simple_linear_system):
    """Test that analyze integrates to pseudo-time s=1."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    config = KPFConfig(min_steps=5, max_steps=100)
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 100
    X = np.random.randn(Np, s["nx"])
    y = np.array([0.5])

    result = kpf.analyze(X, y)

    # Should reach s=1
    assert np.isclose(result.s, 1.0, atol=1e-6)


def test_analyze_respects_min_steps(simple_linear_system):
    """Test that analyze takes at least min_steps."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    config = KPFConfig(min_steps=10, ds_init=0.05)  # Small step size to need more steps
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 100
    X = np.random.randn(Np, s["nx"])
    y = np.array([0.5])

    result = kpf.analyze(X, y)

    # Should reach min_steps
    assert result.steps >= config.min_steps


def test_analyze_respects_max_steps(simple_linear_system):
    """Test that analyze doesn't exceed max_steps."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    config = KPFConfig(max_steps=20)
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 100
    X = np.random.randn(Np, s["nx"])
    y = np.array([0.5])

    result = kpf.analyze(X, y)

    assert result.steps <= 20


def test_analyze_with_diagonal_kernel(simple_linear_system):
    """Test analyze with diagonal (matrix-valued) kernel."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    config = KPFConfig(kernel_type="diagonal")
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 100
    X = np.random.randn(Np, s["nx"])
    y = np.array([0.5])

    result = kpf.analyze(X, y)

    assert result.particles.shape == (Np, s["nx"])
    assert np.isclose(result.s, 1.0, atol=1e-6)


def test_analyze_with_scalar_kernel(simple_linear_system):
    """Test analyze with scalar (isotropic) kernel."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    config = KPFConfig(kernel_type="scalar")
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 100
    X = np.random.randn(Np, s["nx"])
    y = np.array([0.5])

    result = kpf.analyze(X, y)

    assert result.particles.shape == (Np, s["nx"])
    assert np.isclose(result.s, 1.0, atol=1e-6)


def test_analyze_with_custom_lengthscales(simple_linear_system):
    """Test analyze with custom lengthscales."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    kpf = KernelParticleFilter(model=model)

    Np = 100
    X = np.random.randn(Np, s["nx"])
    y = np.array([0.5])

    lengthscales = np.array([1.5, 2.0])
    result = kpf.analyze(X, y, lengthscales=lengthscales)

    assert result.particles.shape == (Np, s["nx"])


def test_analyze_with_fixed_lengthscale_mode(simple_linear_system):
    """Test analyze with fixed lengthscale mode."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    config = KPFConfig(lengthscale_mode="fixed", fixed_lengthscale=2.5)
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 100
    X = np.random.randn(Np, s["nx"])
    y = np.array([0.5])

    result = kpf.analyze(X, y)

    assert result.particles.shape == (Np, s["nx"])


def test_analyze_deterministic_with_seed(simple_linear_system):
    """Test that analyze is reproducible with same RNG seed."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    config = KPFConfig(random_order=True)
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 100
    X = np.random.randn(Np, s["nx"])
    y = np.array([0.5])

    rng1 = np.random.default_rng(42)
    result1 = kpf.analyze(X.copy(), y, rng=rng1)

    rng2 = np.random.default_rng(42)
    result2 = kpf.analyze(X.copy(), y, rng=rng2)

    # Should be identical
    np.testing.assert_array_equal(result1.particles, result2.particles)
    assert result1.steps == result2.steps


def test_analyze_without_random_order(simple_linear_system):
    """Test analyze with random_order=False is deterministic."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    config = KPFConfig(random_order=False)
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 100
    X = np.random.randn(Np, s["nx"])
    y = np.array([0.5])

    result1 = kpf.analyze(X.copy(), y)
    result2 = kpf.analyze(X.copy(), y)

    # Should be identical
    np.testing.assert_array_equal(result1.particles, result2.particles)


def test_analyze_with_localization(simple_linear_system):
    """Test analyze with localization."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    config = KPFConfig(localization_radius=5.0)
    kpf = KernelParticleFilter(model=model, config=config)

    Np = 100
    X = np.random.randn(Np, s["nx"])
    y = np.array([0.5])

    result = kpf.analyze(X, y)

    assert result.particles.shape == (Np, s["nx"])


def test_nonlinear_system_analysis(nonlinear_system):
    """Test filtering on nonlinear system."""
    s = nonlinear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    kpf = KernelParticleFilter(model=model)

    Np = 200
    X = np.random.randn(Np, s["nx"])
    y = np.array([0.5, 0.3])

    result = kpf.analyze(X, y)

    assert result.particles.shape == (Np, s["nx"])
    assert np.isclose(result.s, 1.0, atol=1e-6)


def test_mahalanobis_distance_computation(simple_linear_system):
    """Test Mahalanobis distance computation."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    kpf = KernelParticleFilter(model=model)

    dx = np.array([1.0, 2.0])
    B_inv = np.eye(2)

    dist = kpf._mahalanobis(dx, B_inv)

    expected = np.sqrt(1.0**2 + 2.0**2)
    assert np.isclose(dist, expected)


def test_weights_remain_normalized(simple_linear_system):
    """Test that weights sum to 1 after analysis."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    kpf = KernelParticleFilter(model=model)

    Np = 100
    X = np.random.randn(Np, s["nx"])
    y = np.array([0.5])

    result = kpf.analyze(X, y)

    # Weights should sum to 1
    assert np.isclose(np.sum(result.weights), 1.0)


def test_particles_move_toward_observation(simple_linear_system):
    """Test that particles move in response to observation."""
    s = simple_linear_system
    model = Model(H=s["h"], JH=s["jh"], R=s["R"])
    kpf = KernelParticleFilter(model=model)

    Np = 100
    # Initialize far from observation
    X_init = np.ones((Np, s["nx"])) * 10.0
    y = np.array([0.5])

    result = kpf.analyze(X_init.copy(), y)

    # Particles should have moved 
    assert not np.allclose(result.particles, X_init)

    # Mean should shift toward observation-compatible values
    mean_init = X_init.mean(axis=0)
    mean_final = result.particles.mean(axis=0)
    assert not np.allclose(mean_init, mean_final)
