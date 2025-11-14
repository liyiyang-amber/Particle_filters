import numpy as np
import pytest
from numpy.linalg import LinAlgError
from simulator.simulator_sensor_network_skewt_dynamic import (
    make_lattice,
    build_spatial_cov,
    cholesky_psd,
    sample_inverse_gamma,
    prepare_gamma_vector,
)

# ===== Tests for make_lattice =====

def test_make_lattice_shape_4x4():
    """Test that make_lattice returns correct shape for a 4x4 grid."""
    d = 16
    R = make_lattice(d)
    assert R.shape == (16, 2)


def test_make_lattice_shape_8x8():
    """Test that make_lattice returns correct shape for an 8x8 grid."""
    d = 64
    R = make_lattice(d)
    assert R.shape == (64, 2)


def test_make_lattice_dtype():
    """Test that coordinates are float type."""
    R = make_lattice(16)
    assert R.dtype == np.float64 or R.dtype == float


def test_make_lattice_values_2x2():
    """Test exact coordinate values for a small 2x2 grid."""
    d = 4
    R = make_lattice(d)
    
    # Expected coordinates row-major order from meshgrid
    expected = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    
    assert np.allclose(R, expected)


def test_make_lattice_values_3x3():
    """Test exact coordinate values for a 3x3 grid."""
    d = 9
    R = make_lattice(d)
    
    expected = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
    ])
    
    assert np.allclose(R, expected)


def test_make_lattice_range():
    """Test that coordinates are in the expected range."""
    d = 64
    R = make_lattice(d)
    
    # For 8x8 grid, coordinates should be 0-7 in both dimensions
    assert R.min() >= 0.0
    assert R.max() <= 7.0


def test_make_lattice_unique_points():
    """Test that all coordinate points are unique."""
    d = 64
    R = make_lattice(d)
    
    unique_coords = set(map(tuple, R))
    assert len(unique_coords) == d


def test_make_lattice_not_perfect_square():
    """Test that non-perfect-square d raises ValueError."""
    with pytest.raises(ValueError, match="not a perfect square"):
        make_lattice(15)
    
    with pytest.raises(ValueError, match="not a perfect square"):
        make_lattice(10)


def test_make_lattice_deterministic():
    """Test that function is deterministic."""
    d = 25
    R1 = make_lattice(d)
    R2 = make_lattice(d)
    assert np.array_equal(R1, R2)


# ===== Tests for build_spatial_cov =====

def test_build_spatial_cov_shape():
    """Test that build_spatial_cov returns correct shape."""
    R = make_lattice(16)
    Sigma = build_spatial_cov(R, alpha0=1.0, alpha1=0.01, beta=8.0)
    assert Sigma.shape == (16, 16)


def test_build_spatial_cov_symmetric():
    """Test that covariance matrix is symmetric."""
    R = make_lattice(25)
    Sigma = build_spatial_cov(R, alpha0=2.0, alpha1=0.05, beta=10.0)
    assert np.allclose(Sigma, Sigma.T)


def test_build_spatial_cov_positive_definite():
    """Test that covariance matrix is positive definite."""
    R = make_lattice(16)
    Sigma = build_spatial_cov(R, alpha0=1.5, alpha1=0.02, beta=8.0)
    
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals > 0)


def test_build_spatial_cov_diagonal():
    """Test that diagonal entries have the expected form."""
    R = make_lattice(9)
    alpha0 = 2.0
    alpha1 = 0.1
    beta = 5.0
    
    Sigma = build_spatial_cov(R, alpha0, alpha1, beta)
    
    # Diagonal should be alpha0 + alpha1
    expected_diag = alpha0 + alpha1
    assert np.allclose(np.diag(Sigma), expected_diag)


def test_build_spatial_cov_decay_with_distance():
    """Test that covariance decays with distance."""
    R = make_lattice(16)
    alpha0 = 2.0
    alpha1 = 0.01
    beta = 8.0
    
    Sigma = build_spatial_cov(R, alpha0, alpha1, beta)
    
    # Points close together should have higher covariance
    # Point (0,0) is at index 0, point (1,0) is at index 1
    # Point (3,3) is at far corner
    
    cov_near = Sigma[0, 1]  # Adjacent points
    cov_far = Sigma[0, 15]  # Far apart
    
    assert cov_near > cov_far


def test_build_spatial_cov_beta_effect():
    """Test that beta (length-scale) affects spatial correlation."""
    R = make_lattice(16)
    alpha0 = 2.0
    alpha1 = 0.01
    
    # Large beta => stronger correlation
    Sigma_large = build_spatial_cov(R, alpha0, alpha1, beta=50.0)
    
    # Small beta => weaker correlation
    Sigma_small = build_spatial_cov(R, alpha0, alpha1, beta=2.0)
    
    # Off-diagonal elements should be larger with large beta
    off_diag_large = np.mean(np.abs(Sigma_large - np.diag(np.diag(Sigma_large))))
    off_diag_small = np.mean(np.abs(Sigma_small - np.diag(np.diag(Sigma_small))))
    
    assert off_diag_large > off_diag_small


# ===== Tests for cholesky_psd =====

def test_cholesky_psd_standard():
    """Test Cholesky on a standard PD matrix."""
    Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    L = cholesky_psd(Sigma)
    
    # L @ L.T should equal Sigma
    assert np.allclose(L @ L.T, Sigma, atol=1e-8)
    
    # L should be lower triangular
    assert np.allclose(np.triu(L, k=1), 0.0)


def test_cholesky_psd_with_jitter():
    """Test that jitter helps with nearly-singular matrices."""
    # Create a nearly singular matrix
    Sigma = np.array([[1.0, 0.99999], [0.99999, 1.0]])
    
    # Should succeed with jitter
    L = cholesky_psd(Sigma, jitter=1e-6)
    assert L.shape == (2, 2)


def test_cholesky_psd_large_matrix():
    """Test Cholesky on a large spatial covariance."""
    R = make_lattice(64)
    Sigma = build_spatial_cov(R, alpha0=2.0, alpha1=0.01, beta=10.0)
    
    L = cholesky_psd(Sigma, jitter=1e-8)
    
    assert L.shape == (64, 64)
    assert np.allclose(L @ L.T, Sigma, atol=1e-6)


def test_cholesky_psd_lower_triangular():
    """Test that result is lower triangular."""
    Sigma = np.eye(5) + 0.5 * np.ones((5, 5))
    L = cholesky_psd(Sigma)
    
    # Upper triangle (excluding diagonal) should be zero
    assert np.allclose(np.triu(L, k=1), 0.0)


# ===== Tests for sample_inverse_gamma =====

def test_sample_inverse_gamma_basic():
    """Test that inverse-gamma sampling returns positive values."""
    rng = np.random.default_rng(42)
    
    for _ in range(100):
        w = sample_inverse_gamma(shape=4.0, scale=4.0, rng=rng)
        assert w > 0


def test_sample_inverse_gamma_mean():
    """Test that sample mean is close to theoretical mean for InvGamma(a,b)."""
    # For InvGamma(a, b), mean = b / (a - 1) when a > 1
    shape = 5.0
    scale = 4.0
    theoretical_mean = scale / (shape - 1)  # 4 / 4 = 1.0
    
    rng = np.random.default_rng(123)
    samples = [sample_inverse_gamma(shape, scale, rng) for _ in range(5000)]
    
    empirical_mean = np.mean(samples)
    
    # Allow 10% tolerance
    assert abs(empirical_mean - theoretical_mean) / theoretical_mean < 0.1


def test_sample_inverse_gamma_variance():
    """Test that sample variance is close to theoretical variance."""
    # For InvGamma(a, b), variance = b^2 / ((a-1)^2 * (a-2)) when a > 2
    shape = 6.0
    scale = 4.0
    theoretical_var = (scale ** 2) / (((shape - 1) ** 2) * (shape - 2))
    
    rng = np.random.default_rng(999)
    samples = [sample_inverse_gamma(shape, scale, rng) for _ in range(10000)]
    
    empirical_var = np.var(samples)
    
    # Allow 15% tolerance due to sampling variability
    assert abs(empirical_var - theoretical_var) / theoretical_var < 0.15


def test_sample_inverse_gamma_different_seeds():
    """Test that different seeds produce different values."""
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    
    w1 = sample_inverse_gamma(4.0, 4.0, rng1)
    w2 = sample_inverse_gamma(4.0, 4.0, rng2)
    
    assert w1 != w2


# ===== Tests for prepare_gamma_vector =====

def test_prepare_gamma_vector_with_provided():
    """Test that provided gamma_vec is returned correctly."""
    d = 16
    rng = np.random.default_rng(42)
    
    gamma_input = np.ones(d) * 0.5
    gamma_output = prepare_gamma_vector(d, gamma_scale=0.1, gamma_vec=gamma_input, rng=rng)
    
    assert np.allclose(gamma_output, gamma_input)


def test_prepare_gamma_vector_random_generation():
    """Test random gamma vector generation."""
    d = 16
    gamma_scale = 0.2
    rng = np.random.default_rng(42)
    
    gamma = prepare_gamma_vector(d, gamma_scale, gamma_vec=None, rng=rng)
    
    assert gamma.shape == (d,)
    
    # Should be approximately unit norm scaled by gamma_scale
    norm = np.linalg.norm(gamma)
    assert abs(norm - gamma_scale) < 1e-10


def test_prepare_gamma_vector_reproducibility():
    """Test that same seed produces same gamma vector."""
    d = 16
    gamma_scale = 0.3
    
    rng1 = np.random.default_rng(123)
    gamma1 = prepare_gamma_vector(d, gamma_scale, gamma_vec=None, rng=rng1)
    
    rng2 = np.random.default_rng(123)
    gamma2 = prepare_gamma_vector(d, gamma_scale, gamma_vec=None, rng=rng2)
    
    assert np.allclose(gamma1, gamma2)


def test_prepare_gamma_vector_wrong_shape():
    """Test that wrong shape gamma_vec raises ValueError."""
    d = 16
    rng = np.random.default_rng(42)
    
    wrong_gamma = np.ones(10)  # Wrong size
    
    with pytest.raises(ValueError, match="incompatible with d="):
        prepare_gamma_vector(d, gamma_scale=0.1, gamma_vec=wrong_gamma, rng=rng)


def test_prepare_gamma_vector_zero_scale():
    """Test that zero scale produces zero vector."""
    d = 16
    rng = np.random.default_rng(42)
    
    gamma = prepare_gamma_vector(d, gamma_scale=0.0, gamma_vec=None, rng=rng)
    
    assert np.allclose(gamma, 0.0)


def test_prepare_gamma_vector_different_seeds():
    """Test that different seeds produce different vectors."""
    d = 16
    gamma_scale = 0.2
    
    rng1 = np.random.default_rng(1)
    gamma1 = prepare_gamma_vector(d, gamma_scale, gamma_vec=None, rng=rng1)
    
    rng2 = np.random.default_rng(2)
    gamma2 = prepare_gamma_vector(d, gamma_scale, gamma_vec=None, rng=rng2)
    
    assert not np.allclose(gamma1, gamma2)
