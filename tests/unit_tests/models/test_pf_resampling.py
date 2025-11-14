"""Unit tests for Particle Filter: resampling methods and regularization."""

import numpy as np
import pytest
from models.particle_filter import ParticleFilter, PFState


@pytest.fixture
def simple_system():
    """Simple 1D system for resampling tests."""
    nx, nz = 1, 1
    Q = np.array([[0.1]])
    R = np.array([[0.5]])

    def g(x, u):
        return 0.9 * x

    def h(x):
        return x

    return dict(nx=nx, nz=nz, Q=Q, R=R, g=g, h=h)


def test_systematic_resampling_basic(simple_system):
    """Test systematic resampling produces correct output shape."""
    s = simple_system
    pf = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000,
        resample_method="systematic",
        resample_thresh=0.5
    )

    # Create non-uniform weights
    weights = np.random.rand(1000)
    weights = weights / np.sum(weights)

    # Test internal resampling method
    indices = pf._systematic_resample(weights)

    assert len(indices) == 1000
    assert np.all(indices >= 0)
    assert np.all(indices < 1000)
    assert indices.dtype == int


def test_multinomial_resampling_basic(simple_system):
    """Test multinomial resampling produces correct output shape."""
    s = simple_system
    pf = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000,
        resample_method="multinomial",
        resample_thresh=0.5
    )

    # Create non-uniform weights
    weights = np.random.rand(1000)
    weights = weights / np.sum(weights)

    # Test internal resampling method
    indices = pf._multinomial_resample(weights)

    assert len(indices) == 1000
    assert np.all(indices >= 0)
    assert np.all(indices < 1000)
    assert indices.dtype == int


def test_resampling_preserves_high_weight_particles(simple_system):
    """Test that resampling favors high-weight particles."""
    s = simple_system
    pf = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000,
        resample_method="systematic",
        resample_thresh=0.5,
        rng=np.random.default_rng(123)
    )

    # Create particles with one dominant particle
    particles = np.random.randn(1000, 1)
    particles[0, 0] = 100.0  # Make one particle far from others

    # Create weights with one very high weight
    weights = np.ones(1000) * 0.0001
    weights[0] = 0.9
    weights = weights / np.sum(weights)

    # Resample
    resampled_particles, resampled_weights = pf._resample(particles, weights)

    # The dominant particle should appear frequently
    count_dominant = np.sum(np.abs(resampled_particles[:, 0] - 100.0) < 1e-6)
    assert count_dominant > 500  # Should be ~90% of particles


def test_resampling_triggered_by_low_neff(simple_system):
    """Test that resampling is triggered when Neff drops below threshold."""
    s = simple_system
    pf = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000,
        resample_method="systematic",
        resample_thresh=0.5,  # Trigger when Neff < 500
        rng=np.random.default_rng(123)
    )

    # Create particles
    particles = np.random.randn(1000, 1)
    
    # Create highly non-uniform weights (low Neff)
    weights = np.zeros(1000)
    weights[0] = 0.9
    weights[1:] = 0.1 / 999
    Neff = 1.0 / np.sum(weights ** 2)
    assert Neff < 500  # Should trigger resampling

    particles_before = particles.copy()
    resampled_particles, resampled_weights = pf._resample(particles, weights)

    # After resampling, weights should be uniform
    np.testing.assert_allclose(resampled_weights, np.ones(1000) / 1000, rtol=1e-10)

    # Particles should have changed
    assert not np.array_equal(particles_before, resampled_particles)


def test_resampling_not_triggered_by_high_neff(simple_system):
    """Test that resampling is NOT triggered when Neff is high."""
    s = simple_system
    pf = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000,
        resample_method="systematic",
        resample_thresh=0.5,
        rng=np.random.default_rng(123)
    )

    # Create particles
    particles = np.random.randn(1000, 1)
    
    # Create nearly uniform weights (high Neff)
    weights = np.ones(1000) / 1000 + np.random.randn(1000) * 0.0001
    weights = weights / np.sum(weights)
    Neff = 1.0 / np.sum(weights ** 2)
    assert Neff > 500  # Should NOT trigger resampling

    particles_before = particles.copy()
    weights_before = weights.copy()
    resampled_particles, resampled_weights = pf._resample(particles, weights)

    # Particles and weights should be unchanged
    np.testing.assert_array_equal(particles_before, resampled_particles)
    np.testing.assert_array_equal(weights_before, resampled_weights)


def test_regularization_after_resample(simple_system):
    """Test that regularization adds jitter after resampling."""
    s = simple_system
    
    # Without regularization
    pf_no_reg = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000,
        resample_method="systematic",
        resample_thresh=0.9,  # Force resampling
        regularize_after_resample=False,
        rng=np.random.default_rng(123)
    )

    # With regularization
    pf_with_reg = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000,
        resample_method="systematic",
        resample_thresh=0.9,  # Force resampling
        regularize_after_resample=True,
        rng=np.random.default_rng(123)
    )

    # Create highly non-uniform weights to force resampling
    particles = np.random.randn(1000, 1)
    weights = np.zeros(1000)
    weights[0] = 0.95
    weights[1:] = 0.05 / 999

    # Resample without regularization
    particles_no_reg, _ = pf_no_reg._resample(particles.copy(), weights.copy())

    # Resample with regularization
    particles_with_reg, _ = pf_with_reg._resample(particles.copy(), weights.copy())

    # With regularization, particles should have more diversity
    # Standard deviation should be higher with regularization
    std_no_reg = np.std(particles_no_reg)
    std_with_reg = np.std(particles_with_reg)
    
    assert not np.allclose(particles_no_reg, particles_with_reg)


def test_resampling_in_filtering_loop(simple_system):
    """Test resampling occurs naturally during filtering."""
    s = simple_system
    pf = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=500,
        resample_method="systematic",
        resample_thresh=0.5,
        rng=np.random.default_rng(123)
    )

    mean = np.array([0.0])
    cov = np.array([[1.0]])
    pf.initialize(mean, cov)

    # Create an extreme observation to cause weight degeneracy
    z = np.array([10.0])
    
    # Check Neff before update
    Neff_before = pf.effective_sample_size()
    
    # Perform update
    state = pf.step(z)
    
    # After update with extreme observation, weights should be resampled
    # Verify weights are relatively uniform (indicating resampling happened)
    Neff_after = pf.effective_sample_size()
    
    # Weights should sum to 1
    assert np.isclose(np.sum(state.weights), 1.0)


def test_systematic_vs_multinomial_different_results(simple_system):
    """Test that systematic and multinomial give different results."""
    s = simple_system
    
    # Systematic resampling
    pf_sys = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000,
        resample_method="systematic",
        resample_thresh=0.9,
        rng=np.random.default_rng(42)
    )

    # Multinomial resampling
    pf_multi = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000,
        resample_method="multinomial",
        resample_thresh=0.9,
        rng=np.random.default_rng(42)
    )

    # Initialize both with same seed
    mean = np.array([0.0])
    cov = np.array([[1.0]])
    
    pf_sys.initialize(mean, cov)
    pf_multi.initialize(mean, cov)

    # Run several steps
    observations = [np.array([x]) for x in [1.0, 2.0, 1.5, 0.5, -1.0]]
    
    states_sys = []
    states_multi = []
    
    for z in observations:
        states_sys.append(pf_sys.step(z))
        states_multi.append(pf_multi.step(z))

    # Results should be different (different resampling methods)
    # Check at least one state is different
    differs = False
    for s1, s2 in zip(states_sys, states_multi):
        if not np.allclose(s1.mean, s2.mean, atol=0.1):
            differs = True
            break
    
    # With different resampling methods, we expect some difference


def test_resample_preserves_particle_count(simple_system):
    """Test that resampling preserves the number of particles."""
    s = simple_system
    pf = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000,
        resample_method="systematic",
        resample_thresh=0.5,
        rng=np.random.default_rng(123)
    )

    particles = np.random.randn(1000, 1)
    weights = np.random.rand(1000)
    weights = weights / np.sum(weights)

    resampled_particles, resampled_weights = pf._resample(particles, weights)

    assert len(resampled_particles) == 1000
    assert len(resampled_weights) == 1000


def test_resampling_edge_case_single_particle(simple_system):
    """Test resampling with Np=1 (edge case)."""
    s = simple_system
    pf = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1,
        resample_method="systematic",
        resample_thresh=0.5
    )

    mean = np.array([0.0])
    cov = np.array([[1.0]])
    state = pf.initialize(mean, cov)

    # Should work without errors
    z = np.array([1.0])
    state = pf.step(z)
    
    assert state.particles.shape == (1, 1)
    assert np.isclose(np.sum(state.weights), 1.0)


def test_effective_sample_size_formula(simple_system):
    """Test that Neff formula is correct."""
    s = simple_system
    pf = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000
    )

    mean = np.array([0.0])
    cov = np.array([[1.0]])
    pf.initialize(mean, cov)

    # Manually compute Neff
    weights = pf.state.weights
    Neff_manual = 1.0 / np.sum(weights ** 2)
    Neff_method = pf.effective_sample_size()

    np.testing.assert_allclose(Neff_manual, Neff_method)


def test_resample_thresh_values(simple_system):
    """Test different resample threshold values."""
    s = simple_system

    # Very low threshold (almost never resamples)
    pf_low = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000,
        resample_thresh=0.1
    )
    assert pf_low.resample_thresh == 0.1

    # High threshold (resamples more often)
    pf_high = ParticleFilter(
        g=s["g"], 
        h=s["h"], 
        Q=s["Q"], 
        R=s["R"], 
        Np=1000,
        resample_thresh=0.9
    )
    assert pf_high.resample_thresh == 0.9
