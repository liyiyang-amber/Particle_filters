import numpy as np
import pytest
from simulator.simulator_sto_volatility_model import simulate_sv_1d, SV1DResults


def test_shapes_and_params():
    """Test basic shapes and parameter storage for 1D SV model."""
    n = 1000
    alpha = 0.9
    sigma = 0.2
    beta = 1.0
    res = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=beta, seed=1)
    
    assert isinstance(res, SV1DResults)
    assert res.X.shape == (n,)
    assert res.Y.shape == (n,)
    assert res.alpha == alpha
    assert res.sigma == sigma
    assert res.beta == beta
    assert res.n == n
    assert res.seed == 1


def test_validation_errors():
    """Test that invalid inputs raise appropriate errors."""
    # n must be positive
    with pytest.raises(ValueError, match="n must be positive"):
        simulate_sv_1d(n=0, alpha=0.9, sigma=0.2, beta=1.0)
    
    with pytest.raises(ValueError, match="n must be positive"):
        simulate_sv_1d(n=-5, alpha=0.9, sigma=0.2, beta=1.0)
    
    # |alpha| must be < 1
    with pytest.raises(ValueError, match="alpha.*< 1"):
        simulate_sv_1d(n=10, alpha=1.0, sigma=0.2, beta=1.0)
    
    with pytest.raises(ValueError, match="alpha.*< 1"):
        simulate_sv_1d(n=10, alpha=-1.5, sigma=0.2, beta=1.0)
    
    # sigma must be non-negative
    with pytest.raises(ValueError, match="sigma.*nonnegative"):
        simulate_sv_1d(n=10, alpha=0.9, sigma=-0.1, beta=1.0)
    
    # beta must be non-negative
    with pytest.raises(ValueError, match="beta.*nonnegative"):
        simulate_sv_1d(n=10, alpha=0.9, sigma=0.2, beta=-1.0)


def test_seed_reproducibility():
    """Test that same seed produces identical results."""
    n = 200
    alpha = 0.9
    sigma = 0.2
    beta = 1.0
    
    r1 = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=beta, seed=42)
    r2 = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=beta, seed=42)
    
    np.testing.assert_array_equal(r1.X, r2.X)
    np.testing.assert_array_equal(r1.Y, r2.Y)


def test_different_seeds_produce_different_streams():
    """Test that different seeds produce different results."""
    n = 200
    alpha = 0.9
    sigma = 0.2
    beta = 1.0
    
    r1 = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=beta, seed=1)
    r2 = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=beta, seed=2)
    
    # High-probability inequality (not strict proof)
    assert not np.allclose(r1.X, r2.X)
    assert not np.allclose(r1.Y, r2.Y)
