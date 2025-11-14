import os
import numpy as np
from simulator.simulator_sto_volatility_model import simulate_sv_1d


def test_save_and_roundtrip(tmp_path):
    """Test saving and loading SV1DResults."""
    n = 250
    alpha = 0.9
    sigma = 0.2
    beta = 1.0
    
    res = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=beta, seed=77)
    
    out = tmp_path / "sv_roundtrip.npz"
    res.save(str(out))
    
    assert out.exists()
    
    # Load and verify
    data = np.load(out)
    np.testing.assert_array_equal(data["X"], res.X)
    np.testing.assert_array_equal(data["Y"], res.Y)
    assert float(data["alpha"]) == alpha
    assert float(data["sigma"]) == sigma
    assert float(data["beta"]) == beta
    assert int(data["n"]) == n
    assert data["seed"] == 77


def test_sigma_zero_constant_state():
    """Test that sigma=0 produces constant latent state."""
    n = 100
    alpha = 0.9
    sigma = 0.0
    beta = 1.0
    x0 = 0.5
    
    res = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=beta, seed=10, x0=x0)
    
    # With sigma=0, X_t = alpha^t * x0
    expected_X = x0 * (alpha ** np.arange(n))
    np.testing.assert_allclose(res.X, expected_X, rtol=1e-10, atol=1e-10)


def test_beta_zero_constant_observations():
    """Test that beta=0 produces zero observations."""
    n = 100
    alpha = 0.9
    sigma = 0.2
    beta = 0.0
    
    res = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=beta, seed=10)
    
    # With beta=0, Y_t = 0 for all t
    np.testing.assert_array_equal(res.Y, np.zeros(n))


def test_extreme_alpha_stability():
    """Test that alpha very close to 1 doesn't cause numerical issues."""
    n = 500
    alpha = 0.999
    sigma = 0.1
    beta = 1.0
    
    # Should not raise and should produce finite results
    res = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=beta, seed=20)
    
    assert np.all(np.isfinite(res.X))
    assert np.all(np.isfinite(res.Y))
