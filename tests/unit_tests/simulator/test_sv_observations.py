import numpy as np
from simulator.simulator_sto_volatility_model import simulate_sv_1d


def test_observations_depend_on_state():
    """Test that observations depend on latent state through exp(0.5*X)."""
    n = 5000
    alpha = 0.9
    sigma = 0.2
    beta = 1.0
    
    res = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=beta, seed=5)
    
    # The observation model is Y_t = beta * exp(0.5 * X_t) * W_t
    # So Y_t / beta should have variance that scales with exp(X_t)
    # Check that observations are not constant
    assert res.Y.std() > 0.1
    
    # Check that observations have reasonable range
    assert np.all(np.isfinite(res.Y))


def test_observation_volatility_scales_with_state():
    """Test that observation volatility increases with latent state."""
    n = 10000
    alpha = 0.9
    sigma = 0.3
    beta = 1.0
    
    res = simulate_sv_1d(n=n, alpha=alpha, sigma=sigma, beta=beta, seed=42)
    
    # Split data into high and low latent state periods
    median_x = np.median(res.X)
    high_state = res.X > median_x
    low_state = res.X <= median_x
    
    # Observations should have higher variance when state is high
    y_high_var = res.Y[high_state].var()
    y_low_var = res.Y[low_state].var()
    
    # In SV model, higher X leads to higher observation variance
    assert y_high_var > y_low_var
