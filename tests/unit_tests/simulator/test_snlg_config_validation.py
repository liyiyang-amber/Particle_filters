import pytest
from simulator.simulator_sensor_network_linear_gaussian import SimConfig


def test_valid_default_config():
    """Test that default configuration is valid."""
    cfg = SimConfig()
    assert cfg.d == 64
    assert cfg.alpha == 0.9
    assert cfg.alpha0 == 3.0
    assert cfg.alpha1 == 0.01
    assert cfg.beta == 20.0
    assert cfg.T == 10
    assert cfg.trials == 100
    assert cfg.sigmas == (2.0, 1.0, 0.5)
    assert cfg.seed == 123


def test_valid_custom_config():
    """Test that a custom valid configuration works."""
    cfg = SimConfig(
        d=100,  # 10x10 grid
        alpha=0.95,
        alpha0=2.5,
        alpha1=0.05,
        beta=15.0,
        T=20,
        trials=50,
        sigmas=(1.0, 0.5),
        seed=999,
    )
    assert cfg.d == 100
    assert cfg.T == 20
    assert cfg.trials == 50
    assert len(cfg.sigmas) == 2


def test_invalid_d_not_perfect_square():
    """Test that d must be a perfect square."""
    with pytest.raises(ValueError, match="d must be a perfect square"):
        SimConfig(d=60)  # Not a perfect square
    
    with pytest.raises(ValueError, match="d must be a perfect square"):
        SimConfig(d=50)
    
    with pytest.raises(ValueError, match="d must be a perfect square"):
        SimConfig(d=10)


def test_valid_d_perfect_squares():
    """Test that perfect square values of d are accepted."""
    for d in [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 144, 225]:
        cfg = SimConfig(d=d)
        assert cfg.d == d


def test_invalid_T_zero():
    """Test that T must be positive."""
    with pytest.raises(ValueError, match="T and trials must be positive integers"):
        SimConfig(T=0)


def test_invalid_T_negative():
    """Test that T cannot be negative."""
    with pytest.raises(ValueError, match="T and trials must be positive integers"):
        SimConfig(T=-5)


def test_invalid_trials_zero():
    """Test that trials must be positive."""
    with pytest.raises(ValueError, match="T and trials must be positive integers"):
        SimConfig(trials=0)


def test_invalid_trials_negative():
    """Test that trials cannot be negative."""
    with pytest.raises(ValueError, match="T and trials must be positive integers"):
        SimConfig(trials=-10)


def test_invalid_sigmas_zero():
    """Test that observation noise std deviations must be positive."""
    with pytest.raises(ValueError, match="All observation std deviations must be positive"):
        SimConfig(sigmas=(2.0, 0.0, 1.0))


def test_invalid_sigmas_negative():
    """Test that observation noise std deviations cannot be negative."""
    with pytest.raises(ValueError, match="All observation std deviations must be positive"):
        SimConfig(sigmas=(2.0, -0.5, 1.0))


def test_invalid_sigmas_all_negative():
    """Test that all negative sigmas are rejected."""
    with pytest.raises(ValueError, match="All observation std deviations must be positive"):
        SimConfig(sigmas=(-1.0, -2.0))


def test_valid_single_sigma():
    """Test that a single sigma value is valid."""
    cfg = SimConfig(sigmas=(1.0,))
    assert cfg.sigmas == (1.0,)


def test_invalid_alpha1_negative():
    """Test that alpha1 (nugget) must be non-negative."""
    with pytest.raises(ValueError, match="alpha1 \\(nugget\\) must be nonnegative"):
        SimConfig(alpha1=-0.01)


def test_valid_alpha1_zero():
    """Test that alpha1 can be zero."""
    cfg = SimConfig(alpha1=0.0)
    assert cfg.alpha1 == 0.0


def test_invalid_beta_zero():
    """Test that beta must be positive."""
    with pytest.raises(ValueError, match="beta must be positive"):
        SimConfig(beta=0.0)


def test_invalid_beta_negative():
    """Test that beta cannot be negative."""
    with pytest.raises(ValueError, match="beta must be positive"):
        SimConfig(beta=-10.0)


def test_valid_beta_small():
    """Test that small positive beta values are valid."""
    cfg = SimConfig(beta=0.01)
    assert cfg.beta == 0.01
