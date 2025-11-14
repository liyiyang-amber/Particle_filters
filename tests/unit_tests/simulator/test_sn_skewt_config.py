import numpy as np
import pytest
from simulator.simulator_sensor_network_skewt_dynamic import (
    GridConfig,
    DynConfig,
    MeasConfig,
    SimConfig,
)


# ===== Tests for GridConfig =====

def test_grid_config_defaults():
    """Test GridConfig default values."""
    cfg = GridConfig()
    
    assert cfg.d == 144
    assert cfg.alpha0 == 1.0
    assert cfg.alpha1 == 1e-3
    assert cfg.beta == 8.0


def test_grid_config_custom():
    """Test GridConfig with custom values."""
    cfg = GridConfig(d=64, alpha0=2.5, alpha1=0.05, beta=15.0)
    
    assert cfg.d == 64
    assert cfg.alpha0 == 2.5
    assert cfg.alpha1 == 0.05
    assert cfg.beta == 15.0


def test_grid_config_perfect_square():
    """Test GridConfig with valid perfect square d."""
    for d in [1, 4, 9, 16, 25, 36, 49, 64, 100, 144, 196, 256, 400]:
        cfg = GridConfig(d=d)
        assert cfg.d == d


def test_grid_config_small():
    """Test GridConfig with smallest valid grid."""
    cfg = GridConfig(d=1)
    assert cfg.d == 1


def test_grid_config_large():
    """Test GridConfig with large grid."""
    cfg = GridConfig(d=10000)  # 100x100
    assert cfg.d == 10000


# ===== Tests for DynConfig =====

def test_dyn_config_defaults():
    """Test DynConfig default values."""
    cfg = DynConfig()
    
    assert cfg.alpha == 0.9
    assert cfg.nu == 8.0
    assert cfg.gamma_scale == 0.1
    assert cfg.gamma_vec is None
    assert cfg.clip_x == (-10.0, 10.0)
    assert cfg.chol_jitter == 1e-8
    assert cfg.seed == 123


def test_dyn_config_custom():
    """Test DynConfig with custom values."""
    cfg = DynConfig(
        alpha=0.95,
        nu=6.0,
        gamma_scale=0.2,
        clip_x=(-5.0, 5.0),
        chol_jitter=1e-6,
        seed=999,
    )
    
    assert cfg.alpha == 0.95
    assert cfg.nu == 6.0
    assert cfg.gamma_scale == 0.2
    assert cfg.clip_x == (-5.0, 5.0)
    assert cfg.chol_jitter == 1e-6
    assert cfg.seed == 999


def test_dyn_config_with_gamma_vec():
    """Test DynConfig with explicit gamma vector."""
    gamma = np.array([0.1, 0.2, 0.3])
    cfg = DynConfig(gamma_vec=gamma)
    
    assert np.array_equal(cfg.gamma_vec, gamma)


def test_dyn_config_no_clip():
    """Test DynConfig with no state clipping."""
    cfg = DynConfig(clip_x=None)
    
    assert cfg.clip_x is None


def test_dyn_config_no_seed():
    """Test DynConfig with no seed (random)."""
    cfg = DynConfig(seed=None)
    
    assert cfg.seed is None


def test_dyn_config_nu_range():
    """Test DynConfig with various nu values (should be > 2)."""
    for nu in [2.1, 3.0, 5.0, 10.0, 20.0]:
        cfg = DynConfig(nu=nu)
        assert cfg.nu == nu


def test_dyn_config_alpha_range():
    """Test DynConfig with various alpha values."""
    for alpha in [0.5, 0.7, 0.9, 0.95, 0.99]:
        cfg = DynConfig(alpha=alpha)
        assert cfg.alpha == alpha


# ===== Tests for MeasConfig =====

def test_meas_config_defaults():
    """Test MeasConfig default values."""
    cfg = MeasConfig()
    
    assert cfg.m1 == 1.0
    assert cfg.m2 == 1.0 / 3.0


def test_meas_config_custom():
    """Test MeasConfig with custom values."""
    cfg = MeasConfig(m1=2.0, m2=0.5)
    
    assert cfg.m1 == 2.0
    assert cfg.m2 == 0.5


def test_meas_config_zero_m1():
    """Test MeasConfig with zero intensity scale."""
    cfg = MeasConfig(m1=0.0)
    
    assert cfg.m1 == 0.0


def test_meas_config_negative_m2():
    """Test MeasConfig with negative exponential sensitivity."""
    cfg = MeasConfig(m2=-0.5)
    
    assert cfg.m2 == -0.5


def test_meas_config_various_values():
    """Test MeasConfig with various parameter combinations."""
    for m1 in [0.5, 1.0, 2.0, 5.0]:
        for m2 in [0.1, 0.33, 0.5, 1.0]:
            cfg = MeasConfig(m1=m1, m2=m2)
            assert cfg.m1 == m1
            assert cfg.m2 == m2


# ===== Tests for SimConfig =====

def test_sim_config_defaults():
    """Test SimConfig default values."""
    cfg = SimConfig()
    
    assert cfg.T == 10
    assert cfg.n_trials == 1
    assert cfg.save_lambda is True


def test_sim_config_custom():
    """Test SimConfig with custom values."""
    cfg = SimConfig(T=50, n_trials=10, save_lambda=False)
    
    assert cfg.T == 50
    assert cfg.n_trials == 10
    assert cfg.save_lambda is False


def test_sim_config_single_timestep():
    """Test SimConfig with single time step."""
    cfg = SimConfig(T=1)
    
    assert cfg.T == 1


def test_sim_config_many_trials():
    """Test SimConfig with many trials."""
    cfg = SimConfig(n_trials=1000)
    
    assert cfg.n_trials == 1000


def test_sim_config_save_lambda_options():
    """Test SimConfig with both save_lambda options."""
    cfg_save = SimConfig(save_lambda=True)
    cfg_no_save = SimConfig(save_lambda=False)
    
    assert cfg_save.save_lambda is True
    assert cfg_no_save.save_lambda is False


def test_sim_config_large_T():
    """Test SimConfig with large number of time steps."""
    cfg = SimConfig(T=1000)
    
    assert cfg.T == 1000


# ===== Integration tests for config combinations =====

def test_configs_together():
    """Test that all configs can be instantiated together."""
    grid_cfg = GridConfig(d=16, alpha0=2.0, alpha1=0.05, beta=10.0)
    dyn_cfg = DynConfig(alpha=0.85, nu=6.0, gamma_scale=0.15, seed=42)
    meas_cfg = MeasConfig(m1=1.5, m2=0.4)
    sim_cfg = SimConfig(T=30, n_trials=5, save_lambda=True)
    
    assert grid_cfg.d == 16
    assert dyn_cfg.alpha == 0.85
    assert meas_cfg.m1 == 1.5
    assert sim_cfg.T == 30


def test_config_immutability():
    """Test that configs are dataclasses can be modified after creation."""
    cfg = GridConfig(d=16)
    
    # Dataclasses are mutable by default unless frozen=True
    cfg.d = 25
    assert cfg.d == 25


def test_config_from_dict():
    """Test creating configs from dictionaries."""
    grid_params = {"d": 64, "alpha0": 3.0, "alpha1": 0.01, "beta": 20.0}
    cfg = GridConfig(**grid_params)
    
    assert cfg.d == 64
    assert cfg.alpha0 == 3.0


def test_config_to_dict():
    """Test converting configs to dictionaries."""
    cfg = DynConfig(alpha=0.92, nu=7.0, seed=555)
    cfg_dict = vars(cfg)
    
    assert cfg_dict["alpha"] == 0.92
    assert cfg_dict["nu"] == 7.0
    assert cfg_dict["seed"] == 555
