import numpy as np
import pytest

# import your simulator from your package/module
# from yourpkg.simulator import simulate_lgssm, LGSSMSimulationResult
from simulator.simulator_LGSSM import simulate_lgssm, LGSSMSimulationResult  

@pytest.fixture(scope="session")
def rng_seed() -> int:
    return 12345

@pytest.fixture(scope="session")
def small_matrices():
    """Stable 2D latent, 2D observation setup."""
    nx, ny, nv, nw = 2, 2, 2, 2
    A = np.array([[0.9, 0.2],
                  [0.0, 0.7]])
    B = np.diag([np.sqrt(0.05), np.sqrt(0.02)])  # (2x2)
    C = np.array([[1.0, 0.0],    
                [0.0, 1.0]])   # (2x2)
    D = np.array([[np.sqrt(0.10), 0.0],
            [0.0, np.sqrt(0.10)]])            # (2x2)
    Sigma = np.eye(nx)
    return dict(nx=nx, ny=ny, nv=nv, nw=nw, A=A, B=B, C=C, D=D, Sigma=Sigma)

@pytest.fixture(scope="session")
def large_N() -> int:
    """Use moderately large N for better statistics without being too slow."""
    return 20000

@pytest.fixture(scope="session")
def small_N() -> int:
    return 8

from models.kalman_filter import kalman_filter_general, KFResults

@pytest.fixture(scope="session")
def small_system():
    nx, ny = 2, 1
    Phi = np.array([[0.9, 0.2],
                    [0.0, 0.7]])
    H   = np.array([[1.0, 0.0]])
    Gamma = np.eye(nx)
    Q = np.diag([0.05, 0.02])
    R = np.array([[0.10]])
    x0 = np.zeros(nx)
    P0 = np.eye(nx)
    return dict(nx=nx, ny=ny, Phi=Phi, H=H, Gamma=Gamma, Q=Q, R=R, x0=x0, P0=P0)

@pytest.fixture(scope="session")
def N_small(): return 12

@pytest.fixture(scope="session")
def rng(): return np.random.default_rng(123)

@pytest.fixture(scope="session")
def Y_synthetic(small_system, N_small, rng):
    # quick synthetic obs (not necessarily from LGSSM) just for API tests
    return rng.normal(size=(N_small, small_system["ny"]))

@pytest.fixture(scope="session")
def sv_1d_params():
    """Standard parameters for 1D stochastic volatility model tests."""
    return dict(
        alpha=0.9,   # AR(1) coefficient
        sigma=0.2,   # Process noise std
        beta=1.0,    # Observation scale
        n=500,       # Number of time steps
        seed=42      # Random seed
    )

# Multi-Target Acoustic Tracking fixtures
from simulator.simulator_Multi_acoustic_tracking import (
    ScenarioConfig,
    DynamicsConfig,
    simulate_acoustic_dataset,
)

@pytest.fixture(scope="session")
def mat_standard_config():
    """Standard configuration for multi-target acoustic tracking tests."""
    return ScenarioConfig(
        n_targets=4,
        n_steps=100,
        area_xy=(40.0, 40.0),
        sensor_grid_shape=(5, 5),
        psi=10.0,
        d0=0.1,
        meas_noise_std=0.1,
        seed=42,
        use_article_init=True,
    )

@pytest.fixture(scope="session")
def mat_dynamics_config():
    """Standard dynamics configuration for MAT tests."""
    return DynamicsConfig(dt=1.0)

@pytest.fixture(scope="session")
def mat_small_config():
    """Small configuration for fast MAT tests."""
    return ScenarioConfig(
        n_targets=2,
        n_steps=20,
        area_xy=(40.0, 40.0),
        sensor_grid_shape=(3, 3),
        psi=10.0,
        d0=0.1,
        meas_noise_std=0.1,
        seed=123,
        use_article_init=False,
    )

@pytest.fixture(scope="session")
def mat_standard_data(mat_standard_config, mat_dynamics_config):
    """Pre-generated standard MAT dataset for reuse in tests."""
    return simulate_acoustic_dataset(mat_standard_config, mat_dynamics_config)

@pytest.fixture(scope="session")
def mat_filter_params():
    """Filter parameters for MAT tracking (from article specification)."""
    Q_filter = np.array([
        [3.0, 0.0, 0.1, 0.0],
        [0.0, 3.0, 0.0, 0.1],
        [0.1, 0.0, 0.03, 0.0],
        [0.0, 0.1, 0.0, 0.03],
    ])
    return {
        'Q': Q_filter,
        'n_particles': 500,
        'resample_ess_ratio': 0.5,
        'n_lambda_steps': 8,
        'init_pos_std': 10.0,
        'init_vel_std': 1.0,
    }

# Sensor Network Linear Gaussian (SNLG) fixtures
from simulator.simulator_sensor_network_linear_gaussian import (
    SimConfig,
    simulate_dataset,
)

@pytest.fixture(scope="session")
def snlg_small_config():
    """Small configuration for fast SNLG tests."""
    return SimConfig(
        d=9,  # 3x3 grid
        alpha=0.9,
        alpha0=2.0,
        alpha1=0.05,
        beta=10.0,
        T=10,
        trials=3,
        sigmas=(1.0,),
        seed=42,
    )

@pytest.fixture(scope="session")
def snlg_standard_config():
    """Standard configuration for SNLG tests."""
    return SimConfig(
        d=16,  # 4x4 grid
        alpha=0.9,
        alpha0=2.5,
        alpha1=0.01,
        beta=15.0,
        T=50,
        trials=10,
        sigmas=(2.0, 1.0, 0.5),
        seed=123,
    )

@pytest.fixture(scope="session")
def snlg_large_config():
    """Large configuration for comprehensive SNLG tests."""
    return SimConfig(
        d=64,  # 8x8 grid
        alpha=0.95,
        alpha0=3.0,
        alpha1=0.01,
        beta=20.0,
        T=100,
        trials=50,
        sigmas=(2.0, 1.0),
        seed=999,
    )

@pytest.fixture(scope="session")
def snlg_small_data(snlg_small_config):
    """Pre-generated small SNLG dataset for reuse in tests."""
    X, Z, coords, Sigma = simulate_dataset(snlg_small_config)
    return {
        'X': X,
        'Z': Z,
        'coords': coords,
        'Sigma': Sigma,
        'config': snlg_small_config,
    }

@pytest.fixture(scope="session")
def snlg_standard_data(snlg_standard_config):
    """Pre-generated standard SNLG dataset for reuse in tests."""
    X, Z, coords, Sigma = simulate_dataset(snlg_standard_config)
    return {
        'X': X,
        'Z': Z,
        'coords': coords,
        'Sigma': Sigma,
        'config': snlg_standard_config,
    }

# Sensor Network Skew-t Dynamic (SNLG Skew-t) fixtures
from simulator.simulator_sensor_network_skewt_dynamic import (
    GridConfig,
    DynConfig,
    MeasConfig,
    SimConfig as SkewTSimConfig,
    simulate_trial,
    simulate_many,
)

@pytest.fixture(scope="session")
def skewt_small_grid_config():
    """Small grid configuration for fast skew-t tests."""
    return GridConfig(
        d=9,  # 3x3 grid
        alpha0=2.0,
        alpha1=0.05,
        beta=8.0,
    )

@pytest.fixture(scope="session")
def skewt_standard_grid_config():
    """Standard grid configuration for skew-t tests."""
    return GridConfig(
        d=16,  # 4x4 grid
        alpha0=2.5,
        alpha1=0.01,
        beta=10.0,
    )

@pytest.fixture(scope="session")
def skewt_large_grid_config():
    """Large grid configuration for comprehensive skew-t tests."""
    return GridConfig(
        d=64,  # 8x8 grid
        alpha0=3.0,
        alpha1=0.01,
        beta=15.0,
    )

@pytest.fixture(scope="session")
def skewt_light_tail_dyn_config():
    """Dynamics configuration with light tails (high nu)."""
    return DynConfig(
        alpha=0.9,
        nu=20.0,  # Light tails
        gamma_scale=0.1,
        clip_x=(-10.0, 10.0),
        chol_jitter=1e-8,
        seed=42,
    )

@pytest.fixture(scope="session")
def skewt_heavy_tail_dyn_config():
    """Dynamics configuration with heavy tails (low nu)."""
    return DynConfig(
        alpha=0.85,
        nu=3.5,  # Heavy tails
        gamma_scale=0.2,
        clip_x=(-10.0, 10.0),
        chol_jitter=1e-8,
        seed=42,
    )

@pytest.fixture(scope="session")
def skewt_standard_dyn_config():
    """Standard dynamics configuration for skew-t tests."""
    return DynConfig(
        alpha=0.9,
        nu=8.0,
        gamma_scale=0.15,
        clip_x=(-10.0, 10.0),
        chol_jitter=1e-8,
        seed=42,
    )

@pytest.fixture(scope="session")
def skewt_standard_meas_config():
    """Standard measurement configuration for skew-t tests."""
    return MeasConfig(
        m1=1.0,
        m2=1.0 / 3.0,
    )

@pytest.fixture(scope="session")
def skewt_small_sim_config():
    """Small simulation configuration for fast skew-t tests."""
    return SkewTSimConfig(
        T=10,
        n_trials=3,
        save_lambda=True,
    )

@pytest.fixture(scope="session")
def skewt_standard_sim_config():
    """Standard simulation configuration for skew-t tests."""
    return SkewTSimConfig(
        T=50,
        n_trials=10,
        save_lambda=True,
    )

@pytest.fixture(scope="session")
def skewt_small_trial_data(skewt_small_grid_config, skewt_standard_dyn_config, 
                           skewt_standard_meas_config, skewt_small_sim_config):
    """Pre-generated small single-trial skew-t dataset for reuse in tests."""
    data = simulate_trial(
        skewt_small_grid_config,
        skewt_standard_dyn_config,
        skewt_standard_meas_config,
        skewt_small_sim_config,
    )
    return data

@pytest.fixture(scope="session")
def skewt_small_multi_trial_data(skewt_small_grid_config, skewt_standard_dyn_config,
                                  skewt_standard_meas_config, skewt_small_sim_config):
    """Pre-generated small multi-trial skew-t dataset for reuse in tests."""
    data = simulate_many(
        skewt_small_grid_config,
        skewt_standard_dyn_config,
        skewt_standard_meas_config,
        skewt_small_sim_config,
    )
    return data

@pytest.fixture(scope="session")
def skewt_standard_trial_data(skewt_standard_grid_config, skewt_standard_dyn_config,
                               skewt_standard_meas_config, skewt_standard_sim_config):
    """Pre-generated standard single-trial skew-t dataset for reuse in tests."""
    data = simulate_trial(
        skewt_standard_grid_config,
        skewt_standard_dyn_config,
        skewt_standard_meas_config,
        skewt_standard_sim_config,
    )
    return data

@pytest.fixture(scope="session")
def skewt_heavy_tail_data(skewt_small_grid_config, skewt_heavy_tail_dyn_config,
                           skewt_standard_meas_config):
    """Pre-generated heavy-tailed skew-t dataset for robustness tests."""
    sim_cfg = SkewTSimConfig(T=40, n_trials=5, save_lambda=True)
    data = simulate_many(
        skewt_small_grid_config,
        skewt_heavy_tail_dyn_config,
        skewt_standard_meas_config,
        sim_cfg,
    )
    return data

