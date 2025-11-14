"""Unit tests for Lorenz 96 dynamics and integration."""

import numpy as np
import pytest
from simulator.simulator_Lorenz_96 import (
    l96_rhs,
    rk4_step,
    l96_integrate,
    compute_rmse,
    compute_ensemble_spread,
)


def test_l96_rhs_shape():
    """Test that RHS function returns correct shape."""
    nx = 40
    x = np.random.randn(nx)
    
    dxdt = l96_rhs(x, F=8.0)
    
    assert dxdt.shape == (nx,)


def test_l96_rhs_periodic_boundary():
    """Test that RHS respects periodic boundary conditions."""
    # For periodic system, shifting indices should give related results
    nx = 10
    x = np.random.randn(nx)
    
    dxdt = l96_rhs(x, F=8.0)
    
    # Roll the state and the derivative
    x_rolled = np.roll(x, 1)
    dxdt_rolled = l96_rhs(x_rolled, F=8.0)
    
    # The derivative should also roll
    np.testing.assert_allclose(dxdt_rolled, np.roll(dxdt, 1))


def test_l96_rhs_forcing_term():
    """Test that changing F affects the RHS."""
    nx = 40
    x = np.random.randn(nx)
    
    dxdt_F8 = l96_rhs(x, F=8.0)
    dxdt_F10 = l96_rhs(x, F=10.0)
    
    # Different forcing should give different derivatives
    assert not np.allclose(dxdt_F8, dxdt_F10)
    
    # The difference should be exactly 2.0 everywhere
    np.testing.assert_allclose(dxdt_F10 - dxdt_F8, 2.0)


def test_rk4_step_advances_state():
    """Test that RK4 step advances the state."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dt = 0.01
    
    def simple_dynamics(x):
        return -0.1 * x  # Simple decay
    
    x_new = rk4_step(x, dt, simple_dynamics)
    
    # State should have changed
    assert not np.allclose(x, x_new)
    
    # For decay dynamics, values should decrease
    assert np.all(np.abs(x_new) < np.abs(x))


def test_rk4_step_conserves_dimension():
    """Test that RK4 step preserves dimension."""
    for nx in [10, 40, 100]:
        x = np.random.randn(nx)
        x_new = rk4_step(x, 0.01, lambda z: l96_rhs(z, F=8.0))
        assert x_new.shape == (nx,)


def test_l96_integrate_shape():
    """Test that integration returns correct shape."""
    nx = 40
    x0 = np.random.randn(nx)
    dt = 0.01
    steps = 100
    
    traj = l96_integrate(x0, dt, steps, F=8.0)
    
    assert traj.shape == (steps + 1, nx)
    
    # First entry should be initial condition
    np.testing.assert_array_equal(traj[0], x0)


def test_l96_integrate_deterministic():
    """Test that deterministic integration is reproducible."""
    nx = 40
    x0 = np.random.randn(nx)
    dt = 0.01
    steps = 50
    
    traj1 = l96_integrate(x0, dt, steps, F=8.0, q_std=0.0)
    traj2 = l96_integrate(x0, dt, steps, F=8.0, q_std=0.0)
    
    np.testing.assert_array_equal(traj1, traj2)


def test_l96_integrate_with_noise():
    """Test that integration with noise uses RNG correctly."""
    nx = 40
    x0 = np.random.randn(nx)
    dt = 0.01
    steps = 50
    q_std = 0.1
    
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    
    traj1 = l96_integrate(x0, dt, steps, F=8.0, q_std=q_std, rng=rng1)
    traj2 = l96_integrate(x0, dt, steps, F=8.0, q_std=q_std, rng=rng2)
    
    # Same seed should give same trajectory
    np.testing.assert_allclose(traj1, traj2)
    
    # Different seed should give different trajectory
    rng3 = np.random.default_rng(99)
    traj3 = l96_integrate(x0, dt, steps, F=8.0, q_std=q_std, rng=rng3)
    assert not np.allclose(traj1, traj3)


def test_l96_integrate_zero_noise():
    """Test that q_std=0 gives deterministic integration."""
    nx = 40
    x0 = np.random.randn(nx)
    dt = 0.01
    steps = 50
    
    # Even with different RNG seeds, q_std=0 should be identical
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(999)
    
    traj1 = l96_integrate(x0, dt, steps, F=8.0, q_std=0.0, rng=rng1)
    traj2 = l96_integrate(x0, dt, steps, F=8.0, q_std=0.0, rng=rng2)
    
    np.testing.assert_array_equal(traj1, traj2)


def test_l96_equilibrium_near_F():
    """Test that L96 states tend to hover around forcing value F."""
    nx = 40
    F = 8.0
    x0 = np.full(nx, F)
    x0[::5] = F + 1.0  # Standard initialization
    
    dt = 0.01
    steps = 5000  # Longer spinup for equilibrium
    
    traj = l96_integrate(x0, dt, steps, F=F, q_std=0.0)
    
    # After spinup, mean should be near F 
    mean_last_quarter = traj[3*steps//4:].mean()
    assert 2.0 < mean_last_quarter < 12.0  # Broad range for chaotic system around F=8


def test_compute_rmse():
    """Test RMSE computation."""
    forecast = np.array([1.0, 2.0, 3.0])
    truth = np.array([1.1, 2.2, 2.8])
    
    rmse = compute_rmse(forecast, truth)
    
    expected = np.sqrt(((0.1)**2 + (0.2)**2 + (0.2)**2) / 3)
    assert np.isclose(rmse, expected)


def test_compute_rmse_perfect_match():
    """Test RMSE is zero for perfect match."""
    x = np.random.randn(100)
    rmse = compute_rmse(x, x)
    assert np.isclose(rmse, 0.0)


def test_compute_ensemble_spread():
    """Test ensemble spread computation."""
    # Create ensemble with known spread
    ensemble = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])
    
    spread = compute_ensemble_spread(ensemble, axis=0)
    
    # All members identical -> zero spread
    np.testing.assert_allclose(spread, 0.0)
    
    # Test with variation
    ensemble = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
    ])
    
    spread = compute_ensemble_spread(ensemble, axis=0)
    expected_std = np.std([0.0, 1.0, -1.0])
    np.testing.assert_allclose(spread, [expected_std] * 3)


def test_chaotic_divergence():
    """Test that tiny perturbations lead to divergence."""
    nx = 40
    x0 = np.random.randn(nx)
    
    # Two slightly different initial conditions
    x1 = x0.copy()
    x2 = x0 + 1e-6  # Small perturbation
    
    dt = 0.01
    steps = 2000  # Longer integration for chaos to manifest
    
    traj1 = l96_integrate(x1, dt, steps, F=8.0, q_std=0.0)
    traj2 = l96_integrate(x2, dt, steps, F=8.0, q_std=0.0)
    
    # Initially very close
    initial_diff = np.linalg.norm(traj1[0] - traj2[0])
    assert initial_diff < 1e-4
    
    # Should diverge significantly after some time 
    final_diff = np.linalg.norm(traj1[-1] - traj2[-1])
    assert final_diff > 0.1  # Significant divergence due to chaos


def test_energy_conservation_order():
    """Test that total energy stays in reasonable bounds."""
    nx = 40
    x0 = np.random.randn(nx)
    dt = 0.01
    steps = 1000
    
    traj = l96_integrate(x0, dt, steps, F=8.0, q_std=0.0)
    
    # Compute L2 norm over time
    energy = np.linalg.norm(traj, axis=1)
    
    # Energy should not explode or vanish
    assert np.all(energy > 0.1)
    assert np.all(energy < 100.0)
