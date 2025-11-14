import numpy as np
import pytest

from simulator.simulator_Multi_acoustic_tracking import (
    simulate_cv_targets,
    DynamicsConfig,
    article_initial_states,
)

def test_article_initial_states_on_cv():
    """Test that article_initial_states produces expected initial states for CV model."""
    dyn = DynamicsConfig(dt=1.0)
    rng = np.random.default_rng(0)
    X = simulate_cv_targets(
        n_steps=1,
        n_targets=4,
        area_xy=(40.0, 40.0),
        dyn_cfg=dyn,
        rng=rng,
        use_article_init=True,
    )
    np.testing.assert_allclose(X[0], article_initial_states(4))

def test_random_initialization_determinism_and_variation():
    """Test that random initialization is deterministic with same seed and varies with different seeds."""
    dyn = DynamicsConfig(dt=0.5)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    rng3 = np.random.default_rng(124)

    X1 = simulate_cv_targets(5, 3, (40.0, 40.0), dyn, rng1, use_article_init=False)
    X2 = simulate_cv_targets(5, 3, (40.0, 40.0), dyn, rng2, use_article_init=False)
    X3 = simulate_cv_targets(5, 3, (40.0, 40.0), dyn, rng3, use_article_init=False)

    np.testing.assert_allclose(X1, X2)   # same seed
    assert not np.allclose(X1, X3) or not np.allclose(X2, X3)  # different seed => different