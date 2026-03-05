"""Unit tests for models/tf_particle_filter.py.

Covers
------
* systematic_resample_tf  – output shape, index bounds, weight preservation
* ParticleFilterTF        – initialization, step shapes/types, ESS sanity
* run_particle_filter_tf  – full-sequence output shapes and ESS
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from models.tf_particle_filter import (
    ParticleFilterTF,
    PFStateTF,
    run_particle_filter_tf,
    systematic_resample_tf,
)

F32 = tf.float32


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _np2tf(x: np.ndarray) -> tf.Tensor:
    return tf.constant(x, dtype=F32)


def _linear_g(x: tf.Tensor, u) -> tf.Tensor:
    """Simple AR(1) scalar dynamics broadcast to (N, nx=2)."""
    Phi = tf.constant([[0.9, 0.0], [0.0, 0.7]], dtype=F32)
    return tf.linalg.matvec(Phi, x)


def _linear_h(X: tf.Tensor) -> tf.Tensor:
    """H = [1, 0]: maps (N, 2) particles to (N, 1) observations."""
    return X[:, :1]


# ---------------------------------------------------------------------------
# systematic_resample_tf
# ---------------------------------------------------------------------------

class TestSystematicResample:
    """Tests for the systematic resampling subroutine."""

    @pytest.fixture
    def uniform_weights(self):
        N = 64
        return tf.ones(N, dtype=F32) / tf.cast(N, F32)

    def test_output_shape(self, uniform_weights):
        N = int(uniform_weights.shape[0])
        idx = systematic_resample_tf(uniform_weights, seed=tf.constant(0))
        assert idx.shape == (N,), f"Expected ({N},), got {idx.shape}"

    def test_index_bounds(self, uniform_weights):
        N = int(uniform_weights.shape[0])
        idx = systematic_resample_tf(uniform_weights, seed=tf.constant(7)).numpy()
        assert idx.min() >= 0,     f"Negative index: {idx.min()}"
        assert idx.max() < N,      f"Index out of bounds: {idx.max()} >= {N}"

    def test_uniform_weights_cover_all(self, uniform_weights):
        """For uniform weights each particle should be selected roughly once."""
        N = int(uniform_weights.shape[0])
        idx = systematic_resample_tf(uniform_weights, seed=tf.constant(1)).numpy()
        unique = np.unique(idx)
        # With uniform weights systematic resampling selects each particle exactly once
        assert len(unique) == N, f"Not all particles selected: {len(unique)} / {N}"

    def test_peaked_weights_concentrate(self):
        """A single high-weight particle should dominate the sample."""
        N = 100
        w = tf.constant(
            [0.98] + [0.02 / (N - 1)] * (N - 1), dtype=F32
        )
        idx = systematic_resample_tf(w, seed=tf.constant(5)).numpy()
        # Particle 0 should appear ~98 times out of 100
        count_0 = np.sum(idx == 0)
        assert count_0 >= 90, f"Peak particle selected {count_0} times (expected ≥90)"

    def test_dtype_int(self, uniform_weights):
        idx = systematic_resample_tf(uniform_weights, seed=tf.constant(0))
        assert idx.dtype in (tf.int32, tf.int64), f"Unexpected dtype: {idx.dtype}"


# ---------------------------------------------------------------------------
# ParticleFilterTF – initialization
# ---------------------------------------------------------------------------

class TestParticleFilterInit:
    """Tests for ParticleFilterTF.initialize()."""

    @pytest.fixture
    def small_pf(self):
        Q = np.diag([0.05, 0.02]).astype(np.float32)
        R = np.array([[0.10]], dtype=np.float32)
        return ParticleFilterTF(
            _linear_g, _linear_h, Q, R,
            n_particles=50, resample_thresh=0.5, vectorised_g=False,
        )

    def test_init_returns_pfstate(self, small_pf):
        m0 = np.zeros(2, dtype=np.float32)
        P0 = np.eye(2, dtype=np.float32)
        state = small_pf.initialize(m0, P0)
        assert isinstance(state, PFStateTF)

    def test_init_particle_shape(self, small_pf):
        m0 = np.zeros(2, dtype=np.float32)
        P0 = np.eye(2, dtype=np.float32)
        state = small_pf.initialize(m0, P0)
        assert state.particles.shape == (50, 2)

    def test_init_weight_shape(self, small_pf):
        m0 = np.zeros(2, dtype=np.float32)
        P0 = np.eye(2, dtype=np.float32)
        state = small_pf.initialize(m0, P0)
        assert state.weights.shape == (50,)

    def test_init_weights_sum_to_one(self, small_pf):
        m0 = np.zeros(2, dtype=np.float32)
        P0 = np.eye(2, dtype=np.float32)
        state = small_pf.initialize(m0, P0)
        w_sum = float(tf.reduce_sum(state.weights))
        assert abs(w_sum - 1.0) < 1e-5, f"Weights sum to {w_sum}"

    def test_init_mean_shape(self, small_pf):
        m0 = np.zeros(2, dtype=np.float32)
        P0 = np.eye(2, dtype=np.float32)
        state = small_pf.initialize(m0, P0)
        assert state.mean.shape == (2,)

    def test_init_cov_shape(self, small_pf):
        m0 = np.zeros(2, dtype=np.float32)
        P0 = np.eye(2, dtype=np.float32)
        state = small_pf.initialize(m0, P0)
        assert state.cov.shape == (2, 2)

    def test_init_mean_close_to_m0(self, small_pf):
        """With N=50 particles and small P0, empirical mean should be near m0."""
        Q = np.diag([0.05, 0.02]).astype(np.float32)
        R = np.array([[0.10]], dtype=np.float32)
        pf = ParticleFilterTF(
            _linear_g, _linear_h, Q, R,
            n_particles=2000, resample_thresh=0.5,
        )
        m0 = np.array([1.0, -0.5], dtype=np.float32)
        P0 = 0.01 * np.eye(2, dtype=np.float32)
        state = pf.initialize(m0, P0)
        diff = np.abs(state.mean.numpy() - m0)
        assert np.all(diff < 0.1), f"Init mean far from m0: diff={diff}"


# ---------------------------------------------------------------------------
# ParticleFilterTF – step
# ---------------------------------------------------------------------------

class TestParticleFilterStep:
    """Tests for ParticleFilterTF.step()."""

    @pytest.fixture
    def pf_with_init(self):
        Q = np.diag([0.05, 0.02]).astype(np.float32)
        R = np.array([[0.10]], dtype=np.float32)
        pf = ParticleFilterTF(
            _linear_g, _linear_h, Q, R,
            n_particles=100, resample_thresh=0.5, vectorised_g=False,
        )
        m0 = np.zeros(2, dtype=np.float32)
        P0 = np.eye(2, dtype=np.float32)
        pf.initialize(m0, P0)
        return pf

    def test_step_returns_pfstate(self, pf_with_init):
        y = np.array([0.2], dtype=np.float32)
        state = pf_with_init.step(y)
        assert isinstance(state, PFStateTF)

    def test_step_particle_shape_preserved(self, pf_with_init):
        y = np.array([0.2], dtype=np.float32)
        state = pf_with_init.step(y)
        assert state.particles.shape == (100, 2)

    def test_step_weights_shape_preserved(self, pf_with_init):
        y = np.array([0.2], dtype=np.float32)
        state = pf_with_init.step(y)
        assert state.weights.shape == (100,)

    def test_step_weights_sum_to_one(self, pf_with_init):
        y = np.array([0.2], dtype=np.float32)
        state = pf_with_init.step(y)
        w_sum = float(tf.reduce_sum(state.weights))
        assert abs(w_sum - 1.0) < 1e-5, f"Post-step weights sum to {w_sum}"

    def test_step_mean_shape(self, pf_with_init):
        y = np.array([0.2], dtype=np.float32)
        state = pf_with_init.step(y)
        assert state.mean.shape == (2,)

    def test_step_cov_shape(self, pf_with_init):
        y = np.array([0.2], dtype=np.float32)
        state = pf_with_init.step(y)
        assert state.cov.shape == (2, 2)

    def test_step_cov_symmetric(self, pf_with_init):
        y = np.array([0.2], dtype=np.float32)
        state = pf_with_init.step(y)
        cov = state.cov.numpy()
        assert np.allclose(cov, cov.T, atol=1e-5), "Covariance not symmetric"

    def test_step_values_finite(self, pf_with_init):
        y = np.array([0.2], dtype=np.float32)
        state = pf_with_init.step(y)
        assert np.all(np.isfinite(state.mean.numpy()))
        assert np.all(np.isfinite(state.cov.numpy()))

    def test_step_increments_t(self, pf_with_init):
        y = np.array([0.2], dtype=np.float32)
        state1 = pf_with_init.step(y)
        state2 = pf_with_init.step(y)
        assert state2.t == state1.t + 1

    def test_no_resampling_when_thresh_zero(self):
        """resample_thresh=0 should leave particles un-resampled (weight spread)."""
        Q = np.diag([0.05, 0.02]).astype(np.float32)
        R = np.array([[0.10]], dtype=np.float32)
        pf = ParticleFilterTF(
            _linear_g, _linear_h, Q, R,
            n_particles=100, resample_thresh=0.0,
        )
        pf.initialize(np.zeros(2, dtype=np.float32), np.eye(2, dtype=np.float32))
        # Extreme observation to cause weight degeneracy
        state = pf.step(np.array([100.0], dtype=np.float32))
        w = state.weights.numpy()
        # With no resampling, weights won't be uniform
        assert not np.allclose(w, 1.0 / 100, atol=1e-3), \
            "Expected non-uniform weights when resampling disabled"


# ---------------------------------------------------------------------------
# run_particle_filter_tf
# ---------------------------------------------------------------------------

class TestRunParticleFilterTF:
    """Tests for the convenience run_particle_filter_tf wrapper."""

    @pytest.fixture
    def simple_scenario(self):
        rng = np.random.default_rng(0)
        T   = 15
        Q   = np.diag([0.05, 0.02]).astype(np.float32)
        R   = np.array([[0.10]], dtype=np.float32)
        Y   = rng.normal(scale=0.3, size=(T, 1)).astype(np.float32)
        m0  = np.zeros(2, dtype=np.float32)
        P0  = np.eye(2, dtype=np.float32)
        pf  = ParticleFilterTF(
            _linear_g, _linear_h, Q, R,
            n_particles=200, resample_thresh=0.5, vectorised_g=False,
        )
        return pf, Y, m0, P0, T

    def test_means_shape(self, simple_scenario):
        pf, Y, m0, P0, T = simple_scenario
        means, covs, ess = run_particle_filter_tf(pf, Y, m0, P0)
        assert means.shape == (T, 2), f"means: {means.shape}"

    def test_covs_shape(self, simple_scenario):
        pf, Y, m0, P0, T = simple_scenario
        means, covs, ess = run_particle_filter_tf(pf, Y, m0, P0)
        assert covs.shape == (T, 2, 2), f"covs: {covs.shape}"

    def test_ess_shape(self, simple_scenario):
        pf, Y, m0, P0, T = simple_scenario
        means, covs, ess = run_particle_filter_tf(pf, Y, m0, P0)
        assert ess.shape == (T,), f"ess: {ess.shape}"

    def test_ess_bounds(self, simple_scenario):
        pf, Y, m0, P0, T = simple_scenario
        means, covs, ess = run_particle_filter_tf(pf, Y, m0, P0)
        assert np.all(ess > 0),   f"ESS ≤ 0 at some step: {ess}"
        assert np.all(ess <= 200 + 1), f"ESS > N at some step: {ess}"

    def test_means_finite(self, simple_scenario):
        pf, Y, m0, P0, T = simple_scenario
        means, covs, ess = run_particle_filter_tf(pf, Y, m0, P0)
        assert np.all(np.isfinite(means)), "means contains non-finite values"

    def test_covs_finite(self, simple_scenario):
        pf, Y, m0, P0, T = simple_scenario
        means, covs, ess = run_particle_filter_tf(pf, Y, m0, P0)
        assert np.all(np.isfinite(covs)), "covs contains non-finite values"

    def test_covs_symmetric(self, simple_scenario):
        pf, Y, m0, P0, T = simple_scenario
        means, covs, ess = run_particle_filter_tf(pf, Y, m0, P0)
        for t in range(covs.shape[0]):
            assert np.allclose(covs[t], covs[t].T, atol=1e-5), f"cov[{t}] not symmetric"

    def test_returns_numpy(self, simple_scenario):
        pf, Y, m0, P0, T = simple_scenario
        means, covs, ess = run_particle_filter_tf(pf, Y, m0, P0)
        assert isinstance(means, np.ndarray)
        assert isinstance(covs, np.ndarray)
        assert isinstance(ess, np.ndarray)

    def test_ess_above_half_with_good_model(self):
        """For on-model observations, ESS should remain above N/2 after resampling."""
        rng = np.random.default_rng(1)
        T   = 20
        nx, ny = 2, 1
        Q   = np.diag([0.05, 0.02]).astype(np.float32)
        R   = np.array([[0.10]], dtype=np.float32)
        # Generate on-model observations
        Y   = rng.normal(scale=0.3, size=(T, ny)).astype(np.float32)
        m0  = np.zeros(nx, dtype=np.float32)
        P0  = np.eye(nx, dtype=np.float32)
        N   = 300
        pf  = ParticleFilterTF(
            _linear_g, _linear_h, Q, R,
            n_particles=N, resample_thresh=0.5, vectorised_g=False,
        )
        _, _, ess = run_particle_filter_tf(pf, Y, m0, P0)
        # After resampling, ESS should not be pathologically low
        assert np.all(ess > N * 0.1), \
            f"ESS collapsed below 10%N at some step: {ess}"
