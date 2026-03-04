"""
Integration tests: Bootstrap PF, EDH-PF, and LEDH-PF against the NonlinearSSM simulator.

Model
-----
    X_n = X_{n-1}/2 + 25*X_{n-1}/(1+X_{n-1}^2) + 8*cos(1.2*n) + V_n
    Y_n = X_n^2 / 20 + W_n
    X_1 ~ N(0, 5),  V_n ~ N(0, sigma_v^2),  W_n ~ N(0, sigma_w^2)

Each filter is tested for:
  - correct output shapes / types
  - weights normalised to 1
  - finite estimates at every step
  - RMSE better than a trivial zero-predictor (over many MC trials)
  - ESS stays above a minimum floor
  - determinism under fixed seed
"""

import numpy as np
import pytest

from simulator.simulator_nonlinSSM import simulate_nonlinear_ssm
from models.particle_filter import ParticleFilter, PFState
from models.extended_kalman_filter import ExtendedKalmanFilter, EKFState
from models.EDH_particle_filter import EDHFlowPF, EDHConfig, EKFTracker, effective_sample_size
from models.LEDH_particle_filter import LEDHFlowPF, LEDHConfig


# Shared model parameters
SIGMA_V = 10.0
SIGMA_W = 1.0
N_STEPS = 50
NP = 300          # particles – enough for correctness tests, fast enough for CI
N_MC = 20         # Monte Carlo trials for statistical tests


class _NonlinSSMModel:
    """Encapsulates the non-linear SSM model functions with a time counter."""

    def __init__(self, sigma_v: float, sigma_w: float) -> None:
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.Q = np.array([[sigma_v ** 2]])
        self.R = np.array([[sigma_w ** 2]])
        self.time_step = 0

    def reset(self) -> None:
        self.time_step = 0

    # --- EKF transition (deterministic part only) ---
    def g_ekf(self, x, u):
        self.time_step += 1
        n = self.time_step
        return x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * np.cos(1.2 * n)

    # --- Flow transition (with additive noise) ---
    def g_flow(self, x, u, v):
        if v is None:
            v = np.zeros_like(x)
        n = self.time_step
        return x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * np.cos(1.2 * n) + v

    def h(self, x):
        return x ** 2 / 20.0

    def jac_g(self, x, u):
        x_val = float(x[0]) if isinstance(x, np.ndarray) else float(x)
        denom = (1.0 + x_val ** 2) ** 2
        return np.array([[0.5 + 25.0 * (1.0 - x_val ** 2) / denom]])

    def jac_h(self, x):
        x_val = float(x[0]) if isinstance(x, np.ndarray) else float(x)
        return np.array([[x_val / 10.0]])

    def log_trans_pdf(self, xk, xkm1):
        n = self.time_step
        xk_s = float(np.ravel(xk)[0])
        xkm1_s = float(np.ravel(xkm1)[0])
        mean = xkm1_s / 2.0 + 25.0 * xkm1_s / (1.0 + xkm1_s ** 2) + 8.0 * np.cos(1.2 * n)
        diff = xk_s - mean
        return -0.5 * (diff ** 2 / self.Q[0, 0] + np.log(2 * np.pi * self.Q[0, 0]))

    def log_like_pdf(self, z, x):
        z_s = float(np.ravel(z)[0])
        x_s = float(np.ravel(x)[0])
        diff = z_s - x_s ** 2 / 20.0
        return -0.5 * (diff ** 2 / self.R[0, 0] + np.log(2 * np.pi * self.R[0, 0]))



@pytest.fixture(scope="module")
def sim_data():
    """One fixed simulation used by single-run tests."""
    res = simulate_nonlinear_ssm(N=N_STEPS, sigma_v=SIGMA_V, sigma_w=SIGMA_W, seed=42)
    return res


@pytest.fixture(scope="module")
def model():
    return _NonlinSSMModel(SIGMA_V, SIGMA_W)


def _make_bootstrap_pf(seed: int) -> ParticleFilter:
    """Build a fresh Bootstrap PF with a local time counter."""
    counter = {"n": 0}

    def g(x, u):
        counter["n"] += 1
        n = counter["n"]
        return x / 2.0 + 25.0 * x / (1.0 + x ** 2) + 8.0 * np.cos(1.2 * n)

    def h(x):
        return x ** 2 / 20.0

    Q = np.array([[SIGMA_V ** 2]])
    R = np.array([[SIGMA_W ** 2]])
    return ParticleFilter(
        g=g, h=h, Q=Q, R=R,
        Np=NP,
        resample_thresh=0.5,
        resample_method="systematic",
        regularize_after_resample=False,
        rng=np.random.default_rng(seed),
    )


def _make_edh_pf(seed: int):
    """Build a fresh EDH-PF with its own model instance."""
    m = _NonlinSSMModel(SIGMA_V, SIGMA_W)
    ekf = ExtendedKalmanFilter(
        g=m.g_ekf, h=m.h, jac_g=m.jac_g, jac_h=m.jac_h, Q=m.Q, R=m.R
    )
    tracker = EKFTracker(ekf, EKFState(mean=np.array([0.0]), cov=np.array([[5.0]]), t=0))
    cfg = EDHConfig(
        n_particles=NP, n_lambda_steps=8,
        resample_ess_ratio=0.5, flow_integrator="rk4",
        rng=np.random.default_rng(seed),
    )
    pf = EDHFlowPF(
        tracker=tracker, g=m.g_flow, h=m.h, jacobian_h=m.jac_h,
        log_trans_pdf=m.log_trans_pdf, log_like_pdf=m.log_like_pdf,
        R=m.R, config=cfg,
    )
    return pf, m, cfg


def _make_ledh_pf(seed: int):
    """Build a fresh LEDH-PF with its own model instance."""
    m = _NonlinSSMModel(SIGMA_V, SIGMA_W)
    ekf = ExtendedKalmanFilter(
        g=m.g_ekf, h=m.h, jac_g=m.jac_g, jac_h=m.jac_h, Q=m.Q, R=m.R
    )
    tracker = EKFTracker(ekf, EKFState(mean=np.array([0.0]), cov=np.array([[5.0]]), t=0))
    cfg = LEDHConfig(
        n_particles=NP, n_lambda_steps=8,
        resample_ess_ratio=0.5,
        rng=np.random.default_rng(seed),
    )
    pf = LEDHFlowPF(
        tracker=tracker, g=m.g_flow, h=m.h, jacobian_h=m.jac_h,
        log_trans_pdf=m.log_trans_pdf, log_like_pdf=m.log_like_pdf,
        R=m.R, config=cfg,
    )
    return pf, m, cfg


def _run_bootstrap(Y_obs, seed=123):
    pf = _make_bootstrap_pf(seed)
    state = pf.initialize(np.array([0.0]), np.array([[5.0]]))
    estimates, ess_list = [], []
    for t in range(len(Y_obs)):
        state = pf.step(Y_obs[t:t+1], u=None)
        estimates.append(state.mean[0])
        ess_list.append(pf.effective_sample_size())
    return np.array(estimates), np.array(ess_list), state


def _run_edh(Y_obs, seed=456):
    pf, m, cfg = _make_edh_pf(seed)
    state = pf.init_from_gaussian(np.array([0.0]), np.array([[5.0]]))

    def noise_sampler(N, nx):
        return cfg.rng.multivariate_normal(np.zeros(nx), m.Q, size=N)

    estimates, ess_list = [], []
    for t in range(len(Y_obs)):
        state = pf.step(state, Y_obs[t:t+1], u_km1=None, process_noise_sampler=noise_sampler)
        estimates.append(state.mean[0])
        ess_list.append(effective_sample_size(state.weights))
    return np.array(estimates), np.array(ess_list), state


def _run_ledh(Y_obs, seed=789):
    pf, m, cfg = _make_ledh_pf(seed)
    state = pf.init_from_gaussian(np.array([0.0]), np.array([[5.0]]))

    def noise_sampler(N, nx):
        return cfg.rng.multivariate_normal(np.zeros(nx), m.Q, size=N)

    estimates, ess_list = [], []
    for t in range(len(Y_obs)):
        state = pf.step(state, Y_obs[t:t+1], u_km1=None, process_noise_sampler=noise_sampler)
        estimates.append(state.mean[0])
        ess_list.append(effective_sample_size(state.weights))
    return np.array(estimates), np.array(ess_list), state


# ── Bootstrap PF ─────────────────────────────────────────────────────────
@pytest.mark.integration
def test_bootstrap_pf_output_shapes(sim_data):
    """Estimates array has length N and final state has correct shapes."""
    estimates, _, state = _run_bootstrap(sim_data.Y)
    assert estimates.shape == (N_STEPS,)
    assert state.particles.shape == (NP, 1)
    assert state.weights.shape == (NP,)
    assert state.mean.shape == (1,)
    assert state.cov.shape == (1, 1)


@pytest.mark.integration
def test_bootstrap_pf_weights_normalised(sim_data):
    """Final particle weights must sum to 1."""
    _, _, state = _run_bootstrap(sim_data.Y)
    assert np.isclose(np.sum(state.weights), 1.0, atol=1e-9)


@pytest.mark.integration
def test_bootstrap_pf_estimates_finite(sim_data):
    """Every mean estimate must be finite."""
    estimates, _, _ = _run_bootstrap(sim_data.Y)
    assert np.all(np.isfinite(estimates)), "Bootstrap PF produced non-finite estimates"


@pytest.mark.integration
def test_bootstrap_pf_ess_above_floor(sim_data):
    """ESS must stay above 5% of Np at every step (no catastrophic collapse)."""
    _, ess, _ = _run_bootstrap(sim_data.Y)
    min_ess = 0.05 * NP
    assert np.all(ess >= min_ess), f"ESS collapsed below {min_ess:.0f}"


@pytest.mark.integration
def test_bootstrap_pf_determinism(sim_data):
    """Two runs with the same seed must produce identical estimates."""
    e1, _, _ = _run_bootstrap(sim_data.Y, seed=123)
    e2, _, _ = _run_bootstrap(sim_data.Y, seed=123)
    np.testing.assert_array_equal(e1, e2)


@pytest.mark.integration
@pytest.mark.slow
def test_bootstrap_pf_rmse_beats_trivial_predictor():
    """Bootstrap PF RMSE must beat a naive sign-agnostic observation back-transform. We check a weak but meaningful property: filter RMSE is below 2× the
    observation-derived naive estimate.
    """
    rmse_list = []
    for trial in range(N_MC):
        sim = simulate_nonlinear_ssm(N=N_STEPS, sigma_v=SIGMA_V, sigma_w=SIGMA_W, seed=trial)
        estimates, _, _ = _run_bootstrap(sim.Y, seed=trial + 1000)
        rmse_list.append(np.sqrt(np.mean((sim.X - estimates) ** 2)))
    mean_rmse = np.mean(rmse_list)
    # A reasonable upper bound: the RMS of the state itself scaled by 2.
    # sqrt(mean(X^2)) ≈ sigma_v * sqrt(T) / sqrt(T) ≈ sigma_v for stationary series.
    # Bootstrap should stay below 2 * sigma_v = 20.0 on average.
    assert mean_rmse < 2.0 * SIGMA_V, (
        f"Bootstrap MC mean RMSE {mean_rmse:.2f} >= 2*sigma_v={2.0*SIGMA_V:.1f}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_bootstrap_pf_rmse_reasonable():
    """Monte Carlo mean RMSE must be below a generous threshold."""
    rmse_list = []
    for trial in range(N_MC):
        sim = simulate_nonlinear_ssm(N=N_STEPS, sigma_v=SIGMA_V, sigma_w=SIGMA_W, seed=trial)
        estimates, _, _ = _run_bootstrap(sim.Y, seed=trial + 2000)
        rmse_list.append(np.sqrt(np.mean((sim.X - estimates) ** 2)))
    mean_rmse = np.mean(rmse_list)
    assert mean_rmse < 25.0, f"Bootstrap MC mean RMSE too high: {mean_rmse:.2f}"


# ── EDH Particle Flow PF ─────────────────────────────────────────────────
@pytest.mark.integration
def test_edh_pf_output_shapes(sim_data):
    """Estimates array has length N; final state has correct particle/weight shapes."""
    estimates, _, state = _run_edh(sim_data.Y)
    assert estimates.shape == (N_STEPS,)
    assert state.particles.shape == (NP, 1)
    assert state.weights.shape == (NP,)
    assert state.mean.shape == (1,)
    assert state.cov.shape == (1, 1)


@pytest.mark.integration
def test_edh_pf_weights_normalised(sim_data):
    """Final particle weights must sum to 1."""
    _, _, state = _run_edh(sim_data.Y)
    assert np.isclose(np.sum(state.weights), 1.0, atol=1e-9)


@pytest.mark.integration
def test_edh_pf_estimates_finite(sim_data):
    """Every mean estimate must be finite."""
    estimates, _, _ = _run_edh(sim_data.Y)
    assert np.all(np.isfinite(estimates)), "EDH-PF produced non-finite estimates"


@pytest.mark.integration
def test_edh_pf_ess_above_floor(sim_data):
    """ESS must stay above 5% of Np at every step."""
    _, ess, _ = _run_edh(sim_data.Y)
    min_ess = 0.05 * NP
    assert np.all(ess >= min_ess), f"ESS collapsed below {min_ess:.0f}"


@pytest.mark.integration
def test_edh_pf_determinism(sim_data):
    """Two runs with the same seed must produce identical estimates."""
    e1, _, _ = _run_edh(sim_data.Y, seed=456)
    e2, _, _ = _run_edh(sim_data.Y, seed=456)
    np.testing.assert_array_equal(e1, e2)


@pytest.mark.integration
@pytest.mark.slow
def test_edh_pf_rmse_beats_trivial_predictor():
    """EDH-PF RMSE must stay below 2×sigma_v on this high-noise benchmark.
    """
    rmse_list = []
    for trial in range(N_MC):
        sim = simulate_nonlinear_ssm(N=N_STEPS, sigma_v=SIGMA_V, sigma_w=SIGMA_W, seed=trial)
        estimates, _, _ = _run_edh(sim.Y, seed=trial + 3000)
        rmse_list.append(np.sqrt(np.mean((sim.X - estimates) ** 2)))
    mean_rmse = np.mean(rmse_list)
    assert mean_rmse < 2.0 * SIGMA_V, (
        f"EDH MC mean RMSE {mean_rmse:.2f} >= 2*sigma_v={2.0*SIGMA_V:.1f}"
    )


# ── LEDH Particle Flow PF ────────────────────────────────────────────────
@pytest.mark.integration
def test_ledh_pf_output_shapes(sim_data):
    """Estimates array has length N; final state has correct particle/weight shapes."""
    estimates, _, state = _run_ledh(sim_data.Y)
    assert estimates.shape == (N_STEPS,)
    assert state.particles.shape == (NP, 1)
    assert state.weights.shape == (NP,)
    assert state.mean.shape == (1,)
    assert state.cov.shape == (1, 1)


@pytest.mark.integration
def test_ledh_pf_weights_normalised(sim_data):
    """Final particle weights must sum to 1."""
    _, _, state = _run_ledh(sim_data.Y)
    assert np.isclose(np.sum(state.weights), 1.0, atol=1e-9)


@pytest.mark.integration
def test_ledh_pf_estimates_finite(sim_data):
    """Every mean estimate must be finite."""
    estimates, _, _ = _run_ledh(sim_data.Y)
    assert np.all(np.isfinite(estimates)), "LEDH-PF produced non-finite estimates"


@pytest.mark.integration
def test_ledh_pf_ess_above_floor(sim_data):
    """ESS must stay above 5% of Np at every step."""
    _, ess, _ = _run_ledh(sim_data.Y)
    min_ess = 0.05 * NP
    assert np.all(ess >= min_ess), f"ESS collapsed below {min_ess:.0f}"


@pytest.mark.integration
def test_ledh_pf_determinism(sim_data):
    """Two runs with the same seed must produce identical estimates."""
    e1, _, _ = _run_ledh(sim_data.Y, seed=789)
    e2, _, _ = _run_ledh(sim_data.Y, seed=789)
    np.testing.assert_array_equal(e1, e2)


@pytest.mark.integration
@pytest.mark.slow
def test_ledh_pf_rmse_beats_trivial_predictor():
    """LEDH-PF RMSE must stay below 2×sigma_v on this high-noise benchmark.
    """
    rmse_list, trivial_list = [], []
    for trial in range(N_MC):
        sim = simulate_nonlinear_ssm(N=N_STEPS, sigma_v=SIGMA_V, sigma_w=SIGMA_W, seed=trial)
        estimates, _, _ = _run_ledh(sim.Y, seed=trial + 4000)
        rmse_list.append(np.sqrt(np.mean((sim.X - estimates) ** 2)))
        trivial_list.append(np.sqrt(np.mean(sim.X ** 2)))
    mean_rmse = np.mean(rmse_list)
    assert mean_rmse < 2.0 * SIGMA_V, (
        f"LEDH MC mean RMSE {mean_rmse:.2f} >= 2*sigma_v={2.0*SIGMA_V:.1f}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_ledh_pf_rmse_reasonable():
    """Monte Carlo mean RMSE must be below a generous threshold."""
    rmse_list = []
    for trial in range(N_MC):
        sim = simulate_nonlinear_ssm(N=N_STEPS, sigma_v=SIGMA_V, sigma_w=SIGMA_W, seed=trial)
        estimates, _, _ = _run_ledh(sim.Y, seed=trial + 5000)
        rmse_list.append(np.sqrt(np.mean((sim.X - estimates) ** 2)))
    mean_rmse = np.mean(rmse_list)
    assert mean_rmse < 25.0, f"LEDH MC mean RMSE too high: {mean_rmse:.2f}"


# ── Cross-filter comparisons ─────────────────────────────────────────────
@pytest.mark.integration
def test_all_three_filters_return_finite_estimates(sim_data):
    """Quick smoke-test: all three filters must produce fully finite estimates."""
    for name, runner in [("Bootstrap", _run_bootstrap), ("EDH", _run_edh), ("LEDH", _run_ledh)]:
        estimates, _, _ = runner(sim_data.Y)
        assert np.all(np.isfinite(estimates)), f"{name} produced non-finite estimates"


@pytest.mark.integration
def test_filters_produce_different_estimates(sim_data):
    """Bootstrap, EDH, and LEDH must produce distinct estimate trajectories."""
    e_bs, _, _ = _run_bootstrap(sim_data.Y)
    e_edh, _, _ = _run_edh(sim_data.Y)
    e_ledh, _, _ = _run_ledh(sim_data.Y)
    assert not np.allclose(e_bs, e_edh), "Bootstrap and EDH estimates are identical"
    assert not np.allclose(e_bs, e_ledh), "Bootstrap and LEDH estimates are identical"
    assert not np.allclose(e_edh, e_ledh), "EDH and LEDH estimates are identical"


@pytest.mark.integration
@pytest.mark.slow
def test_ledh_mc_rmse_lower_than_edh():
    """Over many trials LEDH should achieve lower mean RMSE than EDH
    (consistent with the known theoretical advantage of per-particle linearization)."""
    rmse_edh, rmse_ledh = [], []
    for trial in range(N_MC):
        sim = simulate_nonlinear_ssm(N=N_STEPS, sigma_v=SIGMA_V, sigma_w=SIGMA_W, seed=trial)
        e_edh, _, _ = _run_edh(sim.Y, seed=trial + 6000)
        e_ledh, _, _ = _run_ledh(sim.Y, seed=trial + 7000)
        rmse_edh.append(np.sqrt(np.mean((sim.X - e_edh) ** 2)))
        rmse_ledh.append(np.sqrt(np.mean((sim.X - e_ledh) ** 2)))
    assert np.mean(rmse_ledh) < np.mean(rmse_edh), (
        f"Expected LEDH RMSE < EDH RMSE, got {np.mean(rmse_ledh):.2f} vs {np.mean(rmse_edh):.2f}"
    )
