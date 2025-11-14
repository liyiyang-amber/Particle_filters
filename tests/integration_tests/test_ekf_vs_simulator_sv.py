"""Integration tests for Extended Kalman Filter with 1D stochastic volatility model."""

import numpy as np
import pytest
from models.extended_kalman_filter import ExtendedKalmanFilter, EKFState
from simulator.simulator_sto_volatility_model import simulate_sv_1d


@pytest.fixture
def sv_params():
    """Standard 1D stochastic volatility model parameters."""
    return {
        "alpha": 0.9,
        "sigma": 0.2,
        "beta": 1.0,
        "n": 500,
        "seed": 42
    }


@pytest.fixture
def sv_simulated_data(sv_params):
    """Generate 1D stochastic volatility simulation data."""
    results = simulate_sv_1d(
        n=sv_params["n"],
        alpha=sv_params["alpha"],
        sigma=sv_params["sigma"],
        beta=sv_params["beta"],
        seed=sv_params["seed"]
    )
    return results


@pytest.fixture
def sv_ekf_setup(sv_simulated_data, sv_params):
    """Setup EKF for 1D stochastic volatility model."""
    X_true = sv_simulated_data.X  # (n,) array
    Y_obs = sv_simulated_data.Y   # (n,) array
    n = len(X_true)
    
    alpha = sv_params["alpha"]
    sigma = sv_params["sigma"]
    beta = sv_params["beta"]
    
    # Process and measurement noise
    Q = np.array([[sigma**2]])  # 1x1 matrix
    R = np.array([[0.1]])  # Measurement noise (assumed)

    # Define model functions for scalar case
    def g(x, u):
        """Latent transition: x_k = alpha * x_{k-1} + w."""
        return np.array([alpha * x[0]])

    def h(x):
        """Observation model: y_k = beta * exp(0.5 * x_k) + v."""
        return np.array([beta * np.exp(0.5 * x[0])])

    def jac_g(x, u):
        """Jacobian of g w.r.t. x."""
        return np.array([[alpha]])

    def jac_h(x):
        """Jacobian of h w.r.t. x."""
        return np.array([[0.5 * beta * np.exp(0.5 * x[0])]])

    return {
        "X_true": X_true,
        "Y_obs": Y_obs,
        "g": g,
        "h": h,
        "jac_g": jac_g,
        "jac_h": jac_h,
        "Q": Q,
        "R": R,
        "n": n,
    }


@pytest.mark.integration
def test_ekf_sv_basic_run(sv_ekf_setup):
    """Test that EKF runs successfully on 1D SV data."""
    s = sv_ekf_setup
    ekf = ExtendedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        jac_g=s["jac_g"],
        jac_h=s["jac_h"],
        joseph=True,
        jitter=1e-8,
    )

    x0 = np.array([0.0])
    P0 = np.array([[1.0]])
    state = EKFState(x0, P0, t=0)

    # Run filtering
    est_X = np.zeros(s["n"])
    est_X[0] = x0[0]

    for k in range(1, s["n"]):
        z_k = np.array([s["Y_obs"][k]])  # Convert scalar to 1D array
        state = ekf.step(state, z_k)
        est_X[k] = state.mean[0]

    # Basic checks
    assert state.t == s["n"] - 1  # t starts at 0, increments to n-1
    assert np.all(np.isfinite(est_X))
    assert np.all(np.isfinite(state.cov))


@pytest.mark.integration
def test_ekf_sv_tracking_performance(sv_ekf_setup):
    """Test EKF tracking performance on 1D SV data."""
    s = sv_ekf_setup
    ekf = ExtendedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        jac_g=s["jac_g"],
        jac_h=s["jac_h"],
        joseph=True,
        jitter=1e-8,
    )

    x0 = np.array([0.0])
    P0 = np.array([[1.0]])
    state = EKFState(x0, P0, t=0)

    # Run filtering
    est_X = np.zeros(s["n"])
    est_X[0] = x0[0]

    for k in range(1, s["n"]):
        z_k = np.array([s["Y_obs"][k]])
        state = ekf.step(state, z_k)
        est_X[k] = state.mean[0]

    # Compute RMSE
    rmse = np.sqrt(np.mean((est_X - s["X_true"]) ** 2))

    # For SV model, EKF should achieve reasonable tracking
    print(f"EKF RMSE: {rmse:.6f}")

    # Check that RMSE is reasonable (not perfect but not terrible)
    assert rmse < 2.0, f"RMSE too high: {rmse}"


@pytest.mark.integration
def test_ekf_sv_covariance_stability(sv_ekf_setup):
    """Test that covariances remain stable during 1D SV filtering with Joseph form."""
    s = sv_ekf_setup
    ekf = ExtendedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        jac_g=s["jac_g"],
        jac_h=s["jac_h"],
        joseph=True,
        jitter=1e-8,
    )

    state = EKFState(mean=np.array([0.0]), cov=np.array([[1.0]]), t=0)

    cov_traces = []
    for k in range(1, s["n"]):
        z_k = np.array([s["Y_obs"][k]])
        state = ekf.step(state, z_k)
        cov_traces.append(np.trace(state.cov))

        # With Joseph form, symmetry should be better maintained
        assert np.allclose(state.cov, state.cov.T, atol=1e-8)

        # Check positive-definiteness
        eigvals = np.linalg.eigvalsh(state.cov)
        assert np.all(eigvals > -1e-10), f"Negative eigenvalues at step {k}: {eigvals}"

    # Check that trace doesn't explode
    cov_traces = np.array(cov_traces)
    assert np.all(cov_traces < 10), "Covariance trace exploded"


@pytest.mark.integration
def test_ekf_sv_joseph_vs_standard(sv_ekf_setup):
    """Compare Joseph form vs standard covariance update for 1D SV."""
    s = sv_ekf_setup

    ekf_joseph = ExtendedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        jac_g=s["jac_g"],
        jac_h=s["jac_h"],
        joseph=True,
        jitter=1e-8,
    )

    ekf_standard = ExtendedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        jac_g=s["jac_g"],
        jac_h=s["jac_h"],
        joseph=False,
        jitter=1e-8,
    )

    state_j = EKFState(mean=np.array([0.0]), cov=np.array([[1.0]]), t=0)
    state_s = EKFState(mean=np.array([0.0]), cov=np.array([[1.0]]), t=0)

    # Run both filters
    for k in range(1, min(100, s["n"])):  # First 100 steps
        z_k = np.array([s["Y_obs"][k]])
        state_j = ekf_joseph.step(state_j, z_k)
        state_s = ekf_standard.step(state_s, z_k)

    # Results should be similar
    np.testing.assert_allclose(state_j.mean, state_s.mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_j.cov, state_s.cov, rtol=1e-4, atol=1e-6)


@pytest.mark.integration
def test_ekf_sv_numerical_jacobians(sv_ekf_setup):
    """Test EKF with numerical Jacobians on 1D SV data."""
    s = sv_ekf_setup

    # EKF with numerical Jacobians
    ekf_numerical = ExtendedKalmanFilter(
        g=s["g"], h=s["h"], Q=s["Q"], R=s["R"], joseph=True, jitter=1e-8
    )

    # EKF with analytic Jacobians
    ekf_analytic = ExtendedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        jac_g=s["jac_g"],
        jac_h=s["jac_h"],
        joseph=True,
        jitter=1e-8,
    )

    state_num = EKFState(mean=np.array([0.0]), cov=np.array([[1.0]]), t=0)
    state_ana = EKFState(mean=np.array([0.0]), cov=np.array([[1.0]]), t=0)

    # Run both filters for a subset of data
    for k in range(1, min(50, s["n"])):
        z_k = np.array([s["Y_obs"][k]])
        state_num = ekf_numerical.step(state_num, z_k)
        state_ana = ekf_analytic.step(state_ana, z_k)

    # Results should be close
    np.testing.assert_allclose(state_num.mean, state_ana.mean, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(state_num.cov, state_ana.cov, rtol=1e-3, atol=1e-4)


@pytest.mark.integration
def test_ekf_sv_consistency_check(sv_ekf_setup):
    """Test innovation consistency for EKF on 1D SV data."""
    s = sv_ekf_setup
    ekf = ExtendedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        jac_g=s["jac_g"],
        jac_h=s["jac_h"],
        joseph=True,
        jitter=1e-8,
    )

    state = EKFState(mean=np.array([0.0]), cov=np.array([[1.0]]), t=0)

    innovations = []

    for k in range(1, s["n"]):
        pred = ekf.predict(state)

        # Compute innovation
        z_pred = s["h"](pred.mean)
        innov = np.array([s["Y_obs"][k]]) - z_pred

        innovations.append(innov[0])

        state = ekf.update(pred, np.array([s["Y_obs"][k]]))

    innovations = np.array(innovations)

    # Check innovation statistics
    innov_mean = np.mean(innovations)
    # Relax this check for the nonlinear SV model
    assert np.abs(innov_mean) < 1.0, f"Innovation mean too large: {innov_mean}"


@pytest.mark.integration
def test_ekf_sv_different_initializations(sv_ekf_setup):
    """Test EKF with different initial conditions for 1D SV."""
    s = sv_ekf_setup

    ekf = ExtendedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        jac_g=s["jac_g"],
        jac_h=s["jac_h"],
        joseph=True,
        jitter=1e-8,
    )

    # Different initializations
    inits = [
        {"mean": np.array([0.0]), "cov": np.array([[1.0]])},
        {"mean": np.array([1.0]), "cov": np.array([[2.0]])},
        {"mean": np.array([-1.0]), "cov": np.array([[0.5]])},
    ]

    final_states = []

    for init in inits:
        state = EKFState(mean=init["mean"], cov=init["cov"], t=0)

        # Run filtering
        for k in range(1, min(200, s["n"])):  # First 200 steps
            z_k = np.array([s["Y_obs"][k]])
            state = ekf.step(state, z_k)

        final_states.append(state)

    # After enough steps, results should converge regardless of initialization
    for i in range(len(final_states) - 1):
        diff = np.linalg.norm(final_states[i].mean - final_states[i + 1].mean)
        assert diff < 1.0, f"Final states diverged: {diff}"


@pytest.mark.integration
def test_ekf_sv_reproducibility(sv_ekf_setup):
    """Test that EKF produces reproducible results for 1D SV."""
    s = sv_ekf_setup

    def run_ekf():
        ekf = ExtendedKalmanFilter(
            g=s["g"],
            h=s["h"],
            Q=s["Q"],
            R=s["R"],
            jac_g=s["jac_g"],
            jac_h=s["jac_h"],
            joseph=True,
            jitter=1e-8,
        )
        state = EKFState(mean=np.array([0.0]), cov=np.array([[1.0]]), t=0)
        for k in range(1, s["n"]):
            z_k = np.array([s["Y_obs"][k]])
            state = ekf.step(state, z_k)
        return state

    result1 = run_ekf()
    result2 = run_ekf()

    # Results should be identical
    np.testing.assert_array_equal(result1.mean, result2.mean)
    np.testing.assert_array_equal(result1.cov, result2.cov)
