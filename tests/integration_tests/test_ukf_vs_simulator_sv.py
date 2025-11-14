"""Integration tests for Unscented Kalman Filter with 1D stochastic volatility model."""

import numpy as np
import pytest
from models.unscented_kalman_filter import UnscentedKalmanFilter, UKFState
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
def sv_ukf_setup(sv_simulated_data, sv_params):
    """Setup UKF for 1D stochastic volatility model."""
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
    def g(x, u=None):
        """Latent transition: x_k = alpha * x_{k-1} + w."""
        return np.array([alpha * x[0]])

    def h(x):
        """Observation model: y_k = beta * exp(0.5 * x_k) + v."""
        return np.array([beta * np.exp(0.5 * x[0])])

    return {
        "X_true": X_true,
        "Y_obs": Y_obs,
        "g": g,
        "h": h,
        "Q": Q,
        "R": R,
        "n": n,
    }


@pytest.mark.integration
def test_ukf_sv_basic_run(sv_ukf_setup):
    """Test that UKF runs successfully on 1D SV data."""
    s = sv_ukf_setup
    ukf = UnscentedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        alpha=1e-3,
        beta=2.0,
        kappa=0.0,
        jitter=1e-9,
    )

    x0 = np.array([0.0])
    P0 = np.array([[1.0]])
    state = UKFState(mean=x0, cov=P0, t=0)

    # Run filtering
    est_X = np.zeros(s["n"])
    est_X[0] = x0[0]

    for k in range(1, s["n"]):
        z_k = np.array([s["Y_obs"][k]])
        state = ukf.step(state, z_k)
        est_X[k] = state.mean[0]

    # Basic checks
    assert state.t == s["n"] - 1
    assert np.all(np.isfinite(est_X))
    assert np.all(np.isfinite(state.cov))


@pytest.mark.integration
def test_ukf_sv_tracking_performance(sv_ukf_setup):
    """Test UKF tracking performance on 1D SV data."""
    s = sv_ukf_setup
    ukf = UnscentedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        alpha=1e-3,
        beta=2.0,
        kappa=0.0,
        jitter=1e-9,
    )

    x0 = np.array([0.0])
    P0 = np.array([[1.0]])
    state = UKFState(mean=x0, cov=P0, t=0)

    # Run filtering
    est_X = np.zeros(s["n"])
    est_X[0] = x0[0]

    for k in range(1, s["n"]):
        z_k = np.array([s["Y_obs"][k]])
        state = ukf.step(state, z_k)
        est_X[k] = state.mean[0]

    # Compute RMSE
    rmse = np.sqrt(np.mean((est_X - s["X_true"]) ** 2))

    print(f"UKF RMSE: {rmse:.6f}")
    assert rmse < 2.0, f"RMSE too high: {rmse}"


@pytest.mark.integration
def test_ukf_sv_covariance_stability(sv_ukf_setup):
    """Test that covariances remain stable during 1D SV filtering."""
    s = sv_ukf_setup
    ukf = UnscentedKalmanFilter(
        g=s["g"],
        h=s["h"],
        Q=s["Q"],
        R=s["R"],
        alpha=1e-3,
        beta=2.0,
        kappa=0.0,
        jitter=1e-9,
    )

    state = UKFState(mean=np.array([0.0]), cov=np.array([[1.0]]), t=0)

    cov_traces = []
    for k in range(1, s["n"]):
        z_k = np.array([s["Y_obs"][k]])
        state = ukf.step(state, z_k)
        cov_traces.append(np.trace(state.cov))

        # Check symmetry
        assert np.allclose(state.cov, state.cov.T, atol=1e-9)

        # Check positive-definiteness
        eigvals = np.linalg.eigvalsh(state.cov)
        assert np.all(eigvals > 0), f"Non-positive eigenvalues at step {k}: {eigvals}"

    # Check that trace doesn't explode
    cov_traces = np.array(cov_traces)
    assert np.all(cov_traces < 10), "Covariance trace exploded"


@pytest.mark.integration
def test_ukf_sv_different_alpha_values(sv_ukf_setup):
    """Test UKF with different alpha values on 1D SV data."""
    s = sv_ukf_setup

    alphas = [1e-4, 1e-3, 0.1]
    rmses = []

    for alpha in alphas:
        ukf = UnscentedKalmanFilter(
            g=s["g"],
            h=s["h"],
            Q=s["Q"],
            R=s["R"],
            alpha=alpha,
            beta=2.0,
            kappa=0.0,
            jitter=1e-9,
        )

        state = UKFState(mean=np.array([0.0]), cov=np.array([[1.0]]), t=0)
        est_X = np.zeros(s["n"])

        for k in range(1, min(200, s["n"])):
            z_k = np.array([s["Y_obs"][k]])
            state = ukf.step(state, z_k)
            est_X[k] = state.mean[0]

        rmse = np.sqrt(np.mean((est_X[:200] - s["X_true"][:200]) ** 2))
        rmses.append(rmse)

    # All should produce reasonable results
    for alpha, rmse in zip(alphas, rmses):
        print(f"Alpha={alpha}: RMSE={rmse:.6f}")
        assert rmse < 5.0, f"RMSE too high for alpha={alpha}: {rmse}"


@pytest.mark.integration
def test_ukf_sv_reproducibility(sv_ukf_setup):
    """Test that UKF produces reproducible results for 1D SV."""
    s = sv_ukf_setup

    def run_ukf():
        ukf = UnscentedKalmanFilter(
            g=s["g"],
            h=s["h"],
            Q=s["Q"],
            R=s["R"],
            alpha=1e-3,
            beta=2.0,
            kappa=0.0,
            jitter=1e-9,
        )
        state = UKFState(mean=np.array([0.0]), cov=np.array([[1.0]]), t=0)
        for k in range(1, s["n"]):
            z_k = np.array([s["Y_obs"][k]])
            state = ukf.step(state, z_k)
        return state

    result1 = run_ukf()
    result2 = run_ukf()

    # Results should be identical
    np.testing.assert_array_equal(result1.mean, result2.mean)
    np.testing.assert_array_equal(result1.cov, result2.cov)
