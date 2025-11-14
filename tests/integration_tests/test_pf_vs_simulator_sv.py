"""Integration tests for Particle Filter with 1D stochastic volatility model."""

import numpy as np
import pytest
from models.particle_filter import ParticleFilter, PFState
from simulator.simulator_sto_volatility_model import simulate_sv_1d


@pytest.fixture
def sv_params():
    """Standard 1D stochastic volatility model parameters."""
    return {
        "alpha": 0.9,
        "sigma": 0.2,
        "beta": 1.0,
        "n": 200,
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


@pytest.mark.integration
def test_pf_sv_basic_filtering(sv_simulated_data, sv_params):
    """Test basic particle filter on simulated 1D SV data."""
    X_true = sv_simulated_data.X
    Y_obs = sv_simulated_data.Y
    n = len(X_true)

    alpha = sv_params["alpha"]
    sigma = sv_params["sigma"]
    beta = sv_params["beta"]

    # Process and measurement noise
    Q = np.array([[sigma**2]])
    R = np.array([[0.1]])

    # Define model functions for PF
    def g(x, u):
        """Latent transition: x_k = alpha * x_{k-1} + w."""
        return np.array([alpha * x[0]])

    def h(x):
        """Observation model: y_k = beta * exp(0.5 * x_k)."""
        return np.array([beta * np.exp(0.5 * x[0])])

    # Initialize particle filter
    pf = ParticleFilter(
        g=g,
        h=h,
        Q=Q,
        R=R,
        Np=2000,
        resample_thresh=0.5,
        resample_method="systematic",
        regularize_after_resample=True,
        rng=np.random.default_rng(123)
    )

    # Initialize with true initial state 
    x0 = np.array([X_true[0]])
    P0 = np.array([[0.5]])
    pf.initialize(x0, P0)

    # Run filtering
    states = []
    for t in range(1, n):
        z_t = np.array([Y_obs[t]])
        state = pf.step(z_t)
        states.append(state)

    # Check all states were created
    assert len(states) == n - 1
    
    # Check final state properties
    final_state = states[-1]
    assert final_state.particles.shape == (2000, 1)
    assert final_state.weights.shape == (2000,)
    assert final_state.mean.shape == (1,)
    assert final_state.cov.shape == (1, 1)
    assert final_state.t == n - 1

    # Weights should sum to 1
    assert np.isclose(np.sum(final_state.weights), 1.0)


@pytest.mark.integration
def test_pf_sv_tracking_accuracy(sv_simulated_data, sv_params):
    """Test that PF tracks true states with reasonable accuracy."""
    X_true = sv_simulated_data.X
    Y_obs = sv_simulated_data.Y
    n = len(X_true)

    alpha = sv_params["alpha"]
    sigma = sv_params["sigma"]
    beta = sv_params["beta"]

    Q = np.array([[sigma**2]])
    R = np.array([[0.1]])

    def g(x, u):
        return np.array([alpha * x[0]])

    def h(x):
        return np.array([beta * np.exp(0.5 * x[0])])

    # Initialize PF
    pf = ParticleFilter(
        g=g, h=h, Q=Q, R=R,
        Np=3000,
        resample_thresh=0.5,
        resample_method="systematic",
        regularize_after_resample=True,
        rng=np.random.default_rng(123)
    )

    # Initialize with some uncertainty
    x0 = np.array([X_true[0]])
    P0 = np.array([[0.3]])
    pf.initialize(x0, P0)

    # Collect estimates
    x_estimates = [X_true[0]]
    for t in range(1, n):
        z_t = np.array([Y_obs[t]])
        state = pf.step(z_t)
        x_estimates.append(state.mean[0])

    x_estimates = np.array(x_estimates)

    # Compute RMSE
    errors = x_estimates - X_true
    rmse = np.sqrt(np.mean(errors ** 2))

    # RMSE should be reasonable
    print(f"PF RMSE: {rmse:.6f}")
    assert rmse < 1.5, f"RMSE too high: {rmse}"


@pytest.mark.integration
def test_pf_sv_effective_sample_size_tracking(sv_simulated_data, sv_params):
    """Test that effective sample size is tracked and resampling occurs."""
    X_true = sv_simulated_data.X
    Y_obs = sv_simulated_data.Y
    n = len(X_true)

    alpha = sv_params["alpha"]
    sigma = sv_params["sigma"]
    beta = sv_params["beta"]

    Q = np.array([[sigma**2]])
    R = np.array([[0.1]])

    def g(x, u):
        return np.array([alpha * x[0]])

    def h(x):
        return np.array([beta * np.exp(0.5 * x[0])])

    pf = ParticleFilter(
        g=g, h=h, Q=Q, R=R,
        Np=1000,
        resample_thresh=0.6,
        resample_method="systematic",
        rng=np.random.default_rng(123)
    )

    x0 = np.array([X_true[0]])
    P0 = np.array([[0.5]])
    pf.initialize(x0, P0)

    # Track Neff over time
    neff_history = []
    for t in range(1, min(50, n)):
        z_t = np.array([Y_obs[t]])
        pf.step(z_t)
        neff_history.append(pf.effective_sample_size())

    neff_history = np.array(neff_history)

    # Neff should stay reasonable (resampling should prevent collapse)
    assert np.all(neff_history > 10), "Neff collapsed despite resampling"
    
    # Neff should vary (indicating resampling happened)
    assert np.std(neff_history) > 1.0, "Neff appears constant (no resampling?)"


@pytest.mark.integration
def test_pf_sv_different_particle_counts(sv_simulated_data, sv_params):
    """Test PF performance with different numbers of particles."""
    X_true = sv_simulated_data.X
    Y_obs = sv_simulated_data.Y

    alpha = sv_params["alpha"]
    sigma = sv_params["sigma"]
    beta = sv_params["beta"]

    Q = np.array([[sigma**2]])
    R = np.array([[0.1]])

    def g(x, u):
        return np.array([alpha * x[0]])

    def h(x):
        return np.array([beta * np.exp(0.5 * x[0])])

    particle_counts = [100, 500, 2000]
    rmse_results = []

    for Np in particle_counts:
        pf = ParticleFilter(
            g=g, h=h, Q=Q, R=R,
            Np=Np,
            resample_thresh=0.5,
            resample_method="systematic",
            rng=np.random.default_rng(456)
        )

        x0 = np.array([X_true[0]])
        P0 = np.array([[0.5]])
        pf.initialize(x0, P0)

        # Run for first 100 steps
        x_estimates = [X_true[0]]
        for t in range(1, min(100, len(X_true))):
            z_t = np.array([Y_obs[t]])
            state = pf.step(z_t)
            x_estimates.append(state.mean[0])

        x_estimates = np.array(x_estimates)
        errors = x_estimates - X_true[:len(x_estimates)]
        rmse = np.sqrt(np.mean(errors ** 2))
        rmse_results.append(rmse)
        print(f"Np={Np}: RMSE={rmse:.6f}")

    # All should be finite and positive
    assert all(np.isfinite(r) and r > 0 for r in rmse_results)


@pytest.mark.integration
def test_pf_sv_systematic_vs_multinomial(sv_simulated_data, sv_params):
    """Test PF with systematic vs multinomial resampling."""
    X_true = sv_simulated_data.X
    Y_obs = sv_simulated_data.Y
    n = len(X_true)

    alpha = sv_params["alpha"]
    sigma = sv_params["sigma"]
    beta = sv_params["beta"]

    Q = np.array([[sigma**2]])
    R = np.array([[0.1]])

    def g(x, u):
        return np.array([alpha * x[0]])

    def h(x):
        return np.array([beta * np.exp(0.5 * x[0])])

    # Systematic resampling
    pf_sys = ParticleFilter(
        g=g, h=h, Q=Q, R=R,
        Np=2000,
        resample_thresh=0.5,
        resample_method="systematic",
        rng=np.random.default_rng(789)
    )

    # Multinomial resampling
    pf_multi = ParticleFilter(
        g=g, h=h, Q=Q, R=R,
        Np=2000,
        resample_thresh=0.5,
        resample_method="multinomial",
        rng=np.random.default_rng(789)
    )

    # Initialize both
    x0 = np.array([X_true[0]])
    P0 = np.array([[0.5]])
    pf_sys.initialize(x0, P0)
    pf_multi.initialize(x0, P0)

    # Run both for 50 steps
    for t in range(1, min(50, n)):
        z_t = np.array([Y_obs[t]])
        pf_sys.step(z_t)
        pf_multi.step(z_t)

    # Both should complete successfully
    assert pf_sys.state is not None
    assert pf_multi.state is not None
    
    # Both should have valid states
    assert np.isfinite(pf_sys.state.mean).all()
    assert np.isfinite(pf_multi.state.mean).all()


@pytest.mark.integration
def test_pf_sv_convergence_from_poor_init(sv_simulated_data, sv_params):
    """Test PF can recover from poor initialization."""
    X_true = sv_simulated_data.X
    Y_obs = sv_simulated_data.Y
    n = len(X_true)

    alpha = sv_params["alpha"]
    sigma = sv_params["sigma"]
    beta = sv_params["beta"]

    Q = np.array([[sigma**2]])
    R = np.array([[0.1]])

    def g(x, u):
        return np.array([alpha * x[0]])

    def h(x):
        return np.array([beta * np.exp(0.5 * x[0])])

    pf = ParticleFilter(
        g=g, h=h, Q=Q, R=R,
        Np=3000,
        resample_thresh=0.5,
        resample_method="systematic",
        regularize_after_resample=True,
        rng=np.random.default_rng(999)
    )

    # Initialize far from true state
    x0_bad = np.array([X_true[0] + 2.0])
    P0 = np.array([[1.0]])
    pf.initialize(x0_bad, P0)

    # Run filtering
    x_estimates = [x0_bad[0]]
    for t in range(1, min(100, n)):
        z_t = np.array([Y_obs[t]])
        state = pf.step(z_t)
        x_estimates.append(state.mean[0])

    x_estimates = np.array(x_estimates)

    # Compute error at start and end
    error_start = np.abs(x_estimates[0] - X_true[0])
    error_end = np.abs(x_estimates[-1] - X_true[len(x_estimates) - 1])

    # Error should decrease (filter should converge toward truth)
    assert error_end < error_start, "Filter did not converge from poor initialization"


@pytest.mark.integration
def test_pf_sv_covariance_properties(sv_simulated_data, sv_params):
    """Test that posterior covariance has correct properties."""
    X_true = sv_simulated_data.X
    Y_obs = sv_simulated_data.Y

    alpha = sv_params["alpha"]
    sigma = sv_params["sigma"]
    beta = sv_params["beta"]

    Q = np.array([[sigma**2]])
    R = np.array([[0.1]])

    def g(x, u):
        return np.array([alpha * x[0]])

    def h(x):
        return np.array([beta * np.exp(0.5 * x[0])])

    pf = ParticleFilter(
        g=g, h=h, Q=Q, R=R,
        Np=2000,
        resample_thresh=0.5,
        resample_method="systematic",
        rng=np.random.default_rng(333)
    )

    x0 = np.array([X_true[0]])
    P0 = np.array([[0.5]])
    pf.initialize(x0, P0)

    # Run several steps
    for t in range(1, 20):
        z_t = np.array([Y_obs[t]])
        pf.step(z_t)

    cov = pf.state.cov

    # Covariance should be symmetric
    np.testing.assert_allclose(cov, cov.T, rtol=1e-10)

    # Covariance should be positive semi-definite
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-10), "Covariance has negative eigenvalues"

    # Diagonal elements should be positive
    assert np.all(np.diag(cov) > 0), "Covariance diagonal not positive"
