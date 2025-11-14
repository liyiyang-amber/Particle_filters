import numpy as np
import pytest
from simulator.simulator_LGSSM import simulate_lgssm
import pytest

def test_burnin_changes_trace_not_shape(small_matrices, small_N, rng_seed):
    A,B,C,D,Sigma = small_matrices["A"], small_matrices["B"], small_matrices["C"], small_matrices["D"], small_matrices["Sigma"]
    r0 = simulate_lgssm(A,B,C,D,Sigma,N=small_N, seed=rng_seed, burn_in=0)
    r1 = simulate_lgssm(A,B,C,D,Sigma,N=small_N, seed=rng_seed, burn_in=25)
    assert r0.X.shape == r1.X.shape
    assert r0.Y.shape == r1.Y.shape
    assert not np.allclose(r0.X[:3], r1.X[:3]) or not np.allclose(r0.Y[:3], r1.Y[:3])

@pytest.mark.slow
def test_initial_state_distribution_mean_cov(small_matrices, rng_seed):
    # Statistical sanity: x1 ~ N(0, Sigma)
    A,B,C,D,Sigma = small_matrices["A"], small_matrices["B"], small_matrices["C"], small_matrices["D"], small_matrices["Sigma"]
    M = 5000  # keep moderate for CI; increase locally if needed
    xs = []
    for i in range(M):
        res = simulate_lgssm(A,B,C,D,Sigma,N=1, seed=rng_seed + i, burn_in=0)
        xs.append(res.X[0])
    X1 = np.vstack(xs)
    mu = X1.mean(axis=0)
    cov = np.cov(X1.T, bias=False)
    # mean close to 0 within 4*SE
    se = np.sqrt(np.diag(Sigma) / M)
    assert np.linalg.norm(mu) <= 4 * np.linalg.norm(se)
    # covariance close to Sigma (relative Frobenius error)
    relerr = np.linalg.norm(cov - Sigma, 'fro') / max(1.0, np.linalg.norm(Sigma, 'fro'))
    assert relerr < 0.10  # conservative for M=5000
