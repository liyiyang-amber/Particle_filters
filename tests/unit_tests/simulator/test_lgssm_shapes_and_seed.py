import numpy as np
import pytest
from simulator.simulator_LGSSM import simulate_lgssm  

def test_shapes(small_matrices, small_N, rng_seed):
    A,B,C,D,Sigma = small_matrices["A"], small_matrices["B"], small_matrices["C"], small_matrices["D"], small_matrices["Sigma"]
    res = simulate_lgssm(A,B,C,D,Sigma,N=small_N, seed=rng_seed, burn_in=0)
    assert res.X.shape == (small_N, small_matrices["nx"])
    assert res.Y.shape == (small_N, small_matrices["ny"])
    assert res.X.dtype == float and res.Y.dtype == float

def test_reproducibility(small_matrices, small_N, rng_seed):
    A,B,C,D,Sigma = small_matrices["A"], small_matrices["B"], small_matrices["C"], small_matrices["D"], small_matrices["Sigma"]
    r1 = simulate_lgssm(A,B,C,D,Sigma,N=small_N, seed=rng_seed, burn_in=5)
    r2 = simulate_lgssm(A,B,C,D,Sigma,N=small_N, seed=rng_seed, burn_in=5)
    assert np.allclose(r1.X, r2.X)
    assert np.allclose(r1.Y, r2.Y)

def test_different_seeds_differ(small_matrices, small_N):
    A,B,C,D,Sigma = small_matrices["A"], small_matrices["B"], small_matrices["C"], small_matrices["D"], small_matrices["Sigma"]
    r1 = simulate_lgssm(A,B,C,D,Sigma,N=small_N, seed=1, burn_in=5)
    r2 = simulate_lgssm(A,B,C,D,Sigma,N=small_N, seed=2, burn_in=5)
    diff = np.linalg.norm(r1.X - r2.X) + np.linalg.norm(r1.Y - r2.Y)
    assert diff > 1e-8
