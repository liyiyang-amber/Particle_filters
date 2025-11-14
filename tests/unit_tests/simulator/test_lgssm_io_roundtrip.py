import numpy as np
import os
from simulator.simulator_LGSSM import simulate_lgssm, LGSSMSimulationResult

def test_npz_roundtrip(tmp_path, small_matrices):
    A,B,C,D,Sigma = small_matrices["A"], small_matrices["B"], small_matrices["C"], small_matrices["D"], small_matrices["Sigma"]
    res = simulate_lgssm(A,B,C,D,Sigma,N=200, seed=7, burn_in=10)
    target = tmp_path / "simdata"
    res.to_file(str(target), format="npz", overwrite=True)
    data = np.load(str(target) + ".npz")
    assert np.allclose(data["X"], res.X)
    assert np.allclose(data["Y"], res.Y)

def test_overwrite_protection(tmp_path, small_matrices):
    A,B,C,D,Sigma = small_matrices["A"], small_matrices["B"], small_matrices["C"], small_matrices["D"], small_matrices["Sigma"]
    res = simulate_lgssm(A,B,C,D,Sigma,N=10, seed=1, burn_in=0)
    target = tmp_path / "simdata"
    res.to_file(str(target), format="npz", overwrite=True)
    # second save without overwrite should raise
    import pytest
    with pytest.raises(FileExistsError):
        res.to_file(str(target), format="npz", overwrite=False)
