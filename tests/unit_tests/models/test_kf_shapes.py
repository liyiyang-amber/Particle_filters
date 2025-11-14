import numpy as np
import pytest
from models.kalman_filter import kalman_filter_general  
def test_shapes_and_symmetry(small_system, Y_synthetic):
    s = small_system
    res = kalman_filter_general(
        Y=Y_synthetic, Phi=s["Phi"], H=s["H"], Gamma=s["Gamma"],
        Q=s["Q"], R=s["R"], x0=s["x0"], P0=s["P0"], use_joseph=False
    )
    N, nx, ny = Y_synthetic.shape[0], s["nx"], s["ny"]
    assert res.x_pred.shape == (N, nx)
    assert res.P_pred.shape == (N, nx, nx)
    assert res.x_filt.shape == (N, nx)
    assert res.P_filt.shape == (N, nx, nx)
    assert res.K.shape == (N, nx, ny)
    assert res.innov.shape == (N, ny)
    assert res.S.shape == (N, ny, ny)
    # symmetry
    for k in range(N):
        assert np.allclose(res.P_pred[k], res.P_pred[k].T, atol=1e-10)
        assert np.allclose(res.P_filt[k], res.P_filt[k].T, atol=1e-10)
