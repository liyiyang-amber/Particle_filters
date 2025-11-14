import numpy as np
import pytest
from models.kalman_filter import kalman_filter_general

def test_zero_control_equivalence(small_system, Y_synthetic):
    s = small_system; N = Y_synthetic.shape[0]; nx = s["nx"]
    B_seq = [np.zeros((nx, 1))]*N
    U = np.zeros((N, 1))
    res1 = kalman_filter_general(Y=Y_synthetic, Phi=s["Phi"], H=s["H"], Gamma=s["Gamma"],
                                 Q=s["Q"], R=s["R"], B=None, U=None, x0=s["x0"], P0=s["P0"])
    res2 = kalman_filter_general(Y=Y_synthetic, Phi=s["Phi"], H=s["H"], Gamma=s["Gamma"],
                                 Q=s["Q"], R=s["R"], B=B_seq, U=U, x0=s["x0"], P0=s["P0"])
    assert np.allclose(res1.x_pred, res2.x_pred)
    assert np.allclose(res1.x_filt, res2.x_filt)

def test_bad_lengths_raise(small_system, Y_synthetic):
    s = small_system; N = Y_synthetic.shape[0]
    with pytest.raises(ValueError):
        kalman_filter_general(Y=Y_synthetic, Phi=[s["Phi"]]*(N-1), H=s["H"],
                              Gamma=s["Gamma"], Q=s["Q"], R=s["R"],
                              x0=s["x0"], P0=s["P0"])
