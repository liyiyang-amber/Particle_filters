import numpy as np
import pytest
from models.kalman_filter import kalman_filter_general
def test_timeinvariant_equals_timevarying(small_system, Y_synthetic):
    s = small_system
    N = Y_synthetic.shape[0]
    res_a = kalman_filter_general(Y=Y_synthetic, Phi=s["Phi"], H=s["H"], Gamma=s["Gamma"],
                                  Q=s["Q"], R=s["R"], x0=s["x0"], P0=s["P0"])
    res_b = kalman_filter_general(Y=Y_synthetic,
                                  Phi=[s["Phi"]]*N, H=[s["H"]]*N, Gamma=[s["Gamma"]]*N,
                                  Q=[s["Q"]]*N, R=[s["R"]]*N, x0=s["x0"], P0=s["P0"])
    for a, b in [(res_a.x_pred, res_b.x_pred), (res_a.x_filt, res_b.x_filt),
                 (res_a.P_pred, res_b.P_pred), (res_a.P_filt, res_b.P_filt),
                 (res_a.K, res_b.K), (res_a.innov, res_b.innov), (res_a.S, res_b.S)]:
        assert np.allclose(a, b)
