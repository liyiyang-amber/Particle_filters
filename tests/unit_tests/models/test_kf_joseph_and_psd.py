import numpy as np
import pytest
from models.kalman_filter import kalman_filter_general

def test_joseph_preserves_psd(small_system, Y_synthetic):
    s = small_system
    # make R tiny to stress numerics
    R_tiny = s["R"] * 1e-8
    res_std = kalman_filter_general(Y=Y_synthetic, Phi=s["Phi"], H=s["H"], Gamma=s["Gamma"],
                                    Q=s["Q"], R=R_tiny, x0=s["x0"], P0=s["P0"], use_joseph=False)
    res_jos = kalman_filter_general(Y=Y_synthetic, Phi=s["Phi"], H=s["H"], Gamma=s["Gamma"],
                                    Q=s["Q"], R=R_tiny, x0=s["x0"], P0=s["P0"], use_joseph=True)
    # Joseph P_filt should be PSD
    mineig = min(np.linalg.eigvalsh(P).min() for P in res_jos.P_filt)
    assert mineig >= -1e-12
