import numpy as np
import pytest
from models.kalman_filter import kalman_filter_general

@pytest.mark.integration
def test_timevarying_parameters_end2end(N=200):
    nx, ny = 2, 1
    # slowly drifting Φ_k
    Phis = []
    for k in range(N):
        a = 0.85 + 0.1*np.sin(2*np.pi*k/N)
        Phis.append(np.array([[a, 0.1],[0.0, 0.7]]))
    H = np.array([[1.0, 0.0]])
    Gammas = [np.eye(nx)]*N
    Qs = [np.diag([0.05, 0.02])]*N
    R = np.array([[0.10]])
    Rs = [R]*N
    Y = np.zeros((N, ny))  # synthetic zeros just to test path
    res = kalman_filter_general(Y=Y, Phi=Phis, H=[H]*N, Gamma=Gammas,
                                Q=Qs, R=Rs, x0=np.zeros(nx), P0=np.eye(nx))
    assert np.isfinite(res.loglik)
    assert res.x_filt.shape == (N, nx)
