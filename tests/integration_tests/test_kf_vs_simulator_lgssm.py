import numpy as np
import pytest
from models.kalman_filter import kalman_filter_general

@pytest.mark.integration
def test_kf_against_simulated_lgssm(tmp_path):
    # Load your saved sim; adapt path if needed
    data = np.load("simulator/data/lgssm_simul_data.npz")
    X = data["X"]; Y = data["Y"]

    # system matrices (use the ones from the data file to ensure consistency)
    A = data["A"]
    B = data["B"]
    C = data["C"]
    D = data["D"]
    Sigma = np.eye(2)
    Q = B @ B.T
    R = D @ D.T
    Gamma = np.eye(2)

    res = kalman_filter_general(Y=Y, Phi=A, H=C, Gamma=Gamma, Q=Q, R=R,
                                x0=np.zeros(2), P0=Sigma, use_joseph=False)

    # 1) Innovation covariance match
    emp_S = np.cov(res.innov.T, bias=False)
    mean_S = res.S.mean(axis=0)
    relerr_S = np.linalg.norm(emp_S - mean_S, 'fro') / max(1.0, np.linalg.norm(mean_S, 'fro'))
    assert relerr_S < 0.08

    # 2) Empirical state error covariance vs mean P_filt
    err = X - res.x_filt
    emp_P = np.cov(err.T, bias=False)
    mean_P = res.P_filt.mean(axis=0)
    relerr_P = np.linalg.norm(emp_P - mean_P, 'fro') / max(1.0, np.linalg.norm(mean_P, 'fro'))
    assert relerr_P < 0.10

    # 3) RMSE improvement: filtered vs prior in observation space
    prior_obs = (C @ res.x_pred.T).T
    filt_obs  = (C @ res.x_filt.T).T
    rmse_prior = np.sqrt(np.mean((Y - prior_obs)**2))
    rmse_filt  = np.sqrt(np.mean((Y - filt_obs )**2))
    assert rmse_filt < rmse_prior
