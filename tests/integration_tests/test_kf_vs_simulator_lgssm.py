import numpy as np
import pytest
from models.kalman_filter import kalman_filter_general

@pytest.mark.integration
def test_kf_against_simulated_lgssm(tmp_path):
    rng = np.random.default_rng(123)
    T = 400

    # Stable 2-D LGSSM with moderate process / observation noise.
    A = np.array([[0.85, 0.10], [0.0, 0.75]])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[0.20, 0.00], [0.00, 0.15]])
    D = np.array([[0.10, 0.00], [0.00, 0.12]])
    Sigma = np.eye(2)
    Q = B @ B.T
    R = D @ D.T
    Gamma = np.eye(2)

    X = np.zeros((T, 2))
    Y = np.zeros((T, 2))
    x = rng.multivariate_normal(np.zeros(2), Sigma)
    for t in range(T):
        x = A @ x + B @ rng.standard_normal(2)
        y = C @ x + D @ rng.standard_normal(2)
        X[t] = x
        Y[t] = y

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
