# From Classical Filters to Particle Flows

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

pytest
```

Recommended starting notebook: `notebooks/EKF_UKF_PF_comparison.ipynb`.

## 1. Abstract

This repository provides a unified codebase illustrating the evolution of filtering methods: from classical Kalman filters (KF, EKF, UKF), to particle filters (PF), and further to particle-flow-based methods including deterministic flows (EDH/LEDH), kernel-embedded flows (KPF), and stochastic particle flow (SPF) variants. We also include differentiable particle filter (DPF) resampling strategies (soft, optimal transport, and learned/RNN-based). The motivation is to address the limitations of classical filters under strong nonlinearity and non-Gaussianity, the degeneracy of particle filters, and to showcase how particle flows provide continuous-time Bayes updates. The codebase includes implementations, benchmarks, visualizations, and stability diagnostics for a wide range of state-space models (SSMs).

The repository provides:
- Implementations in `models/` and simulators in `simulator/`
- Reproducible experiments and visualizations in `notebooks/`


## 2. Repository Structure

```
/models/                 # Filter implementations (KF,EKF,UKF,PF,EDH,LEDH,KPF,SPF,DPF)
/simulator/              # Synthetic data generators for SSMs and experiment data
/notebooks/              # Reproducible experiments, comparisons, and figure generation
/tests/                  # Unit + integration tests
README.md                # Project overview (this file)
```

## Where to start (recommended notebooks)

| Goal | Notebook |
|---|---|
| Kalman filter check (LGSSM) | `notebooks/kalman_filter_LGSSM.ipynb` |
| EKF vs UKF vs PF (core comparison) | `notebooks/EKF_UKF_PF_comparison.ipynb` |
| EDH/LEDH/KPF on nonlinear SSMs | `notebooks/EDH_LEDH_KPF_NLNGSSM.ipynb` |
| Differentiable PF (DPF) resampling comparisons | `notebooks/DPF_resampling_comparison_nonlinear.ipynb` |

## Common issues

- **TensorFlow install:** DPF modules require TensorFlow (pinned in `requirements.txt`). If installation fails, confirm you’re using the intended Python version and an up-to-date pip. On some systems, TensorFlow wheels can be platform-specific.
- **Notebooks can’t import `models/` or `simulator/`:** Ensure the notebook kernel is the same environment where you installed `requirements.txt`, and open/run notebooks from the repo root so imports resolve.

## 3. Installation & Dependencies

### 3.1 Python

- Tested with Python 3.11.13 (should work on Python 3.10+).

### 3.2 Install

This repo uses a simple pip-based setup via `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3.3 Quick sanity check

Run the test suite:

```bash
pytest
```

### 3.4 Main dependencies

Pinned versions live in `requirements.txt`. Key packages include NumPy/SciPy, Matplotlib, Pandas, and PyTest.


## 4. Linear–Gaussian SSM & Kalman Filter

### 4.1 Model 

The Linear Gaussian State Space Model (LGSSM) is defined as:

$$
\begin{align*}
    x_{k+1} &= A x_k + B v_k \\
    y_k &= C x_k + D w_k
\end{align*}
$$

where:
- $x_k$ is the latent state vector at time $k$
- $y_k$ is the observed output
- $A$ is the state transition matrix
- $B$ is the process noise input matrix
- $C$ is the observation matrix
- $D$ is the measurement noise input matrix
- $v_k \sim \mathcal{N}(0, I)$ is process noise
- $w_k \sim \mathcal{N}(0, I)$ is measurement noise

**Parameters for synthetic data generation:**
- State dimension: 2
- Observation dimension: 2
- A = $`\begin{bmatrix} 0.9 & 0.5 \\ 0.0 & 0.7 \end{bmatrix}`$
- $B = \mathrm{diag}(\sqrt{0.05}, \sqrt{0.02})$
- C = $`\begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}`$
- $D = \mathrm{diag}(\sqrt{0.10}, \sqrt{0.10})$
- Initial state covariance: $\Sigma = I$
- Time steps: 1000
- Random seed: 42

### 4.2 Experiments
- Run Kalman filter in both Standard and Joseph forms
- Compute and compare:
    - Mean NEES (Normalized Estimation Error Squared)
    - Condition numbers of filtered covariance
    - Minimum eigenvalues (PSD validation)
    - RMSE between true and filtered states
- Visualize:
    - True vs filtered states with 95% confidence intervals
    - Observations vs filtered estimates (first 200 steps)
    - Monte Carlo robustness analysis (multiple seeds)
    - Boundary tests for numerical stability (extreme noise, unstable dynamics, partial observability)


## 5. Nonlinear / Non-Gaussian SSM

### 5.1 Model 

A key example is the 1-D stochastic volatility (SV) model:

$$
\begin{align*}
    X_1 &\sim \mathcal{N}\left(0, \frac{\sigma^2}{1 - \alpha^2}\right) \\
    X_t &= \alpha X_{t-1} + \sigma V_t, \quad V_t \sim \mathcal{N}(0, 1) \\
    Y_t &= \beta \exp\left(0.5 X_t\right) W_t, \quad W_t \sim \mathcal{N}(0, 1)
\end{align*}
$$

where:
- $X_t$ is the latent log-volatility state
- $Y_t$ is the observed return (or measurement)
- $\alpha$ is the AR(1) coefficient ($|\alpha| < 1$ for stationarity)
- $\sigma$ is the process noise standard deviation
- $\beta$ is the observation scale
- $V_t, W_t$ are independent standard normal noises

**Parameters for synthetic data generation:**
- Typical: $\alpha = 0.95$, $\sigma = 0.2$, $\beta = 1.0$, $n = 1000$, seed = 42
- Initial state $X_1$ drawn from stationary distribution if not specified

This model is nonlinear and non-Gaussian in the observation equation, making it a strong test for EKF, UKF, and PF methods.

### 5.2 Methods
- EKF: Linearizes the nonlinear functions around current estimate
- UKF: Uses sigma points to approximate nonlinear transformations
- Both methods may struggle with the non-Gaussian, heavy-tailed observation noise

### 5.3 Particle Filter Implementation
- Sequential Importance Resampling (SIR) algorithm
- Effective sample size (ESS) monitoring
- Visualization of particle degeneracy and weight collapse

### 5.4 Experiments
- Run EKF, UKF, and PF on the same data
- Compare:
    - RMSE, MAE, NLL, ESS trajectory, coverage
    - Runtime (CPU), memory usage
- Visualize:
    - True vs estimated trajectories
    - Residual distributions
    - ESS collapse plots
    - Particle weight histograms
- Stress-test filters in high-volatility and heavy-tailed regimes


## 6. Deterministic Particle Flows

### 6.1 EDH Flow
- The Exact Daum-Huang (EDH) flow implements a deterministic continuous-time Bayes update for particles.
- The flow is defined by an ODE:
  $\frac{dx}{d\lambda} = K(x, \lambda) \nabla_x \log p(y|x)$
  where $K(x, \lambda)$ is a gain matrix, typically from linearization.
- In EDH, the gain is computed globally from the linearized model at the ensemble mean.
- Jacobian determinant and condition number are monitored for stability.

### 6.2 LEDH Flow
- The Local EDH (LEDH) flow computes the gain matrix separately for each particle, using local linearization.
- This improves accuracy in highly nonlinear regimes but increases computational cost.
- Trade-offs between runtime and stability are explored in experiments.

### 6.3 Invertible PF-PF Framework
- Combines deterministic flow with importance weighting to maintain invertibility and correct the target distribution.
- Log-Jacobian and invertibility diagnostics are included.
- Visualizations: flow field, Jacobian conditioning, flow magnitude heatmap.

### 6.4 Experiments
- Simulate nonlinear/non-Gaussian SSMs and apply EDH, LEDH, and PF-PF methods.
- Compare:
    - RMSE, ESS, flow stability, runtime, memory usage
    - Jacobian condition number and log-determinant
- Visualize:
    - Particle trajectories under flow
    - Flow field and magnitude
    - Jacobian spectrum and conditioning
    - Marginal distributions before and after flow


## 7. Kernel-Embedded Particle Flow Filter (KPF)

### 7.1 Motivation
- KPF moves particles in a reproducing kernel Hilbert space (RKHS), enabling nonparametric updates.
- Matrix-valued kernels are used to prevent marginal collapse in high dimensions.

### 7.2 Scalar Kernel Implementation
- Uses Gaussian RBF kernels for particle interactions.
- Limitations: scalar kernels may collapse marginals in high-dimensional settings.

### 7.3 Matrix-Valued Kernel Implementation
- Diagonal or block-diagonal kernels encode conditional independence and stabilize updates.
- Experiments show improved stability and coverage in high dimensions.

### 7.4 Experiments
- Apply KPF (scalar and matrix-valued) to nonlinear/non-Gaussian SSMs.
- Compare:
    - RMSE, ESS, flow stability, time & memory
    - Collapse of marginal distributions in high-d
- Visualize:
    - Kernel Gram matrix conditioning
    - Marginal distributions of observed variables
    - Particle movement in RKHS


## 8. Stochastic Particle Flow (SPF)

## 8.1 Motivation
SPF performs the measurement update by tempering the likelihood over pseudo-time $\lambda\in[0,1]$, and adds diffusion to reduce particle collapse.

Tempered posterior:
$\pi_{\lambda}(x) \propto p(y\mid x)^{\beta(\lambda)}p(x),\quad \beta(0)=0,\quad \beta(1)=1.$
Schedules used in the repo: linear $\beta(\lambda)=\lambda$ and a numerically-solved “optimal” $\beta^*(\lambda)$ (bisection/shooting).

Local linear–Gaussian model used in the SPF:
$y = Hx + v,\quad v\sim\mathcal{N}(0, R),\quad x\sim\mathcal{N}(m_0, P_0).$

Stochastic flow (integrated by Euler–Maruyama):
$dX_{\lambda} = a(X_{\lambda},\lambda)d\lambda + B(X_{\lambda},\lambda)dW_{\lambda}.$

### 8.2 Experiments and outputs
Reproductions compare SPF (optimal $\beta^*$) vs SPF (linear $\beta$) vs SIR PF.



## 9. Differentiable Particle Filters (DPF) and Resampling Variants

The common goal is to map a weighted particle set $`\{(x^{(i)}, w^{(i)})\}_{i=1}^{N}`$ to an approximately unweighted set $`\{\tilde{x}^{(j)}\}_{j=1}^{N}`$ smoothly (and in some cases differentiably), improving stability in nonlinear/non-Gaussian settings.

### 9.1 Soft resampling (Gumbel–Softmax mixture)

Soft resampling uses a mixture distribution
$q_i = (1-\alpha) w_i + \alpha\,\frac{1}{N},\quad \alpha\in[0,1],$
and draws (soft) ancestor assignments using a Gumbel–Softmax reparameterization. This yields a differentiable approximation to categorical resampling and helps avoid hard particle impoverishment.

### 9.2 Optimal-transport (OT) resampling (Sinkhorn + barycentric projection)

OT resampling computes an entropic-regularized transport plan $P\in\mathbb{R}^{N\times N}$ between the weighted empirical measure and a uniform target:
$a=w,\quad b=\tfrac{1}{N}\mathbf{1},\quad C_{ij}=\|x^{(i)}-x^{(j)}\|^2.$
Using Sinkhorn iterations, it forms $P$ (approximately satisfying $P\mathbf{1}=a$ and $P^T\mathbf{1}=b$), then applies the **barycentric projection**:
$\tilde{x}^{(j)} = \frac{1}{b_j}\sum_i P_{ij} x^{(i)};\text{(so for }b_j=1/N:\; \tilde{x}^{(j)}=N\sum_i P_{ij}x^{(i)}\text{)}.$
This produces smoothly moved particles with uniform weights and typically reduces Monte Carlo noise relative to discrete resampling.

### 9.3 Learned / RNN-based resampling

An RNN (LSTM/GRU) can be used to output assignment probabilities over old particles given particle states and weights, enabling data-driven resampling policies. The implementation supports a baseline mode (no training) and a learned mode.


## 10. References

- Doucet, A., & Johansen, A. (2009). A Tutorial on Particle Filtering and Smoothing: Fifteen Years Later. *Handbook of Nonlinear Filtering*, 12.
- Li, Q., Li, R., Ji, K., & Dai, W. (2015). Kalman Filter and Its Application. *2015 8th International Conference on Intelligent Networks and Intelligent Systems (ICINIS)*, 74-77. https://doi.org/10.1109/ICINIS.2015.35
- Humpherys, J., Redd, P., & West, J. (2012). A Fresh Look at the Kalman Filter. *SIAM Review*, 54(4), 801-823. https://doi.org/10.1137/100799666
- Thrun, S. (2005). Extended Kalman Filter Lecture Notes. Carnegie Mellon University. [PDF](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/kalman/ekf_lecture_notes.pdf)
- Wan, E. A., & van der Merwe, R. (2000). The Unscented Kalman Filter for Nonlinear Estimation. *IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium (AS-SPCC)*, 153-158.
- Daum, F., Huang, J., & Noushin, A. (2010). Exact particle flow for nonlinear filters. *Signal Processing, Sensor Fusion, and Target Recognition XIX*, SPIE, 769704. https://doi.org/10.1117/12.839590
- Daum, F., & Huang, J. (2011). Particle degeneracy: root cause and solution. *Signal Processing, Sensor Fusion, and Target Recognition XX*, SPIE, 80500W. https://doi.org/10.1117/12.877167
- Li, Y., & Coates, M. (2017). Particle Filtering With Invertible Particle Flow. *IEEE Transactions on Signal Processing*, 65(15), 4102-4116. https://doi.org/10.1109/TSP.2017.2703684
- Dhayalkar, S. R. (2025). Particle Filter Made Simple: A Step-by-Step Beginner-friendly Guide. arXiv:2511.01281. https://arxiv.org/abs/2511.01281
- Hu, C.-C., & van Leeuwen, P. J. (2021). A particle flow filter for high-dimensional system applications. *Quarterly Journal of the Royal Meteorological Society*, 147(737), 2352-2374. https://doi.org/10.1002/qj.4028
- Dai, L., & Daum, F. (2021). A New Parameterized Family of Stochastic Particle Flow Filters. arXiv:2103.09676. https://arxiv.org/abs/2103.09676
- Dai, L., & Daum, F. E. (2021). Stiffness Mitigation in Stochastic Particle Flow Filters. arXiv:2107.04672. https://arxiv.org/abs/2107.04672
- Corenflos, A., Thornton, J., Deligiannidis, G., & Doucet, A. (2021). Differentiable Particle Filtering via Entropy-Regularized Optimal Transport. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139, 2100–2111. https://proceedings.mlr.press/v139/corenflos21a.html
- Chen, X., & Li, Y. (2025). An overview of differentiable particle filters for data-adaptive sequential Bayesian inference. *Foundations of Data Science*, 7(4), 915–943. https://doi.org/10.3934/fods.2023014
- Jonschkowski, R., Rastogi, D., & Brock, O. (2018). Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors. arXiv:1805.11122. https://arxiv.org/abs/1805.11122
- Ma, X., Karkus, P., & Hsu, D. (2020). Particle Filter Recurrent Neural Networks. *AAAI Conference on Artificial Intelligence*, 34(04), 5101–5108. https://doi.org/10.1609/aaai.v34i04.5952





