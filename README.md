# From Classical Filters to Particle Flows

## 1. Abstract

This repository provides a unified codebase illustrating the evolution of filtering methods: from classical Kalman filters (KF, EKF, UKF), to particle filters (PF), deterministic particle flows (EDH, LEDH), and kernel-embedded flows (KPF). The motivation is to address the limitations of classical filters under strong nonlinearity and non-Gaussianity, the degeneracy of particle filters, and to showcase how particle flows provide continuous-time Bayes updates. The codebase includes implementations, benchmarks, visualizations, and stability diagnostics for a wide range of state-space models (SSMs).


## 2. Repository Structure

```
/models/         # Filtering algorithms (KF, EKF, UKF, PF, EDH, LEDH, KPF)
/simulator/      # Synthetic data generators for SSMs and experiment data
/notebooks/      # Interactive demos and experiments
/tests/          # Unit and integration tests
```

**File Reference Guide:**
For further details, please refer to the 'Report'.


## 3. Installation & Dependencies

- Python 3.11.13 
- numpy==2.3.3
- scipy==1.16.2
- matplotlib==3.10.7
- pandas==2.3.3
- pytest==8.4.2


## 4. Linear–Gaussian SSM & Kalman Filter

### 4.1 Model Description

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

### 5.1 Model Construction

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

### 5.2 EKF & UKF Implementation
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
- The flow is defined by an ODE: $\frac{dx}{d\lambda} = K(x, \lambda) \nabla_x \log p(y|x),$
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


## 8. Stability Diagnostics

Comprehensive stability diagnostics and comparison experiments are performed for EDH, LEDH, and KPF filters in the notebook:

- **Metrics Tracked:**
    - Particle displacement ($\|\Delta \eta\|$)
    - Condition number of flow matrices (EDH/LEDH)
    - Pseudo-time step sizes ($\theta$) for KPF
    - Weight coefficient of variation (CV)
    - Minimum effective sample size (ESS)
    - Number of resampling events
    - RMSE and runtime


## 9. Reproducible Experiments

- Run experiment scripts in `/notebooks/`
- Control randomness with seeds

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
