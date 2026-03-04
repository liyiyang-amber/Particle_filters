"""
Hamiltonian Monte Carlo (HMC) for parameter inference.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable, Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class HMCConfig:
    """Configuration for HMC sampler."""
    n_samples: int = 1000
    """Number of samples to draw (after burnin)"""
    
    n_burnin: int = 500
    """Number of burnin samples to discard"""
    
    step_size: float = 0.01
    """HMC step size (epsilon)"""
    
    n_leapfrog_steps: int = 10
    """Number of leapfrog steps per HMC iteration"""
    
    adapt_step_size: bool = True
    """Whether to adapt step size during burnin"""
    
    target_accept_prob: float = 0.65
    """Target acceptance probability for step size adaptation"""
    
    verbose: bool = True
    """Whether to print progress"""


class HMCSampler:
    """
    Hamiltonian Monte Carlo sampler for parameter inference.
    """
    
    def __init__(
        self,
        log_posterior_fn: Callable[[tf.Tensor], tf.Tensor],
        config: Optional[HMCConfig] = None,
    ):
        """
        Parameters
        ----------
        log_posterior_fn : callable
            TensorFlow function that computes log p(θ | Y) given parameters θ.
            Must be differentiable (e.g., using tf.function and tf.GradientTape).
        config : HMCConfig, optional
            Configuration for the sampler.
        """
        self.log_posterior_fn = log_posterior_fn
        self.config = config or HMCConfig()
        
    def run(self, initial_params: np.ndarray) -> Dict:
        """
        Run HMC sampler using TensorFlow Probability.
        
        Parameters
        ----------
        initial_params : ndarray, shape (n_params,)
            Initial parameter values (in constrained space if needed).
            
        Returns
        -------
        results : dict
            Dictionary containing:
                - samples: ndarray of shape (n_samples, n_params)
                - is_accepted: ndarray of shape (n_samples,)
                - accept_rate: float
                - runtime: float
        """
        config = self.config
        
        # Convert to TensorFlow tensor
        initial_state = tf.constant(initial_params, dtype=tf.float32)
        
        if config.verbose:
            print(f"\n{'='*80}")
            print("Hamiltonian Monte Carlo (HMC)")
            print('='*80)
            print(f"Initial parameters: {initial_params}")
            print(f"HMC settings:")
            print(f"  Samples: {config.n_samples} (+ {config.n_burnin} burnin)")
            print(f"  Step size: {config.step_size}")
            print(f"  Leapfrog steps: {config.n_leapfrog_steps}")
            print(f"  Adaptive: {config.adapt_step_size}")
            print()
        
        start_time = time.time()
        
        # Setup HMC kernel
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.log_posterior_fn,
            step_size=config.step_size,
            num_leapfrog_steps=config.n_leapfrog_steps,
        )
        
        # Adaptive step size during burnin
        if config.adapt_step_size:
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=kernel,
                num_adaptation_steps=int(0.8 * config.n_burnin),
                target_accept_prob=config.target_accept_prob,
            )
        
        if config.verbose:
            print("Running HMC sampler...")
        
        # Run MCMC
        def run_chain():
            return tfp.mcmc.sample_chain(
                num_results=config.n_samples,
                num_burnin_steps=config.n_burnin,
                current_state=initial_state,
                kernel=kernel,
                trace_fn=lambda _, pkr: (
                    pkr.inner_results.is_accepted if config.adapt_step_size
                    else pkr.is_accepted
                ),
            )
        
        samples, is_accepted = run_chain()
        
        runtime = time.time() - start_time
        
        # Convert to numpy
        samples_np = samples.numpy()
        is_accepted_np = is_accepted.numpy()
        accept_rate = float(np.mean(is_accepted_np))
        
        if config.verbose:
            print(f"\n{'='*80}")
            print("HMC Sampling Complete")
            print('='*80)
            print(f"Runtime: {runtime:.2f}s")
            print(f"Acceptance rate: {accept_rate:.3%}")
            print(f"Samples collected: {config.n_samples}")
            print()
        
        return {
            'samples': samples_np,
            'is_accepted': is_accepted_np,
            'accept_rate': accept_rate,
            'runtime': runtime,
        }


def compute_ess(samples: np.ndarray, max_lag: Optional[int] = None) -> float:
    """
    Compute effective sample size (ESS) using the initial positive sequence method.
    
    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_params)
        MCMC samples
    max_lag : int, optional
        Maximum lag for autocorrelation computation
        
    Returns
    -------
    ess : float
        Effective sample size (averaged over parameters)
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    
    n_samples, n_params = samples.shape
    
    if max_lag is None:
        max_lag = min(n_samples // 2, 1000)
    
    ess_per_param = []
    
    for j in range(n_params):
        x = samples[:, j]
        
        # Standardize
        x = (x - np.mean(x)) / (np.std(x) + 1e-10)
        
        # Compute autocorrelation
        acf = np.correlate(x, x, mode='full')[len(x)-1:]
        acf = acf / acf[0]
        acf = acf[:max_lag]
        
        # Sum until first negative autocorrelation (initial positive sequence)
        tau = 1.0
        for lag in range(1, len(acf)):
            if acf[lag] < 0:
                break
            tau += 2 * acf[lag]
        
        # ESS = N / tau
        ess = n_samples / tau
        ess_per_param.append(ess)
    
    return np.mean(ess_per_param)


def compute_ess_per_second(samples: np.ndarray, runtime: float) -> float:
    """
    Compute effective sample size per second.
    
    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_params)
        MCMC samples
    runtime : float
        Total runtime in seconds
        
    Returns
    -------
    ess_per_sec : float
        Effective samples per second
    """
    ess = compute_ess(samples)
    return ess / runtime
