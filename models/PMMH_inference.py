"""
Particle Marginal Metropolis-Hastings (PMMH) for parameter inference.
"""

import numpy as np
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class PMMHConfig:
    """Configuration for PMMH sampler."""
    n_samples: int = 1000
    """Number of samples to draw (after burnin)"""
    
    n_burnin: int = 500
    """Number of burnin samples to discard"""
    
    proposal_std: np.ndarray = None
    """Standard deviation for Gaussian random walk proposal (array for each parameter)"""
    
    adapt_proposal: bool = True
    """Whether to adapt proposal during burnin"""
    
    adapt_interval: int = 50
    """Number of iterations between proposal adaptations"""
    
    target_accept_rate: float = 0.234
    """Target acceptance rate for proposal adaptation (optimal for Gaussian targets)"""
    
    verbose: bool = True
    """Whether to print progress"""
    
    print_interval: int = 100
    """Number of iterations between progress prints"""


class PMMHSampler:
    """
    Particle Marginal Metropolis-Hastings sampler for parameter inference.
    
    PMMH uses a particle filter to estimate the likelihood p(Y_{1:T} | θ),
    which is then used in a Metropolis-Hastings algorithm to sample from
    the posterior p(θ | Y_{1:T}).
    """
    
    def __init__(
        self,
        log_likelihood_fn: Callable[[np.ndarray], float],
        log_prior_fn: Callable[[np.ndarray], float],
        config: Optional[PMMHConfig] = None,
    ):
        """
        Parameters
        ----------
        log_likelihood_fn : callable
            Function that computes log p(Y_{1:T} | θ) given parameters θ.
            Should run a particle filter and return the log marginal likelihood.
        log_prior_fn : callable
            Function that computes log p(θ) given parameters θ.
        config : PMMHConfig, optional
            Configuration for the sampler.
        """
        self.log_likelihood_fn = log_likelihood_fn
        self.log_prior_fn = log_prior_fn
        self.config = config or PMMHConfig()
        
    def run(self, initial_params: np.ndarray) -> Dict:
        """
        Run PMMH sampler.
        
        Parameters
        ----------
        initial_params : ndarray, shape (n_params,)
            Initial parameter values.
            
        Returns
        -------
        results : dict
            Dictionary containing:
                - samples: ndarray of shape (n_samples, n_params)
                - log_likelihood_trace: ndarray of shape (n_samples + n_burnin,)
                - log_prior_trace: ndarray of shape (n_samples + n_burnin,)
                - accept_trace: ndarray of shape (n_samples + n_burnin,)
                - accept_rate: float
                - runtime: float
                - proposal_std: ndarray, final proposal standard deviation
        """
        config = self.config
        n_total = config.n_samples + config.n_burnin
        n_params = len(initial_params)
        
        # Initialize proposal covariance
        if config.proposal_std is None:
            proposal_std = 0.1 * np.abs(initial_params)
            proposal_std[proposal_std < 0.01] = 0.01  # Minimum std
        else:
            proposal_std = config.proposal_std.copy()
        
        # Storage
        samples = np.zeros((n_total, n_params))
        log_likelihood_trace = np.zeros(n_total)
        log_prior_trace = np.zeros(n_total)
        accept_trace = np.zeros(n_total, dtype=bool)
        
        # Initialize chain
        current_params = initial_params.copy()
        current_log_prior = self.log_prior_fn(current_params)
        current_log_likelihood = self.log_likelihood_fn(current_params)
        current_log_posterior = current_log_prior + current_log_likelihood
        
        samples[0] = current_params
        log_likelihood_trace[0] = current_log_likelihood
        log_prior_trace[0] = current_log_prior
        
        if config.verbose:
            print(f"\n{'='*80}")
            print("Particle Marginal Metropolis-Hastings (PMMH)")
            print('='*80)
            print(f"Initial parameters: {current_params}")
            print(f"Initial log-posterior: {current_log_posterior:.2f}")
            print(f"  log-likelihood: {current_log_likelihood:.2f}")
            print(f"  log-prior: {current_log_prior:.2f}")
            print(f"\nRunning {n_total} iterations ({config.n_burnin} burnin)...")
            print()
        
        start_time = time.time()
        n_accepted = 0
        recent_accepts = []  # For adaptive proposal
        
        # Main PMMH loop
        for i in range(1, n_total):
            # Propose new parameters (Gaussian random walk)
            proposed_params = current_params + proposal_std * np.random.randn(n_params)
            
            # Evaluate prior
            proposed_log_prior = self.log_prior_fn(proposed_params)
            
            # Check if prior is valid
            if not np.isfinite(proposed_log_prior):
                # Reject automatically if prior is zero/invalid
                accept = False
                proposed_log_likelihood = -np.inf
            else:
                # Evaluate likelihood via particle filter
                proposed_log_likelihood = self.log_likelihood_fn(proposed_params)
                proposed_log_posterior = proposed_log_prior + proposed_log_likelihood
                
                # Metropolis-Hastings acceptance ratio
                log_accept_ratio = proposed_log_posterior - current_log_posterior
                
                # Accept/reject
                if np.log(np.random.rand()) < log_accept_ratio:
                    accept = True
                    current_params = proposed_params
                    current_log_prior = proposed_log_prior
                    current_log_likelihood = proposed_log_likelihood
                    current_log_posterior = proposed_log_posterior
                    n_accepted += 1
                else:
                    accept = False
            
            # Store
            samples[i] = current_params
            log_likelihood_trace[i] = current_log_likelihood
            log_prior_trace[i] = current_log_prior
            accept_trace[i] = accept
            recent_accepts.append(accept)
            
            # Adapt proposal during burnin
            if config.adapt_proposal and i < config.n_burnin:
                if i % config.adapt_interval == 0 and len(recent_accepts) > 0:
                    recent_accept_rate = np.mean(recent_accepts)
                    
                    # Adjust proposal scale to achieve target acceptance rate
                    if recent_accept_rate > config.target_accept_rate:
                        proposal_std *= 1.1
                    else:
                        proposal_std *= 0.9
                    
                    recent_accepts = []
            
            # Print progress
            if config.verbose and i % config.print_interval == 0:
                accept_rate_so_far = n_accepted / i
                phase = "burnin" if i < config.n_burnin else "sampling"
                print(f"Iteration {i}/{n_total} ({phase}): "
                      f"accept={accept}, "
                      f"accept_rate={accept_rate_so_far:.3f}, "
                      f"log_post={current_log_posterior:.2f}")
        
        runtime = time.time() - start_time
        accept_rate = n_accepted / n_total
        
        # Discard burnin
        samples_post_burnin = samples[config.n_burnin:]
        
        if config.verbose:
            print(f"\n{'='*80}")
            print("PMMH Sampling Complete")
            print('='*80)
            print(f"Runtime: {runtime:.2f}s")
            print(f"Total acceptance rate: {accept_rate:.3%}")
            print(f"Post-burnin acceptance rate: {np.mean(accept_trace[config.n_burnin:]):.3%}")
            print(f"Final proposal std: {proposal_std}")
            print()
        
        return {
            'samples': samples_post_burnin,
            'samples_full': samples,
            'log_likelihood_trace': log_likelihood_trace,
            'log_prior_trace': log_prior_trace,
            'accept_trace': accept_trace,
            'accept_rate': accept_rate,
            'accept_rate_post_burnin': np.mean(accept_trace[config.n_burnin:]),
            'runtime': runtime,
            'proposal_std': proposal_std,
        }


def compute_ess(samples: np.ndarray, max_lag: Optional[int] = None) -> float:
    """
    Compute effective sample size (ESS) using the initial positive sequence method.
    
    Parameters
    ----------
    samples : ndarray, shape (n_samples,) or (n_samples, n_params)
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
        
        # ESS
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
