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
    """Configuration parameters for :class:`HMCSampler`.

    Attributes
    ----------
    n_samples : int
        Number of posterior samples to retain after burn-in. Default ``1000``.
    n_burnin : int
        Number of warm-up iterations to discard. Default ``500``.
    step_size : float
        Initial leapfrog step size. Default ``0.01``.
    n_leapfrog_steps : int
        Number of leapfrog steps per HMC transition. Default ``10``.
    adapt_step_size : bool
        If ``True``, adapt the step size during burn-in using TFP's
        simple step-size adaptation. Default ``True``.
    target_accept_prob : float
        Target acceptance probability used by adaptive step-size tuning.
        Default ``0.65``.
    verbose : bool
        If ``True``, print progress and summary information. Default ``True``.
    """
    n_samples: int = 1000
    
    n_burnin: int = 500
    
    step_size: float = 0.01
    
    n_leapfrog_steps: int = 10
    
    adapt_step_size: bool = True
    
    target_accept_prob: float = 0.65
    
    verbose: bool = True


class HMCSampler:
    """Hamiltonian Monte Carlo sampler for parameter inference.

    Wraps TensorFlow Probability's HMC kernel to sample from a differentiable
    log-posterior defined in TensorFlow.

    Parameters
    ----------
    log_posterior_fn : callable
        Callable mapping a TensorFlow parameter tensor to a scalar log
        posterior value.
    config : HMCConfig, optional
        Sampler configuration. If omitted, default :class:`HMCConfig`
        settings are used.

    Notes
    -----
    Assumes ``log_posterior_fn`` is differentiable with respect to its input
    and operates in the same parameter space as the supplied initial state.
    """
    
    def __init__(
        self,
        log_posterior_fn: Callable[[tf.Tensor], tf.Tensor],
        config: Optional[HMCConfig] = None,
    ):
        """Construct an HMC sampler.

        Parameters
        ----------
        log_posterior_fn : callable
            TensorFlow function that computes log p(θ | Y) given parameters θ.
            Must be differentiable (e.g., using tf.function and tf.GradientTape).
        config : HMCConfig, optional
            Configuration for the sampler.

        Returns
        -------
        None
            Stores the sampling callable and sampler configuration.

        Notes
        -----
        Assumes the callable is compatible with TensorFlow Probability MCMC
        kernels and returns a scalar log density.
        """
        self.log_posterior_fn = log_posterior_fn
        self.config = config or HMCConfig()
        
    def run(self, initial_params: np.ndarray) -> Dict:
        """Run HMC sampling using TensorFlow Probability.
        
        Parameters
        ----------
        initial_params : ndarray, shape (n_params,)
            Initial parameter values (in constrained space if needed).
            
        Returns
        -------
        results : dict
            Dictionary containing posterior samples, acceptance indicators,
            acceptance rate, and run time.  Keys are ``samples``,
            ``is_accepted``, ``accept_rate``, and ``runtime``.

        Notes
        -----
        Assumes ``initial_params`` has the shape and parameterisation expected
        by ``log_posterior_fn``. Returned samples remain in that same space.
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
    """Compute effective sample size using the initial positive sequence method.
    
    Parameters
    ----------
    samples : ndarray, shape (n_samples,) or (n_samples, n_params)
        MCMC samples.
    max_lag : int, optional
        Maximum lag for autocorrelation computation.
        
    Returns
    -------
    ess : float
        Effective sample size averaged over parameter dimensions.

    Notes
    -----
    Assumes samples are ordered draws from a stationary Markov chain. For
    vector-valued samples, ESS is computed per parameter and averaged.
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
    """Compute effective sample size per second.
    
    Parameters
    ----------
    samples : ndarray, shape (n_samples,) or (n_samples, n_params)
        MCMC samples.
    runtime : float
        Total runtime in seconds.
        
    Returns
    -------
    ess_per_sec : float
        Effective samples per second.

    Notes
    -----
    Assumes ``runtime`` is positive and measured in seconds for the chain
    represented by ``samples``.
    """
    ess = compute_ess(samples)
    return ess / runtime
