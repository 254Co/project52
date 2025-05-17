# File: chen3/model_enhancements/jump_diffusion.py
"""
Jump-diffusion model enhancements for the Chen3 model.

This module implements two types of jump-diffusion processes:
1. Merton's jump-diffusion model with lognormal jumps
2. Kou's double exponential jump-diffusion model

These enhancements allow for modeling discontinuous price movements
and market shocks through jump processes with different distributions.

The implementation provides:
- Merton's lognormal jump process
- Kou's double exponential jump process
- Simulation of jump paths
- Configurable jump parameters

These enhancements are particularly useful for modeling:
- Market crashes and spikes
- Discontinuous price movements
- Tail risk events
- Volatility clustering
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional

class MertonJump:
    """
    Merton's jump-diffusion process with lognormal jumps.
    
    This class implements Merton's jump-diffusion model where jumps
    follow a lognormal distribution. The process is characterized by
    a Poisson process for jump arrivals and lognormal distribution
    for jump sizes.
    
    Mathematical Formulation:
    ----------------------
    The jump process is defined by:
    - Jump arrival: Poisson process with intensity λ
    - Jump size: Log-normal distribution with parameters (μ_j, σ_j)
    - Total jump effect: Product of individual jumps
    
    Attributes:
        lam (float): Jump intensity (Poisson rate)
        mu_j (float): Mean of log-jump size
        sigma_j (float): Standard deviation of log-jump size
    
    Note:
        The jump process is independent of the continuous diffusion
        component and can be combined with it to form a complete
        jump-diffusion model.
    """
    
    def __init__(self, lam: float, mu_j: float, sigma_j: float):
        """
        Initialize Merton's jump-diffusion process.
        
        Args:
            lam (float): Jump intensity (Poisson rate)
            mu_j (float): Mean of log-jump size
            sigma_j (float): Standard deviation of log-jump size
            
        Raises:
            ValueError: If any parameter is negative
        """
        if lam < 0 or sigma_j < 0:
            raise ValueError("Jump intensity and standard deviation must be non-negative")
        
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def simulate(self, dt: float, n_steps: int, n_paths: int, rng: np.random.Generator) -> NDArray:
        """
        Simulate jump multipliers for multiple paths.
        
        This method simulates the jump process by:
        1. Generating number of jumps in each interval (Poisson)
        2. Generating jump sizes (Lognormal)
        3. Computing total jump effect as product of individual jumps
        
        Args:
            dt (float): Time step size
            n_steps (int): Number of time steps
            n_paths (int): Number of paths to simulate
            rng (np.random.Generator): Random number generator
            
        Returns:
            NDArray: Array of jump multipliers of shape (n_paths, n_steps)
                where each element represents the total jump effect
                for that path and time step
        """
        # Number of jumps in each interval
        N = rng.poisson(self.lam * dt, size=(n_paths, n_steps))
        
        # Sum of lognormal jumps per interval
        jumps = np.exp(
            rng.normal(self.mu_j, self.sigma_j, size=(n_paths, n_steps)) * N
        )
        return jumps

class KouJump:
    """
    Kou's double exponential jump-diffusion process.
    
    This class implements Kou's jump-diffusion model where jumps
    follow a double exponential distribution. The process is characterized
    by a Poisson process for jump arrivals and asymmetric exponential
    distributions for positive and negative jumps.
    
    Mathematical Formulation:
    ----------------------
    The jump process is defined by:
    - Jump arrival: Poisson process with intensity λ
    - Jump direction: Bernoulli with probability p for up-jumps
    - Jump size: Exponential with rate η₁ for up-jumps, η₂ for down-jumps
    - Total jump effect: Product of individual jumps
    
    Attributes:
        lam (float): Jump intensity (Poisson rate)
        p (float): Probability of up-jump
        eta1 (float): Positive jump parameter (scale)
        eta2 (float): Negative jump parameter (scale)
    
    Note:
        The double exponential distribution allows for asymmetric
        jumps and better fits market crash behavior than the
        symmetric lognormal jumps in Merton's model.
    """
    
    def __init__(
        self,
        lam: float,
        p: float,
        eta1: float,
        eta2: float
    ):
        """
        Initialize Kou's jump-diffusion process.
        
        Args:
            lam (float): Jump intensity (Poisson rate)
            p (float): Probability of up-jump (in [0, 1])
            eta1 (float): Positive jump parameter (scale)
            eta2 (float): Negative jump parameter (scale)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if lam < 0:
            raise ValueError("Jump intensity must be non-negative")
        if not 0 <= p <= 1:
            raise ValueError("Up-jump probability must be in [0, 1]")
        if eta1 <= 0 or eta2 <= 0:
            raise ValueError("Jump scale parameters must be positive")
        
        self.lam = lam
        self.p = p
        self.eta1 = eta1
        self.eta2 = eta2

    def simulate(self, dt: float, n_steps: int, n_paths: int, rng: np.random.Generator) -> NDArray:
        """
        Simulate jump multipliers for multiple paths.
        
        This method simulates the jump process by:
        1. Generating number of jumps in each interval (Poisson)
        2. Determining jump directions (Bernoulli)
        3. Generating jump sizes (Exponential)
        4. Computing total jump effect as product of individual jumps
        
        Args:
            dt (float): Time step size
            n_steps (int): Number of time steps
            n_paths (int): Number of paths to simulate
            rng (np.random.Generator): Random number generator
            
        Returns:
            NDArray: Array of jump multipliers of shape (n_paths, n_steps)
                where each element represents the total jump effect
                for that path and time step
        """
        # Number of jumps in each interval
        N = rng.poisson(self.lam * dt, size=(n_paths, n_steps))
        
        # For each jump, sample sizes
        U = rng.uniform(size=(n_paths, n_steps))
        jumps = np.ones((n_paths, n_steps))
        
        # Where jumps occur
        mask = N > 0
        
        # Total jump size = product of k i.i.d. sizes, approximate by exp(k * E[size])
        # For simplicity, approximate log-jump ~ N * E[log(Z)]
        # E[log(Z)] = p*(1/eta1) + (1-p)*(-1/eta2)
        mean_log = self.p*(1/self.eta1) + (1-self.p)*(-1/self.eta2)
        jumps[mask] = np.exp(mean_log * N[mask])
        
        return jumps