"""
Path Generation Module for Monte Carlo Simulation

This module provides classes and functions for generating simulated price paths
used in Monte Carlo methods for option pricing. The module includes:
- Abstract Process class defining the interface for stochastic processes
- GBMProcess implementation of Geometric Brownian Motion
- Helper function for generating GBM paths

The implementation supports:
- Antithetic variates for variance reduction
- Configurable random number generation
- Flexible path dimensions and time steps
"""

# File: lsm_engine/engine/pathgen.py
import numpy as np
from abc import ABC, abstractmethod
from numpy.random import default_rng


class Process(ABC):
    """
    Abstract base class for stochastic processes.
    
    This class defines the interface that all stochastic process implementations
    must follow for use in Monte Carlo simulations. The interface requires
    implementing a simulate() method that generates price paths according to
    the specific process dynamics.
    """
    
    @abstractmethod
    def simulate(
        self,
        s0: float,
        mu: float,
        sigma: float,
        maturity: float,
        n_steps: int,
        n_paths: int,
        seed: int | None = None,
        antithetic: bool = True,
    ) -> np.ndarray:
        """
        Generate simulated price paths according to the process dynamics.
        
        Args:
            s0: Initial price
            mu: Drift rate (annualized)
            sigma: Volatility (annualized)
            maturity: Time to maturity in years
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            seed: Random number generator seed
            antithetic: Whether to use antithetic variates
            
        Returns:
            Array of shape (n_paths*(2 if antithetic else 1), n_steps+1)
            containing the simulated price paths
        """
        ...


class GBMProcess(Process):
    """
    Geometric Brownian Motion (GBM) process implementation.
    
    This class implements the GBM stochastic process, which is commonly used
    to model stock prices in the Black-Scholes framework. The process follows
    the stochastic differential equation:
    
        dS = μS dt + σS dW
    
    where:
    - S is the stock price
    - μ is the drift rate
    - σ is the volatility
    - W is a Wiener process
    """
    
    def simulate(
        self,
        s0: float,
        mu: float,
        sigma: float,
        maturity: float,
        n_steps: int,
        n_paths: int,
        seed: int | None = None,
        antithetic: bool = True,
    ) -> np.ndarray:
        """
        Generate GBM price paths using the exact solution.
        
        The exact solution to the GBM SDE is:
            S(t) = S(0) * exp((μ - 0.5σ²)t + σW(t))
            
        This implementation:
        1. Generates standard normal increments
        2. Applies antithetic variates if requested
        3. Computes log returns using the exact solution
        4. Converts to price paths
        
        Args:
            s0: Initial price
            mu: Drift rate (annualized)
            sigma: Volatility (annualized)
            maturity: Time to maturity in years
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            seed: Random number generator seed
            antithetic: Whether to use antithetic variates
            
        Returns:
            Array of shape (n_paths*(2 if antithetic else 1), n_steps+1)
            containing the simulated GBM price paths
        """
        dt = maturity / n_steps
        rng = default_rng(seed)
        dw = rng.standard_normal(size=(n_paths, n_steps))
        if antithetic:
            dw = np.concatenate([dw, -dw], axis=0)
        increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * dw
        log_paths = np.cumsum(np.insert(increments, 0, 0.0, axis=1), axis=1)
        return s0 * np.exp(log_paths)


def generate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = True,
) -> np.ndarray:
    """
    Generate GBM price paths using the GBMProcess implementation.
    
    This is a convenience function that creates a GBMProcess instance and
    calls its simulate() method with the provided parameters.
    
    Args:
        s0: Initial price
        mu: Drift rate (annualized)
        sigma: Volatility (annualized)
        maturity: Time to maturity in years
        n_steps: Number of time steps
        n_paths: Number of paths to simulate
        seed: Random number generator seed
        antithetic: Whether to use antithetic variates
        
    Returns:
        Array of shape (n_paths*(2 if antithetic else 1), n_steps+1)
        containing the simulated GBM price paths
    """
    process = GBMProcess()
    return process.simulate(s0, mu, sigma, maturity, n_steps, n_paths, seed, antithetic)