"""
Path generation module for the LSM Engine.

This module provides functions for generating Monte Carlo simulation paths
using Geometric Brownian Motion (GBM). It supports antithetic variates
for variance reduction and configurable random number generation.
"""

# File: lsm_engine/engine/pathgen.py
import numpy as np
from numpy.random import default_rng

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
    Generate Monte Carlo simulation paths using Geometric Brownian Motion.
    
    This function generates asset price paths following GBM:
    dS = μSdt + σSdW
    
    Args:
        s0: Initial asset price
        mu: Drift rate (expected return)
        sigma: Volatility
        maturity: Time to maturity in years
        n_steps: Number of time steps
        n_paths: Number of simulation paths
        seed: Random seed for reproducibility
        antithetic: Whether to use antithetic variates for variance reduction
        
    Returns:
        np.ndarray: Array of shape (n_paths, n_steps + 1) containing the simulated paths
    """
    dt = maturity / n_steps
    rng = default_rng(seed)
    dw = rng.standard_normal(size=(n_paths, n_steps))
    if antithetic:
        dw = np.concatenate([dw, -dw], axis=0)  # Add antithetic paths
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * dw
    log_paths = np.cumsum(np.insert(increments, 0, 0.0, axis=1), axis=1)
    return s0 * np.exp(log_paths)  # Convert log-paths to price paths