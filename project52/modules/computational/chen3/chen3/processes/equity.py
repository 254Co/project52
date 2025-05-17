# File: chen3/processes/equity.py
"""
Heston-like Equity and Variance Process Implementation

This module implements a Heston-like stochastic process for modeling equity prices
and their variance. The process combines a geometric Brownian motion for the
equity price with a CIR process for the variance, allowing for stochastic
volatility effects.

The process follows the coupled SDEs:
    dS_t = (μ - q)S_t dt + √v_t S_t dW^S_t
    dv_t = κ_v(θ_v - v_t)dt + σ_v√v_t dW^v_t

where:
- S_t: Equity price
- v_t: Variance process
- μ: Drift rate
- q: Dividend yield
- κ_v: Variance mean reversion speed
- θ_v: Long-term variance level
- σ_v: Volatility of variance
- W^S_t, W^v_t: Correlated Brownian motions
"""

import numpy as np
from .base import StateProcess


class EquityProcess(StateProcess):
    """
    Implementation of a Heston-like equity and variance process.
    
    This class implements a two-dimensional stochastic process combining:
    1. A geometric Brownian motion for the equity price
    2. A CIR process for the variance
    
    The process is particularly well-suited for:
    - Modeling equity prices with stochastic volatility
    - Capturing volatility clustering effects
    - Pricing options with volatility smile
    - Simulating correlated equity and variance dynamics
    
    Attributes:
        mu (float): Drift rate of the equity process
        q (float): Continuous dividend yield
        kappa_v (float): Mean reversion speed of variance
        theta_v (float): Long-term mean level of variance
        sigma_v (float): Volatility of variance
        v0 (float): Initial variance level
    
    Example:
        >>> process = EquityProcess(
        ...     mu=0.05, q=0.02,
        ...     kappa_v=2.0, theta_v=0.04, sigma_v=0.3, v0=0.04
        ... )
        >>> state = np.array([[100.0, 0.04], [101.0, 0.05]])
        >>> drift = process.drift(state)
        >>> diffusion = process.diffusion(state)
    """
    
    def __init__(self, mu: float, q: float,
                 kappa_v: float, theta_v: float, sigma_v: float, v0: float):
        """
        Initialize the equity and variance process with specified parameters.
        
        Args:
            mu (float): Drift rate of the equity process
            q (float): Continuous dividend yield
            kappa_v (float): Mean reversion speed of variance (κ_v > 0)
            theta_v (float): Long-term mean level of variance (θ_v > 0)
            sigma_v (float): Volatility of variance (σ_v > 0)
            v0 (float): Initial variance level (v0 ≥ 0)
        
        Raises:
            ValueError: If any parameter is negative or if 2κ_vθ_v < σ_v²
                      (Feller condition for non-negative variance)
        """
        self.mu = mu
        self.q = q
        self.kappa_v = kappa_v
        self.theta_v = theta_v
        self.sigma_v = sigma_v
        self.v0 = v0

    def drift(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the drift terms for both equity and variance processes.
        
        The drift terms are:
        - Equity: (μ - q)S_t
        - Variance: κ_v(θ_v - v_t)
        
        Args:
            state (np.ndarray): Current state values, shape (n_paths, 2)
                               state[:,0] = equity price (S)
                               state[:,1] = variance (v)
        
        Returns:
            np.ndarray: Drift terms for both processes, shape (n_paths, 2)
                       [equity_drift, variance_drift]
        """
        # state[:,0]=S, state[:,1]=v
        S, v = state[:,0], state[:,1]
        return np.stack([
            (self.mu - self.q)*S,
            self.kappa_v*(self.theta_v - v)
        ], axis=1)

    def diffusion(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the diffusion terms for both equity and variance processes.
        
        The diffusion terms are:
        - Equity: √v_t S_t
        - Variance: σ_v√v_t
        
        Note: The maximum with 0 ensures non-negative variance
        
        Args:
            state (np.ndarray): Current state values, shape (n_paths, 2)
                               state[:,0] = equity price (S)
                               state[:,1] = variance (v)
        
        Returns:
            np.ndarray: Diffusion terms for both processes, shape (n_paths, 2)
                       [equity_diffusion, variance_diffusion]
        """
        S, v = state[:,0], state[:,1]
        return np.stack([
            np.sqrt(np.maximum(v, 0.0))*S,
            self.sigma_v*np.sqrt(np.maximum(v, 0.0))
        ], axis=1)
