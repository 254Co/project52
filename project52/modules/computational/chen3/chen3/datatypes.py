# File: chen3/datatypes.py
"""
Core Parameter Data Structures for the Chen3 Model

This module defines the fundamental data structures used to parameterize the three-factor
Chen model. These structures encapsulate all necessary parameters for interest rates,
equity dynamics, and their correlations.

The module provides three main parameter classes:
1. RateParams: Parameters for the interest rate process
2. EquityParams: Parameters for the equity and variance processes
3. ModelParams: Container for all model parameters including correlations
"""
from typing import NamedTuple
import numpy as np

class RateParams(NamedTuple):
    """
    Parameters for the Cox-Ingersoll-Ross (CIR) interest rate process.
    
    The CIR process follows the SDE:
        dr_t = κ(θ - r_t)dt + σ√r_t dW_t
    
    Attributes:
        kappa (float): Mean reversion speed of the interest rate
                      Higher values indicate faster mean reversion
        theta (float): Long-term mean level of the interest rate
                      The rate reverts to this level in the long run
        sigma (float): Volatility of the interest rate process
                      Controls the magnitude of random fluctuations
        r0 (float): Initial interest rate level
                   Starting point for the simulation
    """
    kappa: float  # mean reversion speed
    theta: float  # long-term mean
    sigma: float  # volatility
    r0: float     # initial rate

class EquityParams(NamedTuple):
    """
    Parameters for the equity and variance processes.
    
    The equity process follows:
        dS_t = (r_t - q)S_t dt + √v_t S_t dW^S_t
    The variance process follows:
        dv_t = κ_v(θ_v - v_t)dt + σ_v√v_t dW^v_t
    
    Attributes:
        mu (float): Drift rate of the equity process
                   Typically set to risk-free rate in risk-neutral measure
        q (float): Continuous dividend yield
                  Reduces the growth rate of the stock price
        S0 (float): Initial stock price
                   Starting point for the simulation
        v0 (float): Initial variance level
                   Starting point for the volatility process
        kappa_v (float): Mean reversion speed of the variance process
                        Higher values indicate faster mean reversion
        theta_v (float): Long-term mean level of the variance
                        The variance reverts to this level in the long run
        sigma_v (float): Volatility of the variance process
                        Controls the magnitude of volatility fluctuations
    """
    mu: float             # drift
    q: float              # dividend yield
    S0: float             # initial stock price
    v0: float             # initial variance
    kappa_v: float        # variance mean-reversion
    theta_v: float        # variance long-run
    sigma_v: float        # volatility of variance

class ModelParams(NamedTuple):
    """
    Container for all model parameters including correlations.
    
    This class combines the interest rate and equity parameters with their
    correlation structure to form a complete model specification.
    
    Attributes:
        rate (RateParams): Parameters for the interest rate process
        equity (EquityParams): Parameters for the equity and variance processes
        corr_matrix (np.ndarray): Correlation matrix between the three factors
                                 Shape (3,3) representing correlations between:
                                 - Interest rate
                                 - Equity price
                                 - Variance
                                 Must be positive definite and symmetric
    """
    rate: RateParams
    equity: EquityParams
    corr_matrix: np.ndarray  # shape (3,3)

