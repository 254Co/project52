# File: chen3/calibration/loss.py
"""
Calibration Loss Functions Module

This module provides loss functions for calibrating the three-factor Chen model
to market data. It implements separate loss functions for volatility surface and
yield curve fitting, using squared error metrics to measure the quality of fit.

The module supports:
1. Volatility surface fitting using squared error
2. Yield curve fitting using squared error
3. Flexible model function interfaces
4. Efficient vectorized computations

The loss functions are designed to:
- Measure the quality of model fit to market data
- Support gradient-based optimization
- Enable joint calibration of all model parameters
"""

import numpy as np
from typing import Dict, Any


def vol_surface_loss(
    params: Dict[str, float],
    market_vols: np.ndarray,
    tenors: np.ndarray,
    strikes: np.ndarray,
    model_vol_fn: Any
) -> float:
    """
    Compute squared error loss between model and market volatility surfaces.
    
    This function calculates the sum of squared differences between model-implied
    and market-observed volatilities across all tenors and strikes. The loss is
    used to fit the model's volatility parameters to market data.
    
    Args:
        params (Dict[str, float]): Model parameters including:
            - kappa_v: Mean reversion speed of variance
            - theta_v: Long-term mean of variance
            - sigma_v: Volatility of variance
            - Other parameters needed by model_vol_fn
        
        market_vols (np.ndarray): Market-observed volatility surface with shape
                                 (n_tenors, n_strikes)
        
        tenors (np.ndarray): Option tenors with shape (n_tenors,)
        
        strikes (np.ndarray): Option strike prices with shape (n_strikes,)
        
        model_vol_fn (callable): Function that computes model-implied volatilities:
            f(params, tenors, strikes) -> np.ndarray of shape (n_tenors, n_strikes)
    
    Returns:
        float: Sum of squared errors between model and market volatilities
    
    Example:
        >>> params = {
        ...     "kappa_v": 2.0, "theta_v": 0.04, "sigma_v": 0.3,
        ...     "kappa": 0.1, "theta": 0.05, "sigma": 0.1, "r0": 0.03
        ... }
        >>> market_vols = np.array([[0.2, 0.3], [0.25, 0.35]])
        >>> tenors = np.array([0.25, 0.5])
        >>> strikes = np.array([0.9, 1.1])
        >>> loss = vol_surface_loss(params, market_vols, tenors, strikes, compute_vols)
    
    Notes:
        - Uses squared error for optimization stability
        - Supports arbitrary volatility surface shapes
        - Assumes market_vols and model_vols have same shape
        - Returns scalar loss value for optimization
    """
    model_vols = model_vol_fn(params, tenors, strikes)
    error = market_vols - model_vols
    return float(np.sum(error**2))


def yield_curve_loss(
    params: Dict[str, float],
    market_curve: np.ndarray,
    maturities: np.ndarray,
    model_rate_fn: Any
) -> float:
    """
    Compute squared error loss between model and market yield curves.
    
    This function calculates the sum of squared differences between model-implied
    and market-observed zero rates across all maturities. The loss is used to
    fit the model's interest rate parameters to market data.
    
    Args:
        params (Dict[str, float]): Model parameters including:
            - kappa: Mean reversion speed of rates
            - theta: Long-term mean of rates
            - sigma: Volatility of rates
            - r0: Initial rate level
            - Other parameters needed by model_rate_fn
        
        market_curve (np.ndarray): Market-observed zero rates with shape (n_points,)
        
        maturities (np.ndarray): Yield curve maturities with shape (n_points,)
        
        model_rate_fn (callable): Function that computes model-implied rates:
            f(params, maturities) -> np.ndarray of shape (n_points,)
    
    Returns:
        float: Sum of squared errors between model and market rates
    
    Example:
        >>> params = {
        ...     "kappa": 0.1, "theta": 0.05, "sigma": 0.1, "r0": 0.03,
        ...     "kappa_v": 2.0, "theta_v": 0.04, "sigma_v": 0.3
        ... }
        >>> market_curve = np.array([0.02, 0.03, 0.04])
        >>> maturities = np.array([1, 2, 3])
        >>> loss = yield_curve_loss(params, market_curve, maturities, compute_rates)
    
    Notes:
        - Uses squared error for optimization stability
        - Supports arbitrary yield curve shapes
        - Assumes market_curve and model_rates have same shape
        - Returns scalar loss value for optimization
    """
    model_rates = model_rate_fn(params, maturities)
    error = market_curve - model_rates
    return float(np.sum(error**2))
