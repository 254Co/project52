# File: chen3/calibration/optim.py
"""
Model Calibration Optimization Module

This module provides optimization routines for calibrating the three-factor Chen
model to market data. It implements both local and global optimization methods
to fit model parameters to observed market prices and rates.

The module supports:
1. Joint calibration of interest rate and volatility parameters
2. Multiple optimization methods (global and local)
3. Customizable loss functions for different market data
4. Parameter bounds and constraints

The calibration process minimizes the difference between:
- Model-generated volatility surface and market quotes
- Model-generated yield curve and market rates
"""

from typing import Any, Dict

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .loss import vol_surface_loss, yield_curve_loss

# Order of keys in the parameter vector:
_OPT_KEYS = [
    "kappa",
    "theta",
    "sigma",
    "r0",  # interest‐rate CIR params
    "kappa_v",
    "theta_v",
    "sigma_v",  # variance (Heston) params
]


def calibrate(
    params_guess: Dict[str, float],
    market_data: Dict[str, Any],
    model_funcs: Dict[str, Any],
    method: str = "global",
) -> Dict[str, float]:
    """
    Calibrate the three-factor Chen model to market data.

    This function performs joint calibration of the model's interest rate and
    volatility parameters to match observed market data. It supports both global
    optimization (differential evolution) and local optimization (L-BFGS-B)
    methods.

    The calibration process:
    1. Combines volatility surface and yield curve fitting
    2. Optimizes all parameters simultaneously
    3. Respects parameter bounds and constraints
    4. Returns the optimal parameter set

    Args:
        params_guess (Dict[str, float]): Initial parameter values for:
            - kappa: Mean reversion speed of rates
            - theta: Long-term mean of rates
            - sigma: Volatility of rates
            - r0: Initial rate level
            - kappa_v: Mean reversion speed of variance
            - theta_v: Long-term mean of variance
            - sigma_v: Volatility of variance

        market_data (Dict[str, Any]): Market data for calibration:
            - vols: Volatility surface quotes (n_tenors × n_strikes)
            - tenors: Option tenors
            - strikes: Option strike prices
            - curve: Yield curve points
            - maturities: Yield curve maturities

        model_funcs (Dict[str, Any]): Model functions for:
            - vol_fn: Function to compute model volatilities
            - rate_fn: Function to compute model rates

        method (str): Optimization method:
            - "global": Use differential evolution
            - other: Use L-BFGS-B local optimization

    Returns:
        Dict[str, float]: Optimized parameter values

    Example:
        >>> params_guess = {
        ...     "kappa": 0.1, "theta": 0.05, "sigma": 0.1, "r0": 0.03,
        ...     "kappa_v": 2.0, "theta_v": 0.04, "sigma_v": 0.3
        ... }
        >>> market_data = {
        ...     "vols": np.array([[0.2, 0.3], [0.25, 0.35]]),
        ...     "tenors": np.array([0.25, 0.5]),
        ...     "strikes": np.array([0.9, 1.1]),
        ...     "curve": np.array([0.02, 0.03, 0.04]),
        ...     "maturities": np.array([1, 2, 3])
        ... }
        >>> model_funcs = {
        ...     "vol_fn": compute_vols,
        ...     "rate_fn": compute_rates
        ... }
        >>> params_opt = calibrate(params_guess, market_data, model_funcs)

    Notes:
        - All parameters are constrained to be positive
        - Global optimization is more robust but slower
        - Local optimization is faster but may find local minima
        - Loss function combines volatility and rate fitting errors
    """

    def objective(x: np.ndarray) -> float:
        """
        Objective function for optimization.

        Combines volatility surface and yield curve fitting errors into a
        single loss function to be minimized.

        Args:
            x (np.ndarray): Flattened parameter vector

        Returns:
            float: Combined loss value
        """
        # rebuild param dict
        p = {k: float(x[i]) for i, k in enumerate(_OPT_KEYS)}
        L1 = vol_surface_loss(
            p,
            market_data["vols"],
            market_data["tenors"],
            market_data["strikes"],
            model_funcs["vol_fn"],
        )
        L2 = yield_curve_loss(
            p, market_data["curve"], market_data["maturities"], model_funcs["rate_fn"]
        )
        return L1 + L2

    x0 = np.array([params_guess[k] for k in _OPT_KEYS], dtype=float)
    bounds = [(1e-8, None)] * len(_OPT_KEYS)

    if method == "global":
        res = differential_evolution(objective, bounds=bounds)
    else:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    x_opt = res.x
    return {k: float(x_opt[i]) for i, k in enumerate(_OPT_KEYS)}
