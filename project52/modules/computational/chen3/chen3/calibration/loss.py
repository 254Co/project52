# File: chen3/calibration/loss.py
"""Calibration loss functions for Chen model parameters."""
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
    Squared‐error between model‐implied vols and market vols.

    params: keys include 'kappa_v','theta_v','sigma_v', etc.
    market_vols: shape (n_tenors, n_strikes)
    tenors:      shape (n_tenors,)
    strikes:     shape (n_strikes,)
    model_vol_fn: function(params, tenors, strikes) -> vols ndarray
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
    Squared‐error between model zero rates and market curve.

    params: keys include 'kappa','theta','sigma','r0'
    market_curve:  shape (n_points,)
    maturities:    shape (n_points,)
    model_rate_fn: function(params, maturities) -> rates ndarray
    """
    model_rates = model_rate_fn(params, maturities)
    error = market_curve - model_rates
    return float(np.sum(error**2))
