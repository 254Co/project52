"""Simple scenario analyses, Delta approximation & historical VaR.

This module provides risk analysis tools for the Campbell-Cochrane habit formation model,
including scenario analysis, sensitivity measures, and Value-at-Risk calculations.
These tools help assess the model's behavior under different market conditions and
measure various risk metrics.
"""
from __future__ import annotations

import numpy as np

from ..core.analytics import Analytics
from ..core.model import HabitModel
from ..utils.validation import assert_finite

__all__ = ["price_under_shock", "compute_delta", "var_historic"]


def price_under_shock(
    model: HabitModel,
    paths: dict[str, np.ndarray],
    *,
    shock_bps: float = 20.0,
) -> float:
    """Compute risk-free rate under a consumption growth shock.
    
    This function performs a scenario analysis by applying a shock to the
    log consumption growth rate and computing the resulting risk-free rate.
    The shock is specified in basis points (bps).
    
    Args:
        model (HabitModel): The habit formation model.
        paths (dict[str, np.ndarray]): Dictionary containing simulated paths:
            - "g_c": Log consumption growth rates
            - "S": Surplus-consumption ratios
        shock_bps (float, optional): Size of the shock in basis points.
            Defaults to 20.0.
            
    Returns:
        float: Average risk-free rate under the shocked scenario.
        
    Raises:
        TypeError: If shock_bps is not numeric.
        ValueError: If shock_bps is non-positive.
        ValueError: If the computed rate is non-finite.
    """
    if not isinstance(shock_bps, (int, float)):
        raise TypeError("shock_bps must be numeric")
    if shock_bps <= 0:
        raise ValueError("shock_bps must be positive")
        
    shift = shock_bps / 10_000  # convert bps to decimal
    g_c_shocked = paths["g_c"] + shift
    analytics = Analytics(model)
    r_f = analytics.risk_free_rate(g_c_shocked, paths["S"][:, :-1], paths["S"][:, 1:])
    result = float(r_f.mean())
    assert_finite(result, "shocked price")
    return result


def compute_delta(
    model: HabitModel,
    paths: dict[str, np.ndarray],
    d_shock: float = 0.0001,
) -> float:
    """Compute the sensitivity of risk-free rate to consumption growth.
    
    This function calculates the first-order sensitivity (delta) of the risk-free
    rate to changes in consumption growth using a finite-difference approximation.
    
    Args:
        model (HabitModel): The habit formation model.
        paths (dict[str, np.ndarray]): Dictionary containing simulated paths:
            - "g_c": Log consumption growth rates
            - "S": Surplus-consumption ratios
        d_shock (float, optional): Size of the shock for finite difference.
            Defaults to 0.0001.
            
    Returns:
        float: Delta of risk-free rate with respect to consumption growth.
        
    Raises:
        TypeError: If d_shock is not numeric.
        ValueError: If d_shock is non-positive.
        ValueError: If the computed delta is non-finite.
    """
    if not isinstance(d_shock, (int, float)):
        raise TypeError("d_shock must be numeric")
    if d_shock <= 0:
        raise ValueError("d_shock must be positive")
        
    base = Analytics(model).risk_free_rate(paths["g_c"], paths["S"][:, :-1], paths["S"][:, 1:])
    up = price_under_shock(model, paths, shock_bps=d_shock * 10_000)
    delta = (up - base.mean()) / d_shock
    assert_finite(delta, "delta")
    return float(delta)


def var_historic(
    pnl: np.ndarray,
    *,
    level: float = 0.99,
) -> float:
    """Compute historical Value-at-Risk (VaR) from a P&L vector.
    
    This function calculates the Value-at-Risk at a specified confidence level
    using the historical simulation method. VaR represents the maximum potential
    loss that is not exceeded with a given probability.
    
    Args:
        pnl (np.ndarray): Array of profit and loss values.
        level (float, optional): Confidence level for VaR calculation.
            Must be between 0 and 1. Defaults to 0.99 (99% confidence).
            
    Returns:
        float: Value-at-Risk at the specified confidence level.
        
    Raises:
        TypeError: If pnl is not a numpy array or level is not numeric.
        ValueError: If pnl array is empty or level is outside [0,1].
        ValueError: If the computed VaR is non-finite.
        
    Note:
        The VaR is computed as the negative of the level-quantile of the P&L
        distribution, representing the maximum loss that will not be exceeded
        with probability equal to the confidence level.
    """
    if not isinstance(pnl, np.ndarray):
        raise TypeError("pnl must be a numpy array")
    if pnl.size == 0:
        raise ValueError("pnl array cannot be empty")
    if not isinstance(level, (int, float)):
        raise TypeError("level must be numeric")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
        
    result = float(np.quantile(-pnl, level))
    assert_finite(result, "VaR")
    return result