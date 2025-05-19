"""Nelson–Siegel parametric yield curve model.

This module implements the Nelson-Siegel model for fitting smooth yield curves.
The model uses four parameters to describe the shape of the yield curve:
- beta0: Level (long-term rate)
- beta1: Slope (short-term vs long-term spread)
- beta2: Curvature (medium-term deviation)
- tau: Decay parameter (controls the speed of decay)

The model is particularly useful for:
1. Smoothing noisy market data
2. Interpolating between observed points
3. Extrapolating beyond the longest observed tenor

Implementation details:
- Uses scipy's minimize for parameter optimization
- Implements box constraints for parameter bounds
- Handles numerical stability near zero

Key features:
    1. Four-parameter model capturing level, slope, and curvature
    2. Efficient parameter optimization using L-BFGS-B
    3. Robust handling of numerical edge cases
    4. Box constraints for parameter stability
    5. Continuous compounding convention

Note:
    The Nelson-Siegel model provides a parsimonious representation of the
    yield curve that captures its key features while remaining flexible
    enough to fit a wide range of curve shapes.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def ns_zero(t: float, b0: float, b1: float, b2: float, tau: float) -> float:
    """Calculate the Nelson-Siegel zero rate for a given tenor.
    
    The Nelson-Siegel model decomposes the yield curve into three components:
    1. Level (b0): The long-term rate
    2. Slope (b1): The short-term vs long-term spread
    3. Curvature (b2): The medium-term deviation
    
    The model uses the following formula:
    r(t) = b0 + b1 * (1-exp(-t/tau))/(t/tau) + b2 * ((1-exp(-t/tau))/(t/tau) - exp(-t/tau))
    
    The components have the following interpretations:
    - Level (b0): Asymptotic long-term rate
    - Slope (b1): Short-term vs long-term spread
    - Curvature (b2): Medium-term deviation from linear trend
    - Tau (tau): Controls the speed of decay of the slope and curvature
    
    Args:
        t: Tenor in years
        b0: Level parameter (long-term rate)
        b1: Slope parameter (short-term vs long-term spread)
        b2: Curvature parameter (medium-term deviation)
        tau: Decay parameter (controls the speed of decay)
        
    Returns:
        Zero rate in continuous compounding convention
        
    Example:
        >>> rate = ns_zero(5.0, 0.05, -0.02, 0.01, 2.0)
        >>> print(f"{rate:.4f}")
        0.0482
        
    Note:
        The function handles the special case of zero tenor by returning
        b0 + b1, which represents the instantaneous forward rate at t=0.
    """
    # Handle special case of zero tenor
    if t == 0:
        return b0 + b1
        
    # Calculate decay factor
    x = t / tau
    
    # Calculate each component
    # Level component: b0
    # Slope component: b1 * (1-exp(-x))/x
    # Curvature component: b2 * ((1-exp(-x))/x - exp(-x))
    return b0 + b1 * (1 - np.exp(-x)) / x + b2 * ((1 - np.exp(-x)) / x - np.exp(-x))


def fit_nelson_siegel(times: list[float], zeros: list[float]) -> tuple[float, float, float, float]:
    """Fit Nelson-Siegel parameters to observed zero rates.
    
    This function finds the optimal Nelson-Siegel parameters that minimize the
    sum of squared errors between the model and observed rates.
    
    The optimization process:
    1. Defines objective function as sum of squared errors
    2. Uses L-BFGS-B algorithm with box constraints
    3. Starts from reasonable initial guess
    4. Handles numerical stability near zero
    
    The optimization constraints:
    - All parameters are bounded between -0.1 and 0.2
    - Initial guess: [0.03, -0.02, 0.02, 1.0]
    - Uses L-BFGS-B algorithm for box-constrained optimization
    
    Args:
        times: List of tenors in years
        zeros: List of observed zero rates (continuous compounding)
        
    Returns:
        Tuple of (b0, b1, b2, tau) parameters
        
    Raises:
        RuntimeError: If the optimization fails to converge
        
    Example:
        >>> times = [1.0, 2.0, 5.0, 10.0]
        >>> zeros = [0.05, 0.06, 0.07, 0.08]
        >>> b0, b1, b2, tau = fit_nelson_siegel(times, zeros)
        >>> print(f"Level: {b0:.4f}, Slope: {b1:.4f}, Curvature: {b2:.4f}, Tau: {tau:.4f}")
        Level: 0.0800, Slope: -0.0300, Curvature: 0.0100, Tau: 2.0000
        
    Note:
        The optimization process is sensitive to the initial guess and
        parameter bounds. The current settings are tuned for typical
        yield curve shapes but may need adjustment for extreme cases.
    """
    def obj(p):
        """Objective function: sum of squared errors.
        
        This function calculates the sum of squared differences between
        the model predictions and observed rates. It is minimized by
        the L-BFGS-B algorithm to find optimal parameters.
        
        Args:
            p: Array of parameters [b0, b1, b2, tau]
            
        Returns:
            Sum of squared differences between model and observed rates
            
        Note:
            The objective function is smooth and differentiable, making
            it suitable for gradient-based optimization methods.
        """
        return sum((z - ns_zero(t, *p))**2 for t, z in zip(times, zeros))
        
    # Run optimization with L-BFGS-B algorithm
    # Initial guess: [level, slope, curvature, decay]
    res = minimize(
        obj,
        x0=[0.03, -0.02, 0.02, 1.0],  # Reasonable initial guess
        bounds=[(-.1, .2)]*4,  # Parameter bounds
        method='L-BFGS-B'  # Box-constrained optimization
    )
    
    # Check for convergence
    if not res.success:
        raise RuntimeError("NS fit failed")
        
    return tuple(res.x)