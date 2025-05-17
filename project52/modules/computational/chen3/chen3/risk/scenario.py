# -------------------- risk/scenario.py --------------------
"""
Scenario Shock Generation and Application Module

This module provides functionality for creating and applying scenario shocks to
simulated paths in the three-factor Chen model. It allows for simultaneous
shocks to multiple risk factors, enabling stress testing and scenario analysis.

The module supports shocks to:
1. Interest rates (additive shift in basis points)
2. Volatility (relative shift in variance)
3. Equity prices (relative shift in price)

This is particularly useful for:
- Stress testing
- Scenario analysis
- Risk factor sensitivity analysis
- PnL attribution
"""

import numpy as np
from typing import Dict, Any


class Scenario:
    """
    Represents a simultaneous shock to multiple risk factors in the model.
    
    This class encapsulates a set of shocks to be applied to simulated paths,
    allowing for stress testing and scenario analysis. The shocks can be applied
    to interest rates, volatility, and equity prices simultaneously.
    
    The class is designed to:
    1. Define scenario shocks for each risk factor
    2. Apply shocks to simulated paths
    3. Support both additive (rates) and relative (vol/equity) shifts
    
    Attributes:
        rate_shift (float): Additive shift in short rate (in basis points)
        vol_shift (float): Relative shift in variance (as a percentage)
        equity_shift (float): Relative shift in equity price (as a percentage)
    
    Example:
        >>> scenario = Scenario(rate_shift=10.0, vol_shift=0.2, equity_shift=-0.1)
        >>> shocked_paths = scenario.apply(simulated_paths)
    
    Notes:
        - Rate shifts are additive (e.g., +10bp)
        - Volatility and equity shifts are relative (e.g., +20%, -10%)
        - All shifts can be positive or negative
        - Default shifts are zero (no change)
    """
    
    def __init__(self,
                 rate_shift: float = 0.0,
                 vol_shift: float = 0.0,
                 equity_shift: float = 0.0):
        """
        Initialize a scenario with specified risk factor shifts.
        
        Args:
            rate_shift (float): Additive shift in short rate (in basis points)
            vol_shift (float): Relative shift in variance (as a percentage)
            equity_shift (float): Relative shift in equity price (as a percentage)
        
        Notes:
            - Rate shifts are in basis points (e.g., 10.0 = +10bp)
            - Volatility and equity shifts are in decimal form (e.g., 0.2 = +20%)
            - All shifts default to zero (no change)
        """
        self.rate_shift = rate_shift
        self.vol_shift = vol_shift
        self.equity_shift = equity_shift

    def apply(self, paths: np.ndarray) -> np.ndarray:
        """
        Apply scenario shocks to simulated paths.
        
        This method applies the defined scenario shocks to a set of simulated
        paths, creating a new array of shocked paths. The shocks are applied
        as follows:
        - Interest rates: Additive shift (in basis points)
        - Variance: Relative shift (multiply by 1 + shift)
        - Equity prices: Relative shift (multiply by 1 + shift)
        
        Args:
            paths (np.ndarray): Array of simulated paths with shape
                              (n_paths, n_steps+1, 3) where the last dimension
                              represents [S, v, r] (equity, variance, rate)
        
        Returns:
            np.ndarray: New array of shocked paths with same shape as input
        
        Example:
            >>> paths = np.array([[[100.0, 0.04, 0.02], ...]])
            >>> scenario = Scenario(rate_shift=10.0, vol_shift=0.2, equity_shift=-0.1)
            >>> shocked = scenario.apply(paths)
            >>> # shocked[:,:,2] = paths[:,:,2] + 0.001  # +10bp
            >>> # shocked[:,:,1] = paths[:,:,1] * 1.2    # +20% vol
            >>> # shocked[:,:,0] = paths[:,:,0] * 0.9    # -10% equity
        
        Notes:
            - Returns a new array, does not modify input
            - Preserves the shape and structure of input paths
            - Applies shocks to all paths and time steps
        """
        shocked = paths.copy()
        # Add shift in short rate (in absolute terms)
        shocked[:,:,2] += self.rate_shift
        # Multiply variance by (1 + vol_shift)
        shocked[:,:,1] *= (1 + self.vol_shift)
        # Multiply stock price by (1 + equity_shift)
        shocked[:,:,0] *= (1 + self.equity_shift)
        return shocked