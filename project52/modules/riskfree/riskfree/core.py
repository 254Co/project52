"""Public interface for building and querying risk‑free curves.

This module provides the main RiskFreeCurve class for constructing and working with
risk-free rate curves. It handles data fetching, curve construction, and rate calculations.

The module implements a high-level interface that combines:
1. Data fetching from Treasury
2. Bootstrapping of zero-coupon yields
3. Optional Nelson-Siegel smoothing
4. Rate and discount factor calculations
"""
from __future__ import annotations
from datetime import date
from math import exp
from typing import Dict

import pandas as pd

from .config import get_logger
from .data.treasury import fetch_par_curve
from .curve.bootstrap import bootstrap_par_yields
from .model.nelson_siegel import fit_nelson_siegel, ns_zero

# Initialize module-level logger
_LOG = get_logger(__name__)

class RiskFreeCurve:
    """Build and query a risk‑free discount curve.
    
    This class provides a high-level interface for working with risk-free rate curves.
    It handles the entire pipeline from data fetching to rate calculations.
    
    The curve can be constructed in two modes:
    1. Raw bootstrapped curve (smooth=False): Preserves market data exactly
    2. Nelson-Siegel smoothed curve (smooth=True): Provides a smooth parametric fit
    
    All rates are in continuous compounding convention.
    
    Example:
        >>> curve = RiskFreeCurve(date.today())
        >>> rate = curve.spot(5.0)  # Get 5-year spot rate
        >>> df = curve.discount(5.0)  # Get 5-year discount factor
    """
    
    def __init__(self, trade_date: date, smooth: bool = True):
        """Initialize a risk-free curve.
        
        The initialization process:
        1. Fetches par yields from Treasury for the given date
        2. Bootstraps zero-coupon yields from par yields
        3. Optionally fits Nelson-Siegel model for smoothing
        
        Args:
            trade_date: The date for which to build the curve. If data is not available
                       for this date, the most recent available date will be used.
            smooth: Whether to apply Nelson-Siegel smoothing to the curve.
                   If False, uses raw bootstrapped rates.
                   
        Note:
            The curve is constructed by:
            1. Fetching par yields from Treasury
            2. Bootstrapping zero-coupon yields
            3. Optionally fitting Nelson-Siegel model
        """
        # Fetch par yields and bootstrap zero-coupon yields
        par = fetch_par_curve(trade_date)
        zeros = bootstrap_par_yields(par)
        
        # Apply Nelson-Siegel smoothing if requested
        if smooth:
            # Fit NS parameters to bootstrapped yields
            beta = fit_nelson_siegel(list(zeros), list(zeros.values()))
            # Replace bootstrapped yields with NS fitted yields
            zeros = {t: ns_zero(t, *beta) for t in zeros}
            
        # Store the zero-coupon yields
        self.zeros = zeros
        _LOG.info("Bootstrapped zeros for %s", trade_date)

    def spot(self, t: float) -> float:
        """Get the continuously compounded spot rate at tenor t.
        
        This method:
        1. Returns the exact rate if the tenor exists in the curve
        2. Uses linear interpolation for tenors between known points
        3. Raises ValueError for out-of-range tenors
        
        Args:
            t: Tenor in years. Must be within the range of available tenors.
            
        Returns:
            Continuously compounded spot rate (e.g., 0.05 for 5%)
            
        Raises:
            ValueError: If the requested tenor is out of range.
            
        Note:
            For tenors between available points, linear interpolation is used.
        """
        # Get sorted list of available tenors
        ts = sorted(self.zeros)
        
        # Return exact rate if tenor exists
        if t in self.zeros:
            return self.zeros[t]
            
        # Find nearest tenors for interpolation
        for i, u in enumerate(ts):
            if t < u:
                t0, t1 = ts[i-1], u
                z0, z1 = self.zeros[t0], self.zeros[t1]
                # Linear interpolation: r(t) = r0 + (r1-r0)*(t-t0)/(t1-t0)
                return z0 + (z1 - z0) * (t - t0) / (t1 - t0)
                
        # Raise error if tenor is out of range
        raise ValueError("Tenor out of range")

    def discount(self, t: float) -> float:
        """Calculate the discount factor for tenor t.
        
        The discount factor represents the present value of $1 received at time t.
        It is calculated using continuous compounding: DF(t) = exp(-r(t)*t)
        
        Args:
            t: Tenor in years. Must be within the range of available tenors.
            
        Returns:
            Discount factor exp(-r*t), where r is the spot rate.
            
        Note:
            The discount factor represents the present value of $1 received at time t.
        """
        return exp(-self.spot(t) * t)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the curve to a pandas DataFrame.
        
        This method creates a DataFrame with:
        - tenor as the index
        - zero-coupon rates
        - discount factors
        
        Returns:
            DataFrame with columns:
            - tenor: Tenor in years (index)
            - zero: Zero-coupon rate (continuous compounding)
            - discount: Discount factor exp(-r*t)
            
        Example:
            >>> df = curve.to_dataframe()
            >>> print(df)
            tenor    zero    discount
            1.0     0.05    0.951229
            2.0     0.06    0.886920
            ...
        """
        # Create DataFrame with tenors and zero rates
        df = pd.DataFrame({"tenor": list(self.zeros), "zero": list(self.zeros.values())})
        # Add discount factors
        df["discount"] = df["tenor"].apply(self.discount)
        # Set tenor as index
        return df.set_index("tenor")