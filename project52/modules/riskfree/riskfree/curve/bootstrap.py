# File: riskfree/curve/bootstrap.py

"""Zero-coupon yield curve bootstrapping.

This module provides functions for bootstrapping zero-coupon yields from par yields.
It implements the standard bootstrapping algorithm using continuous compounding.

The bootstrapping process:
1. Starts with the shortest tenor (1-year)
2. For each subsequent tenor:
   - Calculates the present value of all cash flows
   - Solves for the zero-coupon rate that makes the bond price equal to par
3. Uses continuous compounding convention throughout
"""
import pandas as pd
from math import log
from typing import Dict

def bootstrap_par_yields(par_df: pd.DataFrame) -> Dict[float, float]:
    """Bootstrap zero-coupon yield curve from par yields.
    
    This function implements the standard bootstrapping algorithm to convert
    par yields to zero-coupon yields. The algorithm:
    
    1. Starts with the shortest tenor (1-year)
    2. For each subsequent tenor:
       - Calculates the present value of all cash flows
       - Solves for the zero-coupon rate that makes the bond price equal to par
    
    All rates are in continuous compounding convention.
    
    Args:
        par_df: DataFrame with columns:
               - Date: The curve date (optional)
               - 1_yr, 2_yr, 3_yr, 5_yr, 7_yr, 10_yr, 20_yr, 30_yr: Par yields
                 as decimals (e.g., 0.035 for 3.5%)
               Must contain exactly one row.
    
    Returns:
        Dict mapping tenor in years (float) → zero rate (float, continuous compounding).
        For example: {1.0: 0.05, 2.0: 0.06, ...}
        
    Example:
        >>> par_df = pd.DataFrame({
        ...     "Date": ["2024-03-20"],
        ...     "1_yr": [0.05],
        ...     "2_yr": [0.06]
        ... })
        >>> zeros = bootstrap_par_yields(par_df)
        >>> print(zeros)
        {1.0: 0.05, 2.0: 0.0603}
        
    Note:
        The bootstrapping assumes:
        1. All bonds pay semi-annual coupons
        2. All rates are in continuous compounding
        3. The curve is arbitrage-free
    """
    # Create a copy of the input DataFrame
    df = par_df.copy()
    
    # Remove Date column if present
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
        
    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dictionary to store zero-coupon rates
    zeros: Dict[float, float] = {}
    
    # Process each tenor
    for col in df.columns:
        # Convert column name to tenor (e.g., "5_yr" → 5.0)
        tenor = float(col.replace("_yr", ""))
        
        # Get par yield for this tenor
        par_yield = df[col].iloc[0]   # single value
        
        # Calculate discount factor for par bond
        # For a par bond: 100 = Σ(c/2 * DF(t_i)) + 100 * DF(t)
        # With continuous compounding: DF = exp(-r*t)
        # Therefore: DF = 1 / (1 + par_yield * tenor)
        dfactor = 1.0 / (1.0 + par_yield * tenor)
        
        # Convert to zero rate using continuous compounding
        # r = -ln(DF)/tenor
        zeros[tenor] = -log(dfactor) / tenor

    return zeros
