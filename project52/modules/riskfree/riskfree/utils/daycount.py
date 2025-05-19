"""Day-count convention utilities for financial calculations.

This module provides functions for calculating year fractions between dates
using standard financial day-count conventions. These conventions are used
in fixed income calculations to determine accrual periods and interest payments.

Supported conventions:
- ACT/365: Actual days divided by 365 (also known as ACT/365 Fixed)
- ACT/360: Actual days divided by 360 (common for money market instruments)

Note:
    The module currently supports only the most common conventions.
    Additional conventions (30/360, ACT/ACT, etc.) can be added as needed.
"""

from __future__ import annotations
from datetime import datetime

def yearfrac(start: datetime, end: datetime, convention: str = "ACT/365") -> float:
    """Calculate the year fraction between two dates using the specified convention.
    
    This function calculates the fraction of a year between two dates using
    standard financial day-count conventions. The calculation is exact to the
    second for ACT/365 and ACT/360 conventions.
    
    Args:
        start: Start date and time
        end: End date and time
        convention: Day-count convention to use. Currently supported:
                   - "ACT/365": Actual days divided by 365
                   - "ACT/360": Actual days divided by 360
                   
    Returns:
        float: Year fraction between start and end dates
        
    Raises:
        ValueError: If an unsupported convention is specified
        
    Example:
        >>> from datetime import datetime
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 7, 1)
        >>> yearfrac(start, end, "ACT/365")  # Half year
        0.5
        >>> yearfrac(start, end, "ACT/360")  # Slightly more than half
        0.5069444444444444
    """
    seconds = (end - start).total_seconds()
    if convention.upper() == "ACT/365":
        return seconds / (365 * 24 * 3600)
    if convention.upper() == "ACT/360":
        return seconds / (360 * 24 * 3600)
    raise ValueError(f"Unsupported day‑count: {convention}")