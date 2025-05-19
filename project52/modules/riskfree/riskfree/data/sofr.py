"""SOFR (Secured Overnight Financing Rate) data fetching module.

This module provides functionality to fetch SOFR fixings from the New York Federal
Reserve's official CSV feed. SOFR is a broad measure of the cost of borrowing cash
overnight collateralized by Treasury securities.

The module implements:
1. Direct CSV download from NY Fed's API
2. Date range filtering
3. Rate conversion from percentage to decimal
4. Error handling for empty data sets
"""

from __future__ import annotations
from datetime import date
import pandas as pd
import requests
import io

from ..config import get_logger

_LOG = get_logger(__name__)

# NY Fed's official SOFR CSV feed URL
NYFED_CSV = "https://markets.newyorkfed.org/api/rates/secured/sofr.csv"


def fetch_sofr(start: date, end: date | None = None) -> pd.Series:
    """Fetch SOFR daily fixings between start and end dates.
    
    This function downloads SOFR fixings from the NY Fed's official CSV feed
    and returns them as a pandas Series. The rates are converted from
    percentage to decimal format (e.g., 5.25% → 0.0525).
    
    Args:
        start: Start date (inclusive)
        end: End date (inclusive). Defaults to today if None.
        
    Returns:
        pandas.Series with:
        - Index: datetime64[ns] dates
        - Values: float SOFR rates in decimal form
        
    Raises:
        RuntimeError: If no data is found for the specified date range
        requests.RequestException: If the HTTP request fails
        
    Example:
        >>> start = date(2024, 1, 1)
        >>> end = date(2024, 1, 31)
        >>> sofr = fetch_sofr(start, end)
        >>> print(sofr.head())
        2024-01-01    0.0525
        2024-01-02    0.0525
        2024-01-03    0.0525
        ...
    """
    end = end or date.today()
    _LOG.debug("Downloading SOFR CSV from NY Fed…")
    text = requests.get(NYFED_CSV, timeout=10).text
    df = pd.read_csv(io.StringIO(text), parse_dates=["Effective Date"], index_col="Effective Date")
    df = df.loc[start:end, "SOFR"] / 100
    if df.empty:
        raise RuntimeError("SOFR series came back empty ➜ check dates.")
    return df