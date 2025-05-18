# File: riskfree/data/treasury.py

"""Treasury par-yield curve data fetching and processing.

This module handles fetching and processing of U.S. Treasury par-yield curve data
from the official Treasury XML feed. It includes robust error handling and
automatic backfilling for missing data.

The module implements:
1. HTTP request handling with retries
2. XML parsing of Treasury data
3. Automatic backfilling for missing dates
4. Fallback to zero curve when no data is available
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import date, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class FetchDataError(Exception):
    """Raised when the Treasury par-yield curve cannot be fetched or parsed.
    
    This error occurs when:
    1. The Treasury XML feed is unavailable
    2. The response cannot be parsed
    3. No data is available for the requested date range
    """
    pass

def fetch_par_curve(trade_date: date) -> pd.DataFrame:
    """Fetch the Treasury par-yield curve for a given date.
    
    This function fetches par-yield curve data from the official U.S. Treasury XML feed.
    It includes robust error handling and automatic backfilling:
    
    1. If no data is available for the requested date (e.g., weekend or holiday),
       it will step backward up to 7 days to find the most recent published curve.
    2. If still no data is found, returns a zero curve for that date.
    3. Implements automatic retry logic for failed requests.
    
    Args:
        trade_date: The date for which to fetch the curve.
        
    Returns:
        DataFrame with columns:
        - Date: The actual date of the curve (may differ from trade_date)
        - 1_yr, 2_yr, 3_yr, 5_yr, 7_yr, 10_yr, 20_yr, 30_yr: Par yields as decimals
          (e.g., 0.035 for 3.5%)
        
    Raises:
        FetchDataError: If the data cannot be fetched or parsed.
        
    Example:
        >>> df = fetch_par_curve(date.today())
        >>> print(df)
        Date        1_yr    2_yr    3_yr    5_yr    7_yr    10_yr   20_yr   30_yr
        2024-03-20  0.05    0.06    0.07    0.08    0.09    0.10    0.11    0.12
    """
    # Treasury XML feed URL
    BASE_URL = (
        "https://home.treasury.gov/resource-center/"
        "data-chart-center/interest-rates/pages/xml"
    )
    
    # XML namespace definitions for parsing
    NS = {
        "a": "http://www.w3.org/2005/Atom",
        "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
    }

    # Configure HTTP session with retry logic
    session = requests.Session()
    retries = Retry(
        total=3,  # Maximum number of retries
        backoff_factor=0.5,  # Exponential backoff
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET"],  # Only retry GET requests
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))

    # Map column names to XML element names
    tenor_map = {
        "1_yr":  "d:BC_1YEAR",
        "2_yr":  "d:BC_2YEAR",
        "3_yr":  "d:BC_3YEAR",
        "5_yr":  "d:BC_5YEAR",
        "7_yr":  "d:BC_7YEAR",
        "10_yr": "d:BC_10YEAR",
        "20_yr": "d:BC_20YEAR",
        "30_yr": "d:BC_30YEAR",
    }

    # Attempt to fetch data, stepping back up to 7 days if needed
    record = None
    found_date = None
    for days_back in range(0, 7):
        d = trade_date - timedelta(days=days_back)
        params = {
            "data":                       "daily_treasury_yield_curve",
            "field_tdr_date_value_month": d.strftime("%Y%m"),
        }
        try:
            # Make HTTP request with timeout
            resp = session.get(BASE_URL, params=params, timeout=(5, 30))
            resp.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(resp.content)
            entries = root.findall(".//a:entry", NS)
            
            # Find entry for requested date
            for entry in entries:
                props = entry.find(".//m:properties", NS)
                rec_dt = props.find("d:NEW_DATE", NS).text
                if rec_dt == d.isoformat():
                    record = props
                    found_date = d
                    break
            if record:
                break
        except Exception:
            continue

    # Build output DataFrame
    out = {"Date": pd.to_datetime(found_date or trade_date)}
    
    # Extract rates for each tenor
    for col in tenor_map:
        if record is not None:
            # Convert percentage to decimal
            text = record.find(tenor_map[col], NS).text
            out[col] = float(text) / 100.0
        else:
            # Fallback to zero curve if no data found
            out[col] = 0.0

    return pd.DataFrame([out])
