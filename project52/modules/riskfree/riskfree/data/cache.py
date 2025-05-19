# riskfree/data/cache.py
"""Caching layer for Treasury XML data fetching.

This module provides a caching mechanism for Treasury XML data to reduce
redundant HTTP requests. It uses Python's built-in LRU cache decorator
to store the most recently used XML responses.

The cache:
- Stores up to 60 most recent month's worth of XML data
- Automatically evicts least recently used entries when full
- Persists for the lifetime of the Python process

Key features:
    1. LRU (Least Recently Used) caching strategy
    2. Automatic cache eviction when full
    3. Process-level persistence
    4. Efficient memory usage
    5. Thread-safe operation

Note:
    The cache is implemented using Python's built-in lru_cache decorator,
    which provides an efficient and thread-safe caching mechanism. The
    cache size of 60 entries allows storing up to 5 years of monthly
    data while maintaining reasonable memory usage.
"""

from functools import lru_cache
import requests

@lru_cache(maxsize=60)
def get_month_xml(ym: str) -> bytes:
    """Fetch and cache Treasury XML data for a given month.
    
    This function fetches Treasury yield curve XML data for a specific month
    and caches the response to avoid redundant HTTP requests. The cache is
    keyed by the year-month string (e.g., "202403" for March 2024).
    
    The function implements:
    - LRU caching with automatic eviction
    - HTTP request with timeout
    - Error handling for failed requests
    - Efficient memory usage through bytes storage
    
    Args:
        ym: Year-month string in YYYYMM format (e.g., "202403" for March 2024)
        
    Returns:
        Raw XML content as bytes
        
    Raises:
        requests.RequestException: If the HTTP request fails
        
    Example:
        >>> xml = get_month_xml("202403")  # Fetch March 2024 data
        >>> len(xml)  # Size of XML response in bytes
        12345
        
    Note:
        The function uses Python's lru_cache decorator to implement
        efficient caching. The cache:
        1. Stores up to 60 most recent month's worth of data
        2. Automatically evicts least recently used entries when full
        3. Persists for the lifetime of the Python process
        4. Is thread-safe for concurrent access
    """
    url = ("https://home.treasury.gov/resource-center/data-chart-center/"
           "interest-rates/pages/xml")
    params = {"data": "daily_treasury_yield_curve",
              "field_tdr_date_value_month": ym}
    r = requests.get(url, params=params, timeout=(5, 30))
    r.raise_for_status()
    return r.content
