#!/usr/bin/env python3
"""
fetch_rates.py

Fetch time-series data from the Bank of Japan Time-Series Data Search
and return it as a pandas DataFrame with:

  • Index named "Month" formatted as "YYYY-MM"
  • Columns ["EoM", "Average"]

This version:
  - Wraps HTML in StringIO for pandas.read_html
  - Coerces non-date rows to NaT, drops them
  - Renames data columns to EoM and Average
  - Formats index as YYYY-MM and names it "Month"
"""

import pandas as pd
import requests
from io import StringIO

def fetch_overnight_call_rate_monthly() -> pd.DataFrame:
    """
    Scrape the last <table> from a BOJ Time-series HTML page and return
    it as a DataFrame indexed by month (YYYY-MM), with columns ['EoM', 'Average'].

    Parameters
    ----------
    url : str
        Full URL of the BOJ time-series HTML page.

    Returns
    -------
    pd.DataFrame
        Indexed by Month (string YYYY-MM), with columns ['EoM', 'Average'].
    """
    url = "https://www.stat-search.boj.or.jp/ssi/mtshtml/fm02_m_1_en.html"
    
    r = requests.get(url)
    r.raise_for_status()

    # wrap text to silence FutureWarning
    dfs = pd.read_html(StringIO(r.text))
    if not dfs:
        raise RuntimeError(f"No tables found at {url!r}")

    df = dfs[-1].copy()

    # parse first column as YYYY/MM, coerce errors → NaT, drop them
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], format="%Y/%m", errors="coerce")
    df = df.dropna(subset=[date_col])

    # set the datetime index
    df = df.set_index(date_col).sort_index(ascending=False)

    # rename data columns
    if len(df.columns) >= 2:
        df.columns = ["EoM", "Average"]
    else:
        raise RuntimeError("Expected at least two data columns to rename to ['EoM','Average'].")

    # format index as "YYYY-MM" strings and name it "Month"
    df.index = df.index.strftime("%Y-%m")
    df.index.name = "Month"

    return df

def fetch_overnight_call_rate_daily() -> pd.DataFrame:
    """
    Scrape the last <table> from a BOJ Time-series HTML page and return
    it as a DataFrame indexed by month (YYYY-MM), with columns ['EoM', 'Average'].

    Parameters
    ----------
    url : str
        Full URL of the BOJ time-series HTML page.

    Returns
    -------
    pd.DataFrame
        Indexed by Month (string YYYY-MM), with columns ['EoM', 'Average'].
    """
    url = "https://www.stat-search.boj.or.jp/ssi/mtshtml/fm01_d_1_en.html"
    
    r = requests.get(url)
    r.raise_for_status()

    # wrap text to silence FutureWarning
    dfs = pd.read_html(StringIO(r.text))
    if not dfs:
        raise RuntimeError(f"No tables found at {url!r}")

    df = dfs[-1].copy()
    # parse first column as YYYY/MM, coerce errors → NaT, drop them
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=[date_col])

    # set the datetime index
    df = df.set_index(date_col).sort_index(ascending=False)

    # rename data columns
    if len(df.columns) >= 1:
        df.columns = ["Average"]
    else:
        raise RuntimeError("Expected at least one data columns to rename to ['Average'].")

    # format index as "YYYY-MM" strings and name it "Month"
    df.index = df.index.strftime("%Y-%m-%d")
    df.index.name = "Date"

    return df

if __name__ == "__main__":
    # Example: BOJ Uncollateralized Overnight Call Rate (End-of-Month & Average)
    df = fetch_overnight_call_rate_monthly()
    print(df)
    df = fetch_overnight_call_rate_daily()
    print(df)
