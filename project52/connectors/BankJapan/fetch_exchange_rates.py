#!/usr/bin/env python3
"""
fetch_rates.py

Fetch time-series data from the Bank of Japan Time-Series Data Search
and return it as a pandas DataFrame with:


This version:
  - Wraps HTML in StringIO for pandas.read_html
  - Coerces non-date rows to NaT, drops them
"""

import pandas as pd
import requests
from io import StringIO



def fetch_tokyo_market_interbank_rates_daily() -> pd.DataFrame:

    url = "https://www.stat-search.boj.or.jp/ssi/mtshtml/fm08_d_1_en.html"
    
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
    if len(df.columns) >= 2:
        df.columns = ["Spot", "Central"]
    else:
        raise RuntimeError("Expected at least one data columns to rename to ['Average'].")

    # format index as "YYYY-MM" strings and name it "Month"
    df.index = df.index.strftime("%Y-%m-%d")
    df.index.name = "Date"

    return df

def fetch_effective_exchange_rates_monthly() -> pd.DataFrame:

    url = "https://www.stat-search.boj.or.jp/ssi/mtshtml/fm09_m_1_en.html"
    
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
        df.columns = ["Nominal", "Real"]
    else:
        raise RuntimeError("Expected at least one data columns to rename to ['Average'].")

    # format index as "YYYY-MM" strings and name it "Month"
    df.index = df.index.strftime("%Y-%m")
    df.index.name = "Month"

    return df

if __name__ == "__main__":
    # Example: BOJ Uncollateralized Overnight Call Rate (End-of-Month & Average)
    df = fetch_tokyo_market_interbank_rates_daily()
    print(df)
    df = fetch_effective_exchange_rates_monthly()
    print(df)
