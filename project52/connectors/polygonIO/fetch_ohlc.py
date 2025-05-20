#!/usr/bin/env python3
"""
fetch_polygon_ohlc_fast.py

Provides a single function fetch_polygon_ohlc_fast(ticker, api_key) which
returns a pandas DataFrame of minute‐bar OHLC from first trade until today,
using a pooled requests.Session, gzip compression, and orjson for speed.
"""
from __future__ import annotations
import datetime as dt
import urllib.parse

import requests
import orjson
import pandas as pd

# pooled session with gzip & UA
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "polygon-ohlc-fetch/1.0 (+https://github.com/yourrepo)",
    "Accept-Encoding": "gzip, deflate, br",
})

def fetch_ohlc_min(ticker: str, api_key: str) -> pd.DataFrame:
    """
    Fetches the full OHLC minute bars history for a given ticker from Polygon.io.

    Parameters
    ----------
    ticker : str
        The equity ticker symbol, e.g. "AAPL".
    api_key : str
        Your Polygon.io API key.

    Returns
    -------
    pd.DataFrame
        Indexed by UTC datetime, columns = [open, high, low, close, volume, transactions].
    """
    REF_URL = "https://api.polygon.io/v3/reference/tickers/{ticker}"
    AGG_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}"

    def _first_trade_date(t: str) -> dt.date:
        resp = _SESSION.get(
            REF_URL.format(ticker=t),
            params={"apiKey": api_key},
            timeout=10
        )
        resp.raise_for_status()
        date_str = orjson.loads(resp.content)["results"].get("list_date")
        if date_str:
            return dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        # fallback to 10 years ago if not listed
        return dt.date.today() - dt.timedelta(days=3650)

    def _append_key(next_url: str | None) -> str | None:
        if not next_url:
            return None
        parts = list(urllib.parse.urlparse(next_url))
        qs = urllib.parse.parse_qs(parts[4], keep_blank_values=True)
        qs["apiKey"] = [api_key]
        parts[4] = urllib.parse.urlencode(qs, doseq=True)
        return urllib.parse.urlunparse(parts)

    start_date = _first_trade_date(ticker)
    end_date = dt.date.today()
    url = AGG_URL.format(ticker=ticker, start=start_date, end=end_date)
    params = {
        "adjusted": "true",
        "sort": "desc",
        "limit": 50_000,
        "apiKey": api_key
    }

    records: list[dict] = []
    while url:
        r = _SESSION.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = orjson.loads(r.content)
        records.extend(js.get("results", []))
        url, params = _append_key(js.get("next_url")), None

    if not records:
        raise RuntimeError(f"No bars returned for {ticker}")

    cols = ["t", "o", "h", "l", "c", "v", "n"]  # timestamp, open, high, low, close, volume, txn count
    df = pd.DataFrame.from_records(records, columns=cols)
    df = df.rename(columns={
        "t": "date",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "transactions"
    })
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    return df.set_index("date")


def fetch_ohlc_day(ticker: str, api_key: str) -> pd.DataFrame:
    """
    Fetches the full OHLC day bars history for a given ticker from Polygon.io.

    Parameters
    ----------
    ticker : str
        The equity ticker symbol, e.g. "AAPL".
    api_key : str
        Your Polygon.io API key.

    Returns
    -------
    pd.DataFrame
        Indexed by UTC datetime, columns = [open, high, low, close, volume, transactions].
    """
    REF_URL = "https://api.polygon.io/v3/reference/tickers/{ticker}"
    AGG_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"

    def _first_trade_date(t: str) -> dt.date:
        resp = _SESSION.get(
            REF_URL.format(ticker=t),
            params={"apiKey": api_key},
            timeout=10
        )
        resp.raise_for_status()
        date_str = orjson.loads(resp.content)["results"].get("list_date")
        if date_str:
            return dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        # fallback to 10 years ago if not listed
        return dt.date.today() - dt.timedelta(days=3650)

    def _append_key(next_url: str | None) -> str | None:
        if not next_url:
            return None
        parts = list(urllib.parse.urlparse(next_url))
        qs = urllib.parse.parse_qs(parts[4], keep_blank_values=True)
        qs["apiKey"] = [api_key]
        parts[4] = urllib.parse.urlencode(qs, doseq=True)
        return urllib.parse.urlunparse(parts)

    start_date = _first_trade_date(ticker)
    end_date = dt.date.today()
    url = AGG_URL.format(ticker=ticker, start=start_date, end=end_date)
    params = {
        "adjusted": "true",
        "sort": "desc",
        "limit": 50_000,
        "apiKey": api_key
    }

    records: list[dict] = []
    while url:
        r = _SESSION.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = orjson.loads(r.content)
        records.extend(js.get("results", []))
        url, params = _append_key(js.get("next_url")), None

    if not records:
        raise RuntimeError(f"No bars returned for {ticker}")

    cols = ["t", "o", "h", "l", "c", "v", "n"]  # timestamp, open, high, low, close, volume, txn count
    df = pd.DataFrame.from_records(records, columns=cols)
    df = df.rename(columns={
        "t": "date",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "transactions"
    })
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    return df.set_index("date")


# ─── Example Usage ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # replace with your own key
    
    df = fetch_ohlc_day("AAPL", "cqWpEROd6Kq0Q0zihGGYEosjAi4IPd_w")
    print(df.head(), "\n")
    print(f"{len(df):,} day bars fetched "
          f"({df.index[0].date()} → {df.index[-1].date()}) for AAPL")
    
    df = fetch_ohlc_min("AAPL", "cqWpEROd6Kq0Q0zihGGYEosjAi4IPd_w")
    print(df.head(), "\n")
    print(f"{len(df):,} minute bars fetched "
          f"({df.index[0].date()} → {df.index[-1].date()}) for AAPL")
