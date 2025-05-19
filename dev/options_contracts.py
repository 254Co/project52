#!/usr/bin/env python3
"""
option_fetcher.py

Fetch all option contracts for a given ticker via Yahoo Finance,
first via query2 (no crumb), then falling back to the crumb-based query1 if needed.
"""

import logging
from typing import List, Tuple

import pandas as pd
import requests


# Realistic User-Agent so Yahoo serves us JSON
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _get_crumb_and_session(ticker: str) -> Tuple[str, requests.Session]:
    """
    Hit the public options page to get cookies + crumb token.
    Returns (crumb, session) or raises on failure.
    """
    import re

    session = requests.Session()
    session.headers.update(HEADERS)
    session.headers["Referer"] = f"https://finance.yahoo.com/quote/{ticker}/options?p={ticker}"
    url = session.headers["Referer"]
    resp = session.get(url, timeout=10)
    resp.raise_for_status()

    m = re.search(r'"CrumbStore":\s*\{\s*"crumb"\s*:\s*"([^"]+)"\s*\}', resp.text)
    if not m:
        raise RuntimeError("Could not extract Yahoo Finance crumb")
    crumb = m.group(1).encode("ascii").decode("unicode_escape")
    logging.info("Extracted crumb token")
    return crumb, session


def _fetch_via_query2(ticker: str, session: requests.Session) -> List[dict]:
    """
    Try the new query2 endpoint. Returns list of option blocks (each with
    'expirationDate' and 'options'), or raises if unauthorized or invalid JSON.
    """
    url = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}"
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    result = data["optionChain"]["result"][0]
    return result.get("options", [])


def _fetch_via_query1(ticker: str, crumb: str, session: requests.Session) -> List[dict]:
    """
    Fallback to query1 with crumb. Returns same shape as _fetch_via_query2.
    """
    base = f"https://query1.finance.yahoo.com/v7/finance/options/{ticker}"
    resp = session.get(base, params={"crumb": crumb}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    result = data["optionChain"]["result"][0]
    expirations = result.get("expirationDates", [])
    options_blocks = []
    for exp_ts in expirations:
        r2 = session.get(base, params={"date": exp_ts, "crumb": crumb}, timeout=10)
        r2.raise_for_status()
        opts = r2.json()["optionChain"]["result"][0]["options"][0]
        opts["expirationDate"] = exp_ts
        options_blocks.append(opts)
    return options_blocks


def fetch_option_contracts(ticker: str) -> pd.DataFrame:
    """
    Fetch all option contracts for the specified ticker.
    Returns an empty DataFrame on any unrecoverable error.
    """
    ticker = ticker.upper()
    configure_logging()

    # Prepare a session with headers
    session = requests.Session()
    session.headers.update(HEADERS)
    session.headers["Referer"] = f"https://finance.yahoo.com/quote/{ticker}/options?p={ticker}"

    # 1) Try query2 (no crumb)
    try:
        logging.info("Trying query2 for %s", ticker)
        blocks = _fetch_via_query2(ticker, session)
    except Exception as e:
        logging.warning("query2 failed (%s), falling back to crumb-based query1", e)
        # 2) Fallback to crumb-based query1
        try:
            crumb, session = _get_crumb_and_session(ticker)
            blocks = _fetch_via_query1(ticker, crumb, session)
        except Exception as e2:
            logging.error("Both query2 and query1 failed for %s: %s", ticker, e2)
            return pd.DataFrame()

    # 3) Normalize into DataFrames
    all_frames: List[pd.DataFrame] = []
    for block in blocks:
        exp_ts = block.get("expirationDate")
        exp_dt = pd.to_datetime(exp_ts, unit="s", utc=True)
        for side in ("calls", "puts"):
            df = pd.DataFrame(block.get(side, []))
            if df.empty:
                continue
            df["expiration"] = exp_dt
            df["type"] = "call" if side == "calls" else "put"
            all_frames.append(df)

    if not all_frames:
        logging.warning("No option data found for %s", ticker)
        return pd.DataFrame()

    df_all = pd.concat(all_frames, ignore_index=True)
    df_all.columns = [col.strip() for col in df_all.columns]
    return df_all


def main() -> None:
    # ◀◀◀ Set your ticker here:
    ticker = "AAPL"

    df = fetch_option_contracts(ticker)
    if df.empty:
        logging.info("No option contracts for %s", ticker)
        return

    print(df.head(10).to_string(index=False))
    logging.info("Fetched %d total contracts for %s", len(df), ticker)


if __name__ == "__main__":
    main()
