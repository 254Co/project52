import logging
import re
import time
from io import StringIO
from typing import Dict

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# ─── Constants ────────────────────────────────────────────────────────────────
_WIKI_URL     = "https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2#Officially_assigned_code_elements"
_FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED free API: 120 requests per minute → one request every 0.5s
RATE_LIMIT_PER_MINUTE = 120
_REQUEST_DELAY = 60.0 / RATE_LIMIT_PER_MINUTE

# ─── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def fetch_iso3166_codes() -> pd.DataFrame:
    """
    Fetch the latest ISO 3166-1 alpha-2 codes and country names from Wikipedia.
    Returns a DataFrame with columns ['code', 'name'].
    """
    logger.info(f"Fetching ISO codes from {_WIKI_URL}")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    resp = session.get(_WIKI_URL, timeout=10)
    resp.raise_for_status()
    time.sleep(_REQUEST_DELAY)

    tables = pd.read_html(StringIO(resp.text))
    if not tables or len(tables) < 5:
        raise RuntimeError("Could not find the ISO codes table on Wikipedia")

    df = tables[4].copy()
    for col in ("Year", "ccTLD", "Notes"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df.rename(columns={
        "Code": "code",
        "Country name (using title case)": "name"
    }, inplace=True)

    # drop missing/invalid codes and keep only A–Z two-letter codes
    df = df.dropna(subset=["code"])
    mask = df["code"].str.fullmatch(r"[A-Z]{2}")
    df = df[mask.fillna(False)].reset_index(drop=True)

    logger.info(f"Retrieved {len(df)} ISO codes")
    return df[["code", "name"]]


def fetch_global_gdp_usd_current(api_key: str) -> pd.DataFrame:
    """
    Fetch GDP time-series from FRED for every ISO country code.
    Series IDs are MKTGDP{code}A646NWDB. Ensures compliance with 120 RPM limit.
    """
    # prepare session with retry/backoff
    session = requests.Session()
    retry_policy = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_policy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    iso_df = fetch_iso3166_codes()
    data: Dict[str, pd.Series] = {}
    pattern = "MKTGDP{code}A646NWDB"

    for _, row in iso_df.iterrows():
        code, country = row["code"], row["name"]
        series_id = pattern.format(code=code)
        logger.info(f"Fetching GDP for {country} ({series_id})")

        params = {
            "series_id": series_id,
            "api_key":    api_key,
            "file_type":  "json"
        }
        try:
            resp = session.get(_FRED_OBS_URL, params=params, timeout=10)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
        except Exception as e:
            logger.warning(f"  → failed for {series_id}: {e}")
            data[country] = pd.Series(dtype="float64")
            time.sleep(_REQUEST_DELAY)
            continue

        # throttle to comply with rate limit
        time.sleep(_REQUEST_DELAY)

        df = pd.DataFrame(obs)
        if {"date", "value"}.issubset(df.columns):
            df["date"]  = pd.to_datetime(df["date"],  errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date")
            data[country] = df["value"].sort_index()
        else:
            logger.warning(f"  → unexpected payload for {series_id}")
            data[country] = pd.Series(dtype="float64")

    result = pd.DataFrame(data)
    result.sort_index(ascending=False, inplace=True)
    return result




if __name__ == "__main__":
    api_key = '817f445ac3ebd65ac75be2af96b5b90d'
    df = fetch_global_gdp_usd_current(api_key)
    print(df)
    
