"""Fetch SOFR fixings (NY Fed CSV)."""
from __future__ import annotations
from datetime import date
import pandas as pd
import requests
import io

from ..config import get_logger

_LOG = get_logger(__name__)
NYFED_CSV = "https://markets.newyorkfed.org/api/rates/secured/sofr.csv"


def fetch_sofr(start: date, end: date | None = None) -> pd.Series:
    """Return SOFR daily series between *start* and *end* (inclusive)."""
    end = end or date.today()
    _LOG.debug("Downloading SOFR CSV from NY Fed…")
    text = requests.get(NYFED_CSV, timeout=10).text
    df = pd.read_csv(io.StringIO(text), parse_dates=["Effective Date"], index_col="Effective Date")
    df = df.loc[start:end, "SOFR"] / 100
    if df.empty:
        raise RuntimeError("SOFR series came back empty ➜ check dates.")
    return df