# -------------------------------------------------
# data_sources/treasury.py  – FRED daily Treasury yields (e.g. 1m‑30y)
# -------------------------------------------------
import os
import requests
import pandas as pd
from typing import List

from .base import BaseDataSource

TREASURY_SERIES = [
    ("DGS1MO", 1 / 12),   # 1‑month
    ("DGS3MO", 0.25),
    ("DGS6MO", 0.5),
    ("DGS1", 1),
    ("DGS2", 2),
    ("DGS3", 3),
    ("DGS5", 5),
    ("DGS7", 7),
    ("DGS10", 10),
    ("DGS20", 20),
    ("DGS30", 30),
]


class TreasuryDataSource(BaseDataSource):
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_data(self, start_date, end_date) -> pd.DataFrame:
        cache_name = f"treasury_{start_date}_{end_date}"
        if (cached := self.try_cache(cache_name)) is not None:
            return cached

        rows: List[pd.DataFrame] = []
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise EnvironmentError("FRED_API_KEY env var not set")

        for series_id, mat in TREASURY_SERIES:
            r = requests.get(
                self.BASE_URL,
                params={
                    "series_id": series_id,
                    "api_key": api_key,
                    "file_type": "json",
                    "observation_start": start_date,
                    "observation_end": end_date,
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()["observations"]
            df = pd.DataFrame(data)[["date", "value"]]
            df["date"] = pd.to_datetime(df["date"])
            df["yield"] = pd.to_numeric(df["value"], errors="coerce") / 100  # pct → decimal
            df["maturity"] = mat
            rows.append(df[["date", "maturity", "yield"]])

        out = pd.concat(rows).dropna()
        self.validate_data(out)
        self.cache_data(out, cache_name)
        return out