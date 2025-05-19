# -------------------------------------------------
# data_sources/sofr.py  – SOFR (overnight)
# -------------------------------------------------
import os
import pandas as pd
import requests
from .base import BaseDataSource

class SOFRDataSource(BaseDataSource):
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_data(self, start_date, end_date):
        cache_name = f"sofr_{start_date}_{end_date}"
        if (c := self.try_cache(cache_name)) is not None:
            return c
        r = requests.get(
            self.BASE_URL,
            params={
                "series_id": "SOFR",
                "api_key": os.getenv("FRED_API_KEY"),
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
        df["yield"] = pd.to_numeric(df["value"], errors="coerce") / 100
        df["maturity"] = 1 / 365  # overnight ≈ 1‑day
        self.validate_data(df)
        self.cache_data(df, cache_name)
        return df[["date", "maturity", "yield"]]

