# -------------------------------------------------
# data_sources/sonia.py (Bank of England JSON)
# -------------------------------------------------

import requests
import pandas as pd
from .base import BaseDataSource

class SONIADataSource(BaseDataSource):
    BASE_URL = "https://www.bankofengland.co.uk/boeapi/sonia"

    def fetch_data(self, start_date, end_date):
        cache_name = f"sonia_{start_date}_{end_date}"
        if (c := self.try_cache(cache_name)) is not None:
            return c
        params = {"from": start_date, "to": end_date, "format": "json"}
        r = requests.get(self.BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()["entries"]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["effectiveDate"])
        df["yield"] = pd.to_numeric(df["rate"], errors="coerce") / 100
        df["maturity"] = 1 / 365
        self.validate_data(df)
        self.cache_data(df, cache_name)
        return df[["date", "maturity", "yield"]]

