# -------------------------------------------------
# data_sources/repo.py (NY Fed – repo operations)
# -------------------------------------------------
import requests
import pandas as pd
from .base import BaseDataSource

class RepoDataSource(BaseDataSource):
    BASE_URL = "https://markets.newyorkfed.org/api/operations/repo"  # JSON daily ops

    def fetch_data(self, start_date, end_date):
        cache_name = f"repo_{start_date}_{end_date}"
        if (c := self.try_cache(cache_name)) is not None:
            return c
        params = {"start": start_date, "end": end_date}
        r = requests.get(self.BASE_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()["repo"]
        # Average weighted rate of operations
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["operationDate"])
        df["yield"] = pd.to_numeric(df["weightedAvgRate"], errors="coerce") / 100
        df["maturity"] = 1 / 365
        self.validate_data(df)
        self.cache_data(df, cache_name)
        return df[["date", "maturity", "yield"]]


