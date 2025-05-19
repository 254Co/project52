# -------------------------------------------------
# data_sources/base.py
# -------------------------------------------------
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
CACHE_DIR.mkdir(exist_ok=True)


class BaseDataSource(ABC):
    """Abstract interface every data‑source class must fulfil."""

    @abstractmethod
    def fetch_data(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        """Return *daily* dataframe with at least the columns `date` and `yield`."""

    # --- Optional helpers -------------------------------------------------
    def _cache_path(self, name: str) -> Path:
        return CACHE_DIR / f"{name}.parquet"

    def cache_data(self, df: pd.DataFrame, name: str) -> None:
        df.to_parquet(self._cache_path(name), index=False)

    def try_cache(self, name: str) -> Optional[pd.DataFrame]:
        p = self._cache_path(name)
        if p.exists():
            return pd.read_parquet(p)
        return None

    def validate_data(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Fetched dataframe is empty")
        if df["yield"].isna().all():
            raise ValueError("Yield column all NaNs")
