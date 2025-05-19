# File: curves/builder.py
import pandas as pd
from datetime import datetime, timedelta
from ..data_sources.treasury import TreasuryDataSource
from ..data_sources.sofr import SOFRDataSource
from ..data_sources.sonia import SONIADataSource
from ..data_sources.estr import ESTRDataSource
from ..data_sources.repo import RepoDataSource
from ..caching.cache import CacheManager
from ..adjustments.liquidity import apply_liquidity_adjustment
from ..adjustments.term_premium import apply_term_premium
from .interpolation import interpolate_curve

class RiskFreeCurveBuilder:
    def __init__(self, config: Dict):
        self.config = config
        self.sources = [
            TreasuryDataSource(),
            SOFRDataSource(),
            SONIADataSource(),
            ESTRDataSource(),
            RepoDataSource(),
        ]

    def _merge_sources(self, *dataframes: pd.DataFrame) -> pd.DataFrame:
        # Align on same valuation date (most recent common date)
        common_date = min(df["date"].max() for df in dataframes)
        frames = [df[df["date"] == common_date] for df in dataframes]
        return pd.concat(frames, ignore_index=True)

    def _fetch_liquidity_metrics(self, valuation_date: str) -> Optional[pd.DataFrame]:
        # No free bid‑ask dataset; return empty → fallback haircut kicks in
        return None

    def build_curve(self, valuation_date: str | date):
        val_str = (pd.to_datetime(valuation_date).date()).isoformat()
        # Restrict fetch window to the last 90 days for speed
        start = (pd.to_datetime(val_str) - timedelta(days=90)).date().isoformat()
        dfs = [s.fetch_data(start, val_str) for s in self.sources]
        combined = self._merge_sources(*dfs)
        combined = apply_liquidity_adjustment(combined, self._fetch_liquidity_metrics(val_str))
        combined = apply_term_premium(combined)
        zero_df = bootstrap_zero_curve(combined)
        spline = interpolate_curve(zero_df, method=self.config.get("interpolation", {}).get("method", "cubic"))
        return {
            "valuation_DATE": val_str,
            "zero_df": zero_df,
            "spline": spline,
        }
