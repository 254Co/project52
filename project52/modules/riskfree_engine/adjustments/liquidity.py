# -------------------------------------------------
# adjustments/liquidity.py  – spread‑based adjustment
# -------------------------------------------------

import pandas as pd


def apply_liquidity_adjustment(yields_df: pd.DataFrame, liquidity_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    If liquidity_df contains `date, spread` columns, subtract the *daily* bid-ask
    spread from the raw yield to compensate for illiquidity.  If not provided,
    apply a simple maturity-dependent haircut: 0 bp for ≤1y, 2 bp for 1-3y,
    5 bp for 3-10y, 7 bp for >10y.
    """
    adj = yields_df.copy()
    if liquidity_df is not None and not liquidity_df.empty:
        merged = adj.merge(liquidity_df, on="date", how="left")
        adj["yield"] = adj["yield"] - merged["spread"].fillna(0) / 10000
    else:
        cuts = (
            adj["maturity"]
                .apply(lambda m: 0 if m <= 1 else 0.0002 if m <= 3 else 0.0005 if m <= 10 else 0.0007)
        )
        adj["yield"] = adj["yield"] - cuts
    return adj

