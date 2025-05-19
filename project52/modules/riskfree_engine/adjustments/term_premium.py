# File: adjustments/term_premium.py
import pandas as pd


def apply_term_premium(yields_df: pd.DataFrame, short_maturity: float = 1) -> pd.DataFrame:
    """Subtract a proxy term premium = (yield – 1y yield) for maturities >1y.
    This mimics the core intuition of ACM without heavy statistics."""
    df = yields_df.copy()
    latest = df.groupby("maturity").last().reset_index()
    one_y = latest.loc[latest["maturity"].sub(short_maturity).abs().idxmin(), "yield"]
    prem = latest["yield"] - one_y
    prem[prem < 0] = 0  # Ignore negative premiums (rare)
    # Map back
    prem_map = dict(zip(latest["maturity"], prem))
    df["yield"] = df.apply(
        lambda r: r["yield"] - prem_map.get(r["maturity"], 0) if r["maturity"] > short_maturity else r["yield"],
        axis=1,
    )
    return df