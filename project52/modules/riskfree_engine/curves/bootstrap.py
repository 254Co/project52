# File: curves/bootstrap.py
import pandas as pd
import numpy as np


def bootstrap_zero_curve(yields_df: pd.DataFrame) -> pd.DataFrame:
    """Assume annual compounding.  Convert coupon yields to zero yields using
    a naive bootstrapping (non‑coupon bonds approximation)."""
    latest = yields_df.groupby("maturity").last().sort_index()
    zeros = []
    for m, y in zip(latest["maturity"], latest["yield"]):
        # Continuous compounding approximation: Z = exp(‑y * t)
        df = np.exp(-y * m)
        zero_rate = -np.log(df) / m
        zeros.append((m, zero_rate))
    return pd.DataFrame(zeros, columns=["maturity", "zero"])