# File: analytics/metrics.py
import pandas as pd


def compute_spot_rates(spline, maturities: List[float]) -> pd.Series:
    return pd.Series(index=maturities, data=[float(spline(t)) for t in maturities])


def compute_forward_rates(spline, t1: float, t2: float) -> float:
    z1, z2 = float(spline(t1)), float(spline(t2))
    return (z2 * t2 - z1 * t1) / (t2 - t1)