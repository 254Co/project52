# File: curves/interpolation.py
from scipy.interpolate import CubicSpline, interp1d
import pandas as pd

def interpolate_curve(curve_df: pd.DataFrame, method: str = "cubic"):
    x = curve_df["maturity"].values
    y = curve_df["zero"].values if "zero" in curve_df.columns else curve_df["yield"].values
    if method == "cubic" and len(x) >= 4:
        return CubicSpline(x, y, extrapolate=True)
    return interp1d(x, y, kind="linear", fill_value="extrapolate")