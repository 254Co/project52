import pandas as pd

def volume_sma_ratios(
    df: pd.DataFrame,
    windows: list[int] = [5, 20, 50, 200],
    volume_col: str = "volume"
) -> pd.DataFrame:
    """
    Compute Volume_t / SMA(window) of Volume for multiple lookback windows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a volume series.
    windows : list[int], default [5, 20, 50, 200]
        Lookback windows for the simple moving averages.
    volume_col : str, default "Volume"
        Name of the column in `df` to use.

    Returns
    -------
    pd.DataFrame
        Indexed like `df`, with one column per window:
        "{volume_col}_over_{window}SMA", each containing
        df[volume_col] / SMA(window) of df[volume_col].
        Rows with insufficient history (first window−1 rows) will be NaN.
    """
    if volume_col not in df.columns:
        raise KeyError(f"Column {volume_col!r} not found in DataFrame.")

    # Ensure chronological order
    df = df.sort_index()

    vol = df[volume_col]
    result = pd.DataFrame(index=df.index)

    for w in windows:
        sma = vol.rolling(window=w, min_periods=w).mean()
        col_name = f"{volume_col}_over_{w}SMA"
        result[col_name] = vol / sma

    result = result.sort_index(ascending=False)
    return result
