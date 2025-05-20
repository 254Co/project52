import pandas as pd


def calculate_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the return per month given a historical daily OHLC DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Historical daily OHLC data with either:
          - a DateTimeIndex, or
          - a 'Date' column convertible to datetime.
        Must include a 'Close' column.

    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame of monthly returns:
          - index: months 1–12
          - columns: years
          - values: (last_close_of_month / first_close_of_month) - 1
    """
    df = df.copy()

    # 1. Ensure a DatetimeIndex
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DateTimeIndex or a 'Date' column.")

    # 2. For each calendar month, get the first and last close
    monthly = df['close'].resample('ME').agg(first='first', last='last')
    monthly['return'] = monthly['last'] / monthly['first'] - 1

    # 3. Build year/month columns for pivot
    monthly = monthly.dropna(subset=['return'])
    monthly['year'] = monthly.index.year
    monthly['month'] = monthly.index.month

    # 4. Pivot so index=month (1–12), columns=year
    pivot = monthly.pivot(index='month', columns='year', values='return')

    # 5. Ensure all months 1–12 appear
    pivot = pivot.reindex(range(1, 13))

    # 6. Clean up axis names
    pivot.index.name = 'month'
    pivot.columns.name = 'year'

    return pivot
