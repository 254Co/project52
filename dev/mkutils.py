import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta, time
from zoneinfo import ZoneInfo
import exchange_calendars as xc
from functools import lru_cache
import requests

# -----------------------------------------------------------------------------
# Calendar utilities
# -----------------------------------------------------------------------------

@lru_cache(maxsize=10)
def get_calendar(symbol: str = "XNYS") -> xc.ExchangeCalendar:
    """
    Fetch and cache an exchange calendar (default: NYSE).
    """
    return xc.get_calendar(symbol)


def next_trading_minute(dt: datetime, cal: xc.ExchangeCalendar) -> datetime:
    """
    Snap any timestamp to the next open trading minute given an exchange calendar.
    """
    # search forward up to 1 trading day
    idx = cal.minutes_in_range(dt, dt + timedelta(days=7))
    for ts in idx:
        if ts > dt:
            return ts
    raise ValueError("No trading minute found within the next 7 days")


def prev_trading_minute(dt: datetime, cal: xc.ExchangeCalendar) -> datetime:
    """
    Snap any timestamp to the previous open trading minute given an exchange calendar.
    """
    idx = cal.minutes_in_range(dt - timedelta(days=7), dt)
    for ts in reversed(idx):
        if ts < dt:
            return ts
    raise ValueError("No trading minute found within the previous 7 days")


def business_days_between(start: datetime, end: datetime, cal: xc.ExchangeCalendar, inclusive: bool = False) -> int:
    """
    Fast integer count of trading days between two datetimes.
    """
    sessions = cal.sessions_in_range(start.date(), end.date())
    count = len(sessions)
    if inclusive and cal.is_session(end.date()):
        count += 1
    return count


def session_bounds(s: date, cal: xc.ExchangeCalendar) -> tuple[datetime, datetime]:
    """
    Returns (open_dt, close_dt) aware datetimes for a given session date.
    """
    open_dt = cal.session_open(s)
    close_dt = cal.session_close(s)
    return open_dt, close_dt


def is_half_day(s: date, cal: xc.ExchangeCalendar) -> tuple[bool, float]:
    """
    Detect if a session is a half-day, returning (True, fraction) if so.
    """
    open_dt, close_dt = session_bounds(s, cal)
    full_minutes = 6.5 * 60
    actual_minutes = (close_dt - open_dt).total_seconds() / 60
    if actual_minutes < full_minutes:
        return True, actual_minutes / full_minutes
    return False, 1.0


def trading_minutes_between(start: datetime, end: datetime, cal: xc.ExchangeCalendar) -> int:
    """
    Mathematical count of trading minutes between two datetimes (handles partial days).
    """
    total = 0
    sessions = cal.sessions_in_range(start.date(), end.date())
    for s in sessions:
        open_dt, close_dt = session_bounds(s, cal)
        s0 = max(start, open_dt)
        e0 = min(end, close_dt)
        if s0 < e0:
            total += int((e0 - s0).total_seconds() // 60)
    return total

# -----------------------------------------------------------------------------
# Date generation / mapping utilities
# -----------------------------------------------------------------------------

def spot_to_settle_lag(contract_symbol: str) -> int:
    """
    Map contract to its standard spot-to-settle lag in days (default T+2 equities).
    """
    # naive mapping by asset class prefix
    if contract_symbol.upper().startswith("BTC") or contract_symbol.upper().startswith("ETH"):
        return 0  # crypto
    return 2  # default T+2


def days_in_contract_cycle(root: str, cycle: str = "monthly") -> list[str]:
    """
    Returns list of future contract codes in the current cycle.
    cycle = "monthly" or "quarterly".
    """
    from dateutil.relativedelta import relativedelta

    now = datetime.now()
    codes = []
    if cycle == "monthly":
        for i in range(1, 13):
            dt = now + relativedelta(months=i)
            codes.append(f"{root}{dt.strftime('%y%m')}")
    elif cycle == "quarterly":
        # next four quarter-ends Mar/Jun/Sep/Dec
        this_q = (now.month - 1) // 3
        for i in range(1, 5):
            q = (this_q + i) % 4 + 1
            year = now.year + ((this_q + i) // 4)
            dt = datetime(year, q * 3, 1)
            codes.append(f"{root}{dt.strftime('%y%m')}")
    else:
        raise ValueError("cycle must be 'monthly' or 'quarterly'")
    return codes

# -----------------------------------------------------------------------------
# Model utilities
# -----------------------------------------------------------------------------

def implied_dividend_yield(under_price: float,
                            call_price: float,
                            put_price: float,
                            strike: float,
                            tau: float) -> float:
    """
    Quick dividend yield snapshot via put-call parity, assuming r=0.
    q ≈ (C - P) / (S * τ)
    """
    return (call_price - put_price) / (under_price * tau)


def black_scholes_greeks_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised Δ, Γ, Θ, ν, ρ for a DataFrame with columns:
    ['under_price','strike','tau','vol','r']
    Returns df[['delta','gamma','theta','vega','rho']].
    """
    from scipy.stats import norm

    S = df['under_price'].to_numpy()
    K = df['strike'].to_numpy()
    T = df['tau'].to_numpy()
    sigma = df['vol'].to_numpy()
    r = df['r'].to_numpy()

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2)

    return pd.DataFrame({'delta': delta,
                         'gamma': gamma,
                         'theta': theta,
                         'vega': vega,
                         'rho': rho}, index=df.index)

# -----------------------------------------------------------------------------
# Array / DataFrame utilities
# -----------------------------------------------------------------------------

def hloc_to_log_returns(arr: np.ndarray, clip_nan: bool = True) -> np.ndarray:
    """
    Converts close prices (4th column) in HLOC array to log returns.
    """
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("Input must be 2D array with at least 4 columns")
    prices = arr[:, 3]
    logs = np.log(prices)
    ret = np.diff(logs, prepend=np.nan)
    if clip_nan:
        ret = np.nan_to_num(ret)
    return ret


def smart_resample(df: pd.DataFrame, rule: str, how: str = "vwap") -> pd.DataFrame:
    """
    Resample tick data to given rule using VWAP, last, mean, or custom agg.
    Assumes df has ['price','volume'] columns.
    """
    if how == "vwap":
        def vwap(x):
            p = x['price'] * x['volume']
            return p.sum() / x['volume'].sum()
        return df.resample(rule).apply(vwap)
    elif how in ("last", "mean", "sum"):  # fallback
        return getattr(df.resample(rule), how)()
    else:
        return df.resample(rule).apply(how)


def ohlc_rolling_windows(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling OHLC over price array using a sliding window.
    Returns array shape (n-window+1, 4).
    """
    n = len(prices)
    if window > n:
        return np.empty((0, 4))
    out = np.empty((n - window + 1, 4))
    for i in range(n - window + 1):
        w = prices[i:i+window]
        out[i, 0] = w[0]
        out[i, 1] = w.max()
        out[i, 2] = w.min()
        out[i, 3] = w[-1]
    return out

# -----------------------------------------------------------------------------
# External data fetching / scheduling
# -----------------------------------------------------------------------------

def peek_open_interest(fut_symbol: str, expiry: date) -> int:
    """
    Pull CME/ICE open interest via their JSON/FTP feeds.
    Caches per-day.
    """
    # This is an example stub; implement actual API calls per exchange.
    raise NotImplementedError("Open interest fetch not implemented for %s" % fut_symbol)


def settle_clock(timezone: str = "America/New_York") -> dict[str, datetime]:
    """
    Returns settlement cut-off times for major asset classes in given tz.
    """
    tz = ZoneInfo(timezone)
    # defaults in ET
    mapping = {
        'equity_option': time(16, 0),
        'futures':      time(16, 15),
        'crypto':       time(0, 0),
    }
    return {k: datetime.combine(date.today(), v, tzinfo=tz) for k, v in mapping.items()}


def generate_expiry_grid(root: str, years_ahead: int = 5) -> list[datetime]:
    """
    Returns all listed equity option expiries (weeklys + monthlys) as UTC datetimes.
    """
    end = datetime.utcnow().year + years_ahead
    rng = pd.date_range(start=datetime.utcnow(), end=f"{end}-12-31", freq='W-FRI')
    return [dt.tz_localize('America/New_York').astimezone(ZoneInfo('UTC')) for dt in rng]


def roll_yield_curve(curve_df: pd.DataFrame, now: datetime = None) -> pd.Series:
    """
    Adjusts futures/forward curve P&L for the daily roll to next front month.
    Assumes curve_df indexed by expiry date, values are price.
    Returns series of roll P&L.
    """
    df = curve_df.sort_index()
    front = df.shift(-1) - df
    return front


# Example
if __name__ == "__main__": 
    xc = get_calendar()
    dt = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    ntm = next_trading_minute(dt, xc)
    print (ntm)
    ptm = prev_trading_minute(dt, xc)
    print (ptm)
    dt_end = dt + timedelta(days = 75)
    bdb = business_days_between(start=dt, end=dt_end, cal=xc)
    print(bdb)
    
    