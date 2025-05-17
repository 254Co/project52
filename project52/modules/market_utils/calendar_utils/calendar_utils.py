import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta, time
from zoneinfo import ZoneInfo
import exchange_calendars as xc
from functools import lru_cache


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




# Example
if __name__ == "__main__": 
    xc = get_calendar()
    dt = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    ntm = next_trading_minute(dt, xc)
    print (f"Next trading minute is {ntm}.")
    
    ptm = prev_trading_minute(dt, xc)
    print (f"Last trading minute was {ptm}.")
    
    dys = 75
    dt_end = dt + timedelta(days = dys)
    bdb = business_days_between(start=dt, end=dt_end, cal=xc)
    print(f"Number of business days between the next {dys} days is {bdb}.")
    
    tmb = trading_minutes_between(start=dt, end=dt_end, cal=xc)
    print(f"Number of trading minutes over the next {dys} days is {tmb}.")
    
    sb = session_bounds("2025-05-16", xc)
    print(sb)
    
    ihd = is_half_day("2025-05-17", xc)
    print(ihd)
    
