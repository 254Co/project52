from zoneinfo import ZoneInfo
import exchange_calendars as xc
from datetime import datetime, date



def time_to_expiry(symbol: str, expiry: datetime, now=None,
                   day_count: str = "ACT/365") -> float:
    """
    Return time to expiry in years.
    symbol   – option root, used for special-case calendars
    expiry   – timezone-aware cutoff datetime
    now      – current UTC timestamp; default = datetime.utcnow()
    day_count– 'ACT/365' or 'ACT/252'
    """


    now = now or datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    if now >= expiry:
        return 1.0 / (365 * 24 * 60)  # clamp

    if day_count.upper() == "ACT/365":
        return (expiry - now).total_seconds() / (365 * 24 * 3600)

    if day_count.upper() == "ACT/252":
        cal = xc.get_calendar("XNYS")
        business_minutes = cal.minutes_in_range(now, expiry)
        return len(business_minutes) / (252 * 6.5 * 60)

    raise ValueError("Unsupported day_count")


def cutoff_utc(date_in: "date | str") -> datetime:
    """
    Return 21:00:00 UTC (16:00 ET) for the supplied calendar date.

    Parameters
    ----------
    date_in : datetime.date | str
        Either a date object or an ISO string 'YYYY-MM-DD'.

    Returns
    -------
    datetime
        Time-zone–aware UTC datetime for the close on that date.
    """
    # Normalise input
    if isinstance(date_in, str):
        date_obj = datetime.strptime(date_in, "%Y-%m-%d").date()
    elif isinstance(date_in, date):
        date_obj = date_in
    else:
        raise TypeError("date_in must be datetime.date or 'YYYY-MM-DD' string")

    # Build in ET first (handles DST), then convert to UTC
    cutoff_et = datetime(
        year=date_obj.year,
        month=date_obj.month,
        day=date_obj.day,
        hour=16, minute=0, second=0,
        tzinfo=ZoneInfo("America/New_York"),
    )
    return cutoff_et.astimezone(ZoneInfo("UTC"))


def expiry252time(ex_date):
    
    symbol = ""
    ex_datetime = cutoff_utc(ex_date)
    tv252 = time_to_expiry(symbol, ex_datetime, day_count = "ACT/252")
    
    return tv252

def expiry365time(ex_date):
    
    symbol = ""
    ex_datetime = cutoff_utc(ex_date)
    tv365 = time_to_expiry(symbol, ex_datetime, day_count = "ACT/365")
    
    return tv365


# Example
if __name__ == "__main__":    
    ex_date = "2025-12-31"
    x252 = expiry252time(ex_date)
    x365 = expiry365time(ex_date)
    yr = datetime.today().year

    print("")
    print(f"Results for {ex_date}:")
    print("")
    print(f"    Time remain for 252 day year: {x252}")
    print(f"    Time remain for 365 day year: {x365}")
    print("")
    print(f"Copyright {yr} | 254StudioZ LLC")
    print("")
    print("")
