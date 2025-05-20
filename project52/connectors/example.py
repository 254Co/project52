import TreasuryUS
import EdgarSEC
import polygonIO

#US Treasury Daily Yields
df = TreasuryUS.fetch_daily_par_yields()
print(df)

df = EdgarSEC.fetch_ticker_cik_map()
print(df)

df = polygonIO.fetch_ohlc_day("AAPL", "cqWpEROd6Kq0Q0zihGGYEosjAi4IPd_w")
print(df.head(), "\n")
print(f"{len(df):,} day bars fetched "
        f"({df.index[0].date()} → {df.index[-1].date()}) for AAPL")

df = polygonIO.fetch_ohlc_min("AAPL", "cqWpEROd6Kq0Q0zihGGYEosjAi4IPd_w")
print(df.head(), "\n")
print(f"{len(df):,} minute bars fetched "
        f"({df.index[0].date()} → {df.index[-1].date()}) for AAPL")