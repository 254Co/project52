import TreasuryUS
import EdgarSEC
import BankJapan
import polygonIO
import InternationalMonetaryFund

#US Treasury Daily Yields
df = TreasuryUS.fetch_daily_par_yields()
print(df)

df = EdgarSEC.fetch_ticker_cik_map()
print(df)

df = BankJapan.fetch_overnight_call_rate_daily()
print(df)

df = BankJapan.fetch_overnight_call_rate_monthly()
print(df)

df = BankJapan.fetch_tokyo_market_interbank_rates_daily()
print(df)

df = InternationalMonetaryFund.fetch_imf_indicators_map()
print(df)

df = polygonIO.fetch_ohlc_day("AAPL", "cqWpEROd6Kq0Q0zihGGYEosjAi4IPd_w")
print(df.head(), "\n")
print(f"{len(df):,} day bars fetched "
        f"({df.index[0].date()} → {df.index[-1].date()}) for AAPL")

df = polygonIO.fetch_ohlc_min("AAPL", "cqWpEROd6Kq0Q0zihGGYEosjAi4IPd_w")
print(df.head(), "\n")
print(f"{len(df):,} minute bars fetched "
        f"({df.index[0].date()} → {df.index[-1].date()}) for AAPL")