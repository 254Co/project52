import TreasuryUS

#US Treasury Daily Yields
df = TreasuryUS.fetch_daily_par_yields()
print(df)