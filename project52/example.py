from modules.Seasonality_Engine import calculate_monthly_returns
from connectors.polygonIO import fetch_ohlc_day
from modules.Volume_Features import volume_sma_ratios

df = fetch_ohlc_day("AAPL", "cqWpEROd6Kq0Q0zihGGYEosjAi4IPd_w")
df_returns = calculate_monthly_returns(df)
df_volume_sma = volume_sma_ratios(df)

print(df)
print(df_returns)
print(df_volume_sma)