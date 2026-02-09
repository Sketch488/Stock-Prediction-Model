import yfinance as yf
import pandas as pd
import os

tickers = ["5253.T", "5032.T"]
column_names = ["ANYCOLOR Inc", "COVER Corperation"]

filepath = "StockInfo.xlsx"

data = yf.download(tickers, period= "6mo", interval="1d")
price = data['Close'].reset_index()
price.columns = ["Date"] + column_names

if os.path.exists(filepath):
    existing = pd.read_excel(filepath)
    combined = pd.merge(existing, price, on='Date', how='outer')
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined = combined.sort_values("Date").reset_index(drop=True)
else:
    combined = price

combined.to_excel(filepath, index=False)

print(combined)