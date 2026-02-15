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

    combined = (
        existing.set_index("Date")
        .combine_first(price.set_index("Date"))
        .reset_index()
        )
    
    combined = (
        combined.drop_duplicates(subset="Date", keep="first")
        .dropna(subset=["Date"] + column_names)
        .reset_index(drop=True)
    )

    price_cols = [c for c in combined.columns if c != "Date"]
    combined[price_cols] = combined[price_cols].round(0).astype("Int64")

else:
    combined = price

combined["Date"] = pd.to_datetime(combined["Date"])
combined = combined.sort_values(by="Date", ascending=False).reset_index(drop=True)
combined["Date"] = combined["Date"].dt.strftime("%Y-%m-%d")

combined.to_excel(filepath, index=False)

print(combined)