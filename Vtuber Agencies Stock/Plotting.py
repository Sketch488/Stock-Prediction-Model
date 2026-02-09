import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

filepath = "StockInfo.xlsx"
column_names = ["ANYCOLOR Inc", "COVER Corperation"]
combined = pd.read_excel(filepath)

# Create subplots: 1 row, 2 columns
fig, axes = plt.subplots(1, len(column_names), figsize=(15, 5))

for i, col in enumerate(column_names):
    ax = axes[i]
    
    # Prepare data
    stock_prices = combined[["Date", col]].copy()
    stock_prices['Date'] = pd.to_datetime(stock_prices['Date'])
    stock_prices.set_index('Date', inplace=True)
    stock_prices['Days'] = (stock_prices.index - stock_prices.index[0]).days
    X = stock_prices[['Days']]
    y = stock_prices[col]
    
    # Train linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next month
    future_days = np.array([stock_prices['Days'].max() + 30]).reshape(-1, 1)
    prediction = model.predict(future_days)
    
    future_date = stock_prices.index[-1] + pd.DateOffset(months=1)
    
    # Plot actual prices
    ax.plot(stock_prices.index, y, label="Actual Price")
    
    # Plot prediction
    ax.scatter(future_date, prediction, color='red', label="Predicted Price")
    
    ax.set_title(f"{col} Stock Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

plt.tight_layout()
plt.show()
