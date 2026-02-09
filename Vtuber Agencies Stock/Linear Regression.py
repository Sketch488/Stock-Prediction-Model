import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

filepath = "StockInfo.xlsx"
column_names = ["ANYCOLOR Inc", "COVER Corperation"]
combined = pd.read_excel(filepath)

# --- Predictive model ---
for col in column_names:
    stock_prices = combined[["Date", col]].copy()
    stock_prices['Date'] = pd.to_datetime(stock_prices['Date'])
    stock_prices.set_index('Date', inplace=True)
    
    # Convert dates to numeric
    stock_prices['Days'] = (stock_prices.index - stock_prices.index[0]).days
    X = stock_prices[['Days']]
    y = stock_prices[col]
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next month
    future_days = np.array([stock_prices['Days'].max() + 1]).reshape(-1, 1)
    prediction = model.predict(future_days)
    
    print(f"Predicted next month price for {col}: {prediction[0]:.2f}")

    # 2/7/2026 Anycolor 4245 -> Tomorrow 4208.35 -> Next Month 3757.27
    # 2/7/2026 Cover Corp 1545 -> Tomorrow 1567.06 -> Next Month  1332.78
