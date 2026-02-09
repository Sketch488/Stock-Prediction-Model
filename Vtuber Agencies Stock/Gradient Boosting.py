import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt

filepath = "StockInfo.xlsx"
column_names = ["ANYCOLOR Inc", "COVER Corperation"]
combined = pd.read_excel(filepath)

fig, axes = plt.subplots(1, len(column_names), figsize=(15,5))
    
# --- Predictive model ---
for i, col in enumerate(column_names):
    stock_prices = combined[["Date", col]].copy()
    stock_prices['Date'] = pd.to_datetime(stock_prices['Date'])
    stock_prices.set_index('Date', inplace=True)
    
    # Convert dates to numeric
    stock_prices['Days'] = (stock_prices.index - stock_prices.index[0]).days
    X = stock_prices[['Days']]
    y = stock_prices[col]
    
    # Train model
    model = GradientBoostingRegressor (
        n_estimators=500,
        learning_rate= 0.03,
        max_depth=5,
        random_state=42
        )
    model.fit(X, y)
    
    # Predict next month
    future_day = stock_prices['Days'].max() + 30
    future_days = np.array([[future_day]])

    predictions = model.predict(future_days)

    ax = axes[i]
    
    ax.plot(stock_prices.index, y, label="Actual Price")
    
    future_date = stock_prices.index.max() + pd.DateOffset(days=30)
    
    ax.scatter(future_date, predictions, color="Red", label="Prediction")
    
    ax.set_title(col)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    
    print(f"Predicted next month price for {col}: {predictions[0]:.2f}")

plt.tight_layout()
plt.show()