import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

try:
    data = pd.read_csv("data.csv")
except:
    print("Error: data.csv file not found!")
    exit()

if 'Date' not in data.columns or 'Sales' not in data.columns:
    print("Error: Dataset must contain 'Date' and 'Sales' columns")
    exit()

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])
data = data.sort_values('Date')

data['Days'] = (data['Date'] - data['Date'].min()).dt.days

X = data[['Days']]
y = data['Sales']

model = LinearRegression()
model.fit(X, y)

last_day = data['Days'].max()
future_days = np.array([last_day + i*30 for i in range(1, 7)]).reshape(-1, 1)

future_sales = model.predict(future_days)

print("\nFuture Sales Predictions:\n")
for i, sale in enumerate(future_sales):
    print(f"Month {i+1}: {round(sale, 2)}")

plt.figure(figsize=(10,5))
plt.scatter(X, y, label='Actual Sales')
plt.plot(X, model.predict(X), label='Regression Line')
plt.scatter(future_days, future_sales, label='Predicted Sales')
plt.xlabel("Days")
plt.ylabel("Sales")
plt.title("Sales Forecasting using Machine Learning")
plt.legend()
plt.show()
