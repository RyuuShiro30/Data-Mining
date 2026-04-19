import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Dataset/new-york_9-24-2016_9-30-2017.csv')

df = df[['Date', 'Low Price']]

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(pd.Timestamp.toordinal)

df = df.dropna()

X = df[['Date']]
y = df['Low Price']

df = df.sort_values('Date')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

X_range = np.linspace(X['Date'].min(), X['Date'].max(), 100).reshape(-1,1)

X_range_poly = poly.transform(X_range)
y_range_pred = model.predict(X_range_poly)

y_pred = model.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

data_plot = pd.DataFrame({
    'Date': X_test.values.flatten(),
    'Actual': y_test.values,
    'Predicted': y_pred
}).sort_values('Date')

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_range, y_range_pred, color='red', label='Predicted Curve')

plt.title("Polynomial Regression - Pumpkin Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()