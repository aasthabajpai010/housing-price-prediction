# ==============================
# House Price Prediction using Linear Regression
# ==============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# ==============================
# Load Dataset
# ==============================

data = pd.read_csv("Housing.csv")

# Display first 5 rows
print(data.head())

# ==============================
# Handle Missing Values
# ==============================

data = data.fillna(data.mean(numeric_only=True))

# ==============================
# Separate Features and Target
# ==============================

# Target column
y = data["price"]

# Feature columns
X = data.drop("price", axis=1)

# ==============================
# Convert Categorical Data to Numeric
# ==============================

X = pd.get_dummies(X, drop_first=True)

# ==============================
# Feature Scaling
# ==============================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# ==============================
# Split Dataset
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# Train Linear Regression Model
# ==============================

model = LinearRegression()

model.fit(X_train, y_train)

# ==============================
# Prediction
# ==============================

y_pred = model.predict(X_test)

# ==============================
# Evaluation Metrics
# ==============================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print("\n===== Evaluation Metrics =====")

print("MAE  :", mae)

print("MSE  :", mse)

print("RMSE :", rmse)

# ==============================
# Graph 1 : Actual vs Predicted
# ==============================

plt.figure(figsize=(8,5))

plt.plot(y_test.values, label="Actual Prices", marker='o')

plt.plot(y_pred, label="Predicted Prices", marker='s')

plt.title("Actual vs Predicted House Prices")

plt.xlabel("Test Samples")

plt.ylabel("House Price")

plt.legend()

plt.grid(True)

plt.show()

# ==============================
# Graph 2 : Scatter Plot
# ==============================

plt.figure(figsize=(6,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Scatter Plot")

plt.grid(True)

plt.show()

# ==============================
# Graph 3 : Residual Error Plot
# ==============================

residuals = y_test - y_pred

plt.figure(figsize=(6,5))

plt.scatter(y_pred, residuals)

plt.axhline(y=0)

plt.xlabel("Predicted Prices")

plt.ylabel("Residual Errors")

plt.title("Residual Error Plot")

plt.grid(True)

plt.show()