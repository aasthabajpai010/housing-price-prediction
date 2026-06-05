import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("Housing.csv")

# Handle missing values
data = data.fillna(method='ffill')

# Encoding
encoder = LabelEncoder()

categorical_columns = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
    'furnishingstatus'
]

for column in categorical_columns:

    data[column] = encoder.fit_transform(data[column])

# Features and target
X = data.drop("price", axis=1)

y = data["price"]

# Scaling
scaler = StandardScaler()

X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

print("MEAN ABSOLUTE ERROR")
print(mae)

print("\nMEAN SQUARED ERROR")
print(mse)

print("\nROOT MEAN SQUARED ERROR")
print(rmse)

# Visualization

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ================= GRAPH CODE HERE =================

import matplotlib.pyplot as plt

plt.plot(y_test.values, label="Actual Prices", marker='o')
plt.plot(y_pred, label="Predicted Prices", marker='s')

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Test Samples")
plt.ylabel("House Price")

plt.legend()
plt.grid(True)

plt.show()