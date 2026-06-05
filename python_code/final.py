import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------

data = pd.read_csv("Housing.csv")

print("FIRST 5 RECORDS\n")

print(data.head())

# -------------------------------------------------
# CHECK MISSING VALUES
# -------------------------------------------------

print("\nMISSING VALUES\n")

print(data.isnull().sum())

# Handle missing values

data = data.fillna(method='ffill')

# -------------------------------------------------
# ENCODE CATEGORICAL DATA
# -------------------------------------------------

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

# -------------------------------------------------
# FEATURES AND TARGET
# -------------------------------------------------

X = data.drop("price", axis=1)

y = data["price"]

# -------------------------------------------------
# FEATURE SCALING
# -------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# SPLIT DATASET
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAINING DATA SIZE:", X_train.shape)

print("TESTING DATA SIZE:", X_test.shape)

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nLINEAR REGRESSION MODEL TRAINED SUCCESSFULLY")

# -------------------------------------------------
# PREDICTIONS
# -------------------------------------------------

predictions = model.predict(X_test)

# -------------------------------------------------
# RESULT ANALYSIS
# -------------------------------------------------

results = pd.DataFrame()

results["Actual Price"] = y_test.values

results["Predicted Price"] = predictions

results["Difference"] = (
    results["Actual Price"] -
    results["Predicted Price"]
)

print("\nRESULT ANALYSIS\n")

print(results.head(15))

# -------------------------------------------------
# MODEL EVALUATION
# -------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")

print("\nMAE =", mae)

print("\nMSE =", mse)

print("\nRMSE =", rmse)

print("\nR2 SCORE =", r2)

# -------------------------------------------------
# CHECK CUSTOM TEST DATA
# -------------------------------------------------

print("\nCUSTOM HOUSE PRICE PREDICTION")

# Example custom house data

custom_data = [[
    7500,   # area
    4,      # bedrooms
    3,      # bathrooms
    2,      # stories
    1,      # mainroad
    1,      # guestroom
    0,      # basement
    0,      # hotwaterheating
    1,      # airconditioning
    2,      # parking
    1,      # prefarea
    1       # furnishingstatus
]]

# Scale custom data

custom_scaled = scaler.transform(custom_data)

# Predict price

predicted_price = model.predict(custom_scaled)

print("\nPREDICTED HOUSE PRICE =")

print(predicted_price[0])

# -------------------------------------------------
# VISUALIZATION 1
# ACTUAL VS PREDICTED
# -------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    y_test.values,
    label="Actual Prices"
)

plt.plot(
    predictions,
    label="Predicted Prices"
)

plt.title("Actual Prices vs Predicted Prices")

plt.xlabel("Test Houses")

plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.show()

# -------------------------------------------------
# VISUALIZATION 2
# SCATTER PLOT
# -------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Scatter Plot: Actual vs Predicted")

plt.show()

# -------------------------------------------------
# VISUALIZATION 3
# ERROR DISTRIBUTION
# -------------------------------------------------

errors = y_test.values - predictions

plt.figure(figsize=(8,6))

plt.hist(
    errors,
    bins=20
)

plt.title("Error Distribution")

plt.xlabel("Prediction Error")

plt.ylabel("Frequency")

plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY")