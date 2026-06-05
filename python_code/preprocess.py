import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# -----------------------------
# LOAD DATASET
# -----------------------------

data = pd.read_csv("Housing.csv")

print("FIRST 5 RECORDS")
print(data.head())

print("\nDATASET INFORMATION")
print(data.info())

# -----------------------------
# CHECK MISSING VALUES
# -----------------------------

print("\nMISSING VALUES")
print(data.isnull().sum())

# Visualization of missing values

missing_values = data.isnull().sum()

plt.figure(figsize=(10,5))

plt.bar(
    missing_values.index,
    missing_values.values
)

plt.title("Missing Values in Dataset")

plt.xlabel("Columns")

plt.ylabel("Missing Values")

plt.xticks(rotation=45)

plt.show()

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------

data = data.fillna(method='ffill')

print("\nMISSING VALUES AFTER HANDLING")
print(data.isnull().sum())

# -----------------------------
# ENCODE CATEGORICAL DATA
# -----------------------------

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

print("\nDATA AFTER ENCODING")
print(data.head())

# -----------------------------
# FEATURES AND TARGET
# -----------------------------

X = data.drop("price", axis=1)

y = data["price"]

# -----------------------------
# FEATURE SCALING
# -----------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nSCALED FEATURES")
print(X_scaled[:5])

# -----------------------------
# VISUALIZATION 1
# HISTOGRAMS
# -----------------------------

data.hist(figsize=(12,10))

plt.suptitle("Feature Distribution")

plt.show()

# -----------------------------
# VISUALIZATION 2
# CORRELATION HEATMAP
# -----------------------------

correlation = data.corr()

plt.figure(figsize=(12,8))

plt.imshow(
    correlation,
    cmap='coolwarm'
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")

plt.show()

print("\nPREPROCESSING COMPLETED SUCCESSFULLY")