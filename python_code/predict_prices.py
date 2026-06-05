import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

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

print("ACTUAL PRICE\t\tPREDICTED PRICE\n")

for i in range(10):

    print(y_test.iloc[i], "\t", predictions[i])

# Scatter plot

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Prices")

plt.show()