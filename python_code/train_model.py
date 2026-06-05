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

print("LINEAR REGRESSION MODEL TRAINED SUCCESSFULLY")

# Visualization of coefficients

coefficients = model.coef_

feature_names = data.drop("price", axis=1).columns

plt.figure(figsize=(10,6))

plt.bar(
    feature_names,
    coefficients
)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Coefficient Value")

plt.xticks(rotation=45)

plt.show()