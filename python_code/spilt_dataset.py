import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("Housing.csv")

# Handle missing values
data = data.fillna(method='ffill')

# Encoding categorical columns
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

# Feature scaling
scaler = StandardScaler()

X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("TRAINING DATA SHAPE")
print(X_train.shape)

print("\nTESTING DATA SHAPE")
print(X_test.shape)

# Visualization

sizes = [len(X_train), len(X_test)]

labels = ["Training Data", "Testing Data"]

plt.figure(figsize=(6,6))

plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%'
)

plt.title("Training and Testing Data Split")

plt.show()