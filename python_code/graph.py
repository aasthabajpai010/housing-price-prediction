import matplotlib.pyplot as plt

# Example values (replace with your actual y_test and y_pred)
y_test = [50, 60, 70, 80, 90, 100]
y_pred = [48, 63, 68, 82, 88, 97]

plt.plot(y_test, label="Actual Prices", marker='o')
plt.plot(y_pred, label="Predicted Prices", marker='s')

plt.title("Actual vs Predicted House Prices using Linear Regression")
plt.xlabel("Test Samples")
plt.ylabel("House Price")
plt.legend()
plt.grid(True)

plt.show()