import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Dataset (House Prices)
# -----------------------------
# [Size (m²), Rooms]
X = np.array([
    [50, 2],
    [70, 3],
    [90, 3],
    [120, 4],
    [150, 5]
])

# Prices
y = np.array([100000, 150000, 180000, 250000, 320000])

# -----------------------------
# Model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Prediction
# -----------------------------
def predict_price(size, rooms):
    data = np.array([[size, rooms]])
    return model.predict(data)[0]

# -----------------------------
# Plot
# -----------------------------
def generate_plot(size, rooms, prediction):
    plt.figure()

    # Scatter real data (Size vs Price)
    plt.scatter(X[:, 0], y)

    # Regression line (using size for visualization)
    plt.plot(X[:, 0], model.predict(X))

    # Predicted point
    plt.scatter([size], [prediction], marker='x', s=100)

    plt.xlabel("House Size (m²)")
    plt.ylabel("Price")
    plt.title("House Price Prediction (Linear Regression)")

    plt.savefig("static/plot.png")
    plt.close()