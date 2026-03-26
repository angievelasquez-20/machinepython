import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([
    [10, 3],
    [20, 5],
    [30, 7],
    [40, 8],
    [50, 9]
])

y = np.array([50, 120, 200, 280, 350])

# Model
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Prediction
# -----------------------------
def calculateCalories(duration, intensity):
    data = np.array([[duration, intensity]])
    return model.predict(data)[0]

# -----------------------------
# Plot
# -----------------------------
def generate_plot(duration, intensity, prediction):
    plt.figure()

    # Scatter real data
    plt.scatter(X[:, 0], y)

    # Regression line (using duration only for visualization)
    plt.plot(X[:, 0], model.predict(X))

    # Predicted point
    plt.scatter([duration], [prediction], marker='x', s=100)

    plt.xlabel("Duration (minutes)")
    plt.ylabel("Calories Burned")
    plt.title("Calories Prediction (Linear Regression)")

    plt.savefig("static/plot.png")
    plt.close()