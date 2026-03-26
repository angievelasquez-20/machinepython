import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [10, 3],
    [20, 5],
    [30, 7],
    [40, 8],
    [50, 9]
])

y = np.array([50, 120, 200, 280, 350])

model = LinearRegression()
model.fit(X, y)

def calculateCalories(duration, intensity):
    data = np.array([[duration, intensity]])
    return model.predict(data)[0]