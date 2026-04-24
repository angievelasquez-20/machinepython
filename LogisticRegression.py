import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression

# Sample data for demonstration
data = {
    "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 10,  # repeat to have more data
    "ingreso_mensual": [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000] * 10,
    "visitas_web_mes": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] * 10,
    "tiempo_sitio_min": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 10,
    "compras_previas": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10,
    "descuento_usado": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45] * 10,
    "target": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1] * 10  # sample targets
}

df = pd.DataFrame(data)

X = df[[
    "age",
    "ingreso_mensual",
    "visitas_web_mes",
    "tiempo_sitio_min",
    "compras_previas",
    "descuento_usado"
]]

y = df["target"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)


def predict_purchase(age, income, visits, time, purchases, discount):
    prob = model.predict_proba([[age, income, visits, time, purchases, discount]])[0][1]
    return prob

def generate_plot(age, income, visits, time, purchases, discount):
    ages = list(range(18, 81))  # assume age range 18-80
    probs = []
    for a in ages:
        prob = predict_purchase(a, income, visits, time, purchases, discount)
        probs.append(prob)
    plt.figure()
    plt.plot(ages, probs)
    plt.scatter([age], [predict_purchase(age, income, visits, time, purchases, discount)], color='red', marker='x', s=100)
    plt.xlabel('Age')
    plt.ylabel('Probability of Purchase')
    plt.title('Logistic Regression: Probability vs Age')
    plt.savefig('static/logistic_plot.png')
    plt.close()

