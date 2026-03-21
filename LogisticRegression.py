import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("cvs/dataset_regresion_logistica.csv")


X = df[[
    "edad",
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