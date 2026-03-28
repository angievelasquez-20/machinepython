import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import os


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "cvs", "dataset_logisticmodel.csv")

df = pd.read_csv(file_path)

# Create target
df['Hired'] = df['salary'] > 78000

# Select features
X = df[['years_experience', 'skills_python', 'skills_sql', 'skills_ml']]
y = df['Hired']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Prediction
def predict_candidate(exp, python, sql, ml):
    data = [[exp, python, sql, ml]]
    prob = model.predict_proba(data)[0][1]
    pred = model.predict(data)[0]
    return prob, pred

# Metrics + plots
def generate_metrics():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("static/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1])
    plt.legend()
    plt.title("ROC Curve")

    plt.savefig("static/roc_curve.png")
    plt.close()

    return acc, report