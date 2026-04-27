import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.utils import resample

# ==========================
# LOAD DATASET
# ==========================
df = pd.read_csv("cvs/Naive-Bayes-Classification-Data.csv")

features = ["glucose", "bloodpressure"]
target = "diabetes"

# ==========================
# BALANCE DATASET
# ==========================
df_majority = df[df[target] == 0]
df_minority = df[df[target] == 1]

df_minority_up = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_up])

# ==========================
# SPLIT DATA
# ==========================
X = df_balanced[features]
y = df_balanced[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================
# TRAIN MODEL
# ==========================
model = GaussianNB()
model.fit(X_train, y_train)

# ==========================
# PREDICT
# ==========================
def predict(glucose, pressure):
    data = pd.DataFrame([[glucose, pressure]], columns=features)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    return prediction, probability

# ==========================
# METRICS
# ==========================
def generate_metrics():
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]
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