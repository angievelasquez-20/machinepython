import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# upload data
data = pd.read_csv('AI_Job_Market_Trends_2026.csv', sep=';')

df = pd.DataFrame(data)

# Crete variable objetiv (target)
df["target"] = df["hiring_urgency"].apply(lambda x: 1 if x == "High" else 0)

# delete unnecessary columns
df = df.drop(columns=["job_id", "hiring_urgency"])

# Convert categorical variables to dummy variables
df = pd.get_dummies(df)

# separete features and target
X = df.drop(columns=["target"])
y = df["target"]

# Model training
model = LogisticRegression(max_iter=2000)
model.fit(X, y)


# Prediction function
def predict_hiring(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)

    # asegurate that input_df has the same columns as X, filling missing columns with 0
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prob = model.predict_proba(input_df)[0][1]
    return prob


# Gráfic of probability vs age
def generate_confusion_matrix():
   
    y_pred = model.predict(X)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.savefig('static/logistic2_plot.png') 
    plt.close()

def generate_roc_curve():
    y_prob = model.predict_proba(X)[:,1]

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], linestyle='--')  # diagonal

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig('static/roc_curve.png')
    plt.close()