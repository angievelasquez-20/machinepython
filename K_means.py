import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def applyClusteringKmeans(k):
    df = pd.read_csv('BankChurners.csv')

    df.columns = df.columns.str.strip()

    X = df[["Customer_Age", "Credit_Limit"]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    df["cluster"] = labels

    results = df.to_dict(orient="records")

    summaryClusters = df["cluster"].value_counts().to_dict()

    centers = scaler.inverse_transform(model.cluster_centers_).tolist()

    return {
        "results": results,
        "summaryClusters": summaryClusters,
        "centers": centers
    }

def generate_plot(results, centers):
    x = [row["Customer_Age"] for row in results]
    y = [row["Credit_Limit"] for row in results]
    clusters = [row["cluster"] for row in results]

    plt.figure(figsize=(8,6))

    plt.scatter(x, y, c=clusters, s=20, cmap='viridis')

    cx = [c[0] for c in centers]
    cy = [c[1] for c in centers]

    plt.scatter(cx, cy, color='black', marker='X', s=1000)

    plt.xlabel("Customer Age")
    plt.ylabel("Credit Limit")
    plt.title("K-Means Clustering - Bank Customers")

    plt.tight_layout()
    plt.savefig('static/kmeans_plot.png')
    plt.close()

