from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def getDataset():
    return[{"name": "Ana", "age": 22, "income": 1200, "bills": 300},
        {"name": "Luis", "age": 25, "income": 1500, "bills": 350},
        {"name": "Carlos", "age": 23, "income": 1300, "bills": 280},
        {"name": "Marta", "age": 45, "income": 4000, "bills": 1200},
        {"name": "Sofía", "age": 50, "income": 4200, "bills": 1400},
        {"name": "Jorge", "age": 47, "income": 3900, "bills": 1100},
        {"name": "Elena", "age": 31, "income": 2500, "bills": 700},
        {"name": "Pedro", "age": 33, "income": 2700, "bills": 750},
        {"name": "Laura", "age": 29, "income": 2400, "bills": 680},
        {"name": "Andrés", "age": 52, "income": 5000, "bills": 1600},
        {"name": "Camila", "age": 21, "income": 1100, "bills": 250},
        {"name": "Diego", "age": 38, "income": 3200, "bills": 900}]

def applyClusteringKmeans():
    data=getDataset()

    x=[[person["age"],person["income"],person["bills"]] for person in data]
    scaler=StandardScaler()
    xScaled=scaler.fit_transform(x)
    model=KMeans(n_clusters=3,random_state=42, n_init=10)
    labels=model.fit_predict(xScaled)

    result=[]
    
    for i, person in enumerate(data):
        row=person.copy()
        row["cluster"]=int(labels[i])
        result.append(row)

    summaryCluster={}

    for label in labels:
        label=int(label)
        summaryCluster[label]=summaryCluster.get(label,0)+1

    centroids=model.cluster_centers_.tolist()

    return{
        "results": result,
        "summaryClusters": summaryCluster,
        "centers": centroids
    }
