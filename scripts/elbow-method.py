import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4' 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

csv_path = os.path.join(BASE_DIR, "data", "fuel_consumption.csv")

df = pd.read_csv(csv_path)

selected_features = [
    "Fuel Consumption City (L/100 km)", 
    "Fuel Consumption Comb (L/100 km)", 
    "Fuel Consumption Hwy (L/100 km)",
    "Engine Size(L)", 
    "Cylinders",
    "CO2 Emissions(g/km)"
]

X = df[selected_features].dropna()

df = pd.get_dummies(df, columns=["Fuel Type"], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()