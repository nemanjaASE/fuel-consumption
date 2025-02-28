import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

csv_path = os.path.join(BASE_DIR, "data", "fuel_consumption.csv")
models_dir = os.path.join(os.path.dirname(__file__), "../models")
model_path = os.path.join(models_dir, "kmeans_model.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

joblib.dump(kmeans, model_path)
print("✅ Model saved as 'kmeans_model.pkl'")
joblib.dump(scaler, scaler_path)
print("✅ Scaler saved as 'scaler.pkl'")
