import joblib
import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

csv_path = os.path.join(BASE_DIR, "data", "fuel_consumption.csv")

def load_models():
    with open("./models/co2_emission_model.pkl", "rb") as f:
        model = joblib.load(f)

    with open("./models/feature_columns.pkl", "rb") as f:
        features = joblib.load(f)

    with open("./models/scaler.pkl", "rb") as f:
        scaler = joblib.load(f)

    with open("./models/kmeans_model.pkl", "rb") as f:
        kmeans = joblib.load(f)

    return model, features, scaler, kmeans

def load_unique_brands():
    df = pd.read_csv(csv_path) 
    return df["Make"].unique().tolist()