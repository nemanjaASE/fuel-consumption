import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

csv_path = os.path.join(BASE_DIR, "data", "fuel_consumption.csv")
models_dir = os.path.join(os.path.dirname(__file__), "../models")
model_path = os.path.join(models_dir, "co2_emission_model.pkl")
features_path = os.path.join(models_dir, "feature_columns.pkl")

df = pd.read_csv(csv_path)

categorical_cols = ["Fuel Type"]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

selected_features = [
    "Fuel Consumption City (L/100 km)", 
    "Fuel Consumption Comb (L/100 km)", 
    "Fuel Consumption Hwy (L/100 km)",
    "Engine Size(L)", 
    "Cylinders"
] + list(df.columns[df.columns.str.startswith(tuple(categorical_cols))])

joblib.dump(selected_features, features_path)
print("✅ Features saved as 'feature_columns.pkl'")

X = df[selected_features]
y = df["CO2 Emissions(g/km)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

joblib.dump(model, model_path)
print("✅ Model saved as 'co2_emission_model.pkl'")