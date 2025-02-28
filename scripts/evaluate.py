import pandas as pd
import time
import numpy as np
import os
from rich.console import Console
from rich.table import Table

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

csv_path = os.path.join(BASE_DIR, "data", "fuel_consumption.csv")

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

X = df[selected_features]
y = df["CO2 Emissions(g/km)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
}

results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    mean_cv_r2 = np.mean(cv_scores)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, mse, rmse, r2, mean_cv_r2, train_time, predict_time])

console = Console()
table = Table(title="Model Performance Results")

columns = ["Model", "MAE", "MSE", "RMSE", "R² Score", "Mean CV R²", "Train Time (s)", "Predict Time (s)"]
for col in columns:
    table.add_column(col, justify="center")

for row in results:
    table.add_row(row[0], f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}", f"{row[5]:.4f}", f"{row[6]:.4f}", f"{row[7]:.4f}")

console.print(table)
