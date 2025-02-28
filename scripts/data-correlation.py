import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

csv_path = os.path.join(BASE_DIR, "data", "fuel_consumption.csv")

df = pd.read_csv(csv_path)

df = pd.get_dummies(df, drop_first=True)
df.head()
target_column = "CO2 Emissions(g/km)"
correlations = df.corr()[target_column].sort_values(ascending=False)

print(correlations.head(10))