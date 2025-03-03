import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

csv_path = os.path.join(BASE_DIR, "data", "fuel_consumption.csv")

df = pd.read_csv(csv_path)

df = pd.get_dummies(df, drop_first=True)
df.head()
target_column = "CO2 Emissions(g/km)"
correlations = df.corr()[target_column].sort_values(ascending=False)

top_features = correlations.head(10)

plt.figure(figsize=(10, 8))
sns.barplot(x=top_features.values, y=top_features.index, palette="coolwarm", width=0.9)
plt.xlabel("Korelacija sa CO₂ Emisijom")
plt.ylabel("Karakteristike")
plt.title("Top 10 faktora koji utiču na emisiju CO₂")

plt.yticks(fontsize=6, rotation=45)

plt.show()
