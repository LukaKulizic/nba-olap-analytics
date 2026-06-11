import pandas as pd

df = pd.read_csv(
    "2023-2024 NBA Player Stats - Regular.csv",
    sep=";",
    encoding="latin1"
)

print(df.head())

top10 = df.sort_values(by="PTS", ascending=False).head(10)
print(top10[["Player", "PTS"]])

print(df["PTS"].describe())

import matplotlib.pyplot as plt

top10 = df.sort_values(by="PTS", ascending=False).head(10)

plt.figure()
plt.bar(top10["Player"], top10["PTS"])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 NBA scorers (2024/25)")
plt.ylabel("Points")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

features = ["MP", "FG", "FGA", "TRB", "AST", "STL", "BLK", "TOV"]

X = df[features]
y = df["PTS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Results:")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R2:", r2_rf)

print("\nMODEL COMPARISON:")
print("Linear Regression R2:", r2)
print("Random Forest R2:", r2_rf)