import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# UČITAVANJE PODATAKA


df = pd.read_csv(
    "2023-2024 NBA Player Stats - Regular.csv",
    sep=";",
    encoding="latin1"
)

print("PRVIH 5 REDOVA:")
print(df.head())


# TOP 10 STRELACA


top10 = df.sort_values(by="PTS", ascending=False).head(10)

print("\nTOP 10 STRELACA:")
print(top10[["Player", "PTS"]])


# DESKRIPTIVNA STATISTIKA


print("\nDESKRIPTIVNA STATISTIKA ZA PTS:")
print(df["PTS"].describe())


# GRAFIKON TOP 10 STRELACA


plt.figure(figsize=(10, 6))

plt.bar(top10["Player"], top10["PTS"])

plt.xticks(rotation=45, ha="right")
plt.title("Top 10 NBA Scorers (2023/24)")
plt.ylabel("Points per Game")

plt.tight_layout()

plt.savefig("top10_scorers.png")
plt.show()


# BOX PLOT ZA PTS


plt.figure(figsize=(6, 6))

plt.boxplot(df["PTS"])

plt.title("Box Plot of Points Per Game (PTS)")
plt.ylabel("PTS")

plt.tight_layout()

plt.savefig("pts_boxplot.png")
plt.show()


# PRIPREMA PODATAKA


features = ["MP", "FG", "FGA", "TRB", "AST", "STL", "BLK", "TOV"]

X = df[features]
y = df["PTS"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# LINEARNA REGRESIJA


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLINEAR REGRESSION RESULTS")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)


# GRAFIKON LINEARNE REGRESIJE


plt.figure(figsize=(6, 6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual PTS")
plt.ylabel("Predicted PTS")
plt.title("Linear Regression: Actual vs Predicted")

plt.tight_layout()

plt.savefig("linear_regression.png")
plt.show()


# RANDOM FOREST


rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRANDOM FOREST RESULTS")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R2:", r2_rf)


# GRAFIKON RANDOM FOREST MODELA


plt.figure(figsize=(6, 6))

plt.scatter(y_test, y_pred_rf)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual PTS")
plt.ylabel("Predicted PTS")
plt.title("Random Forest: Actual vs Predicted")

plt.tight_layout()

plt.savefig("random_forest.png")
plt.show()


# POREĐENJE MODELA


print("\nMODEL COMPARISON")
print("Linear Regression R2:", r2)
print("Random Forest R2:", r2_rf)

print("\nSačuvani grafikoni:")
print("top10_scorers.png")
print("pts_boxplot.png")
print("linear_regression.png")
print("random_forest.png")