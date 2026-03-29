import pandas as pd

# Define column names
columns = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]

for i in range(1, 22):
    columns.append(f"sensor_{i}")

# Load training data
train_df = pd.read_csv(
    "../data/train_FD001.txt",
    sep=r"\s+",
    header=None
)

# Assign column names
train_df.columns = columns

# Basic checks
print("Dataset shape:", train_df.shape)
print(train_df.head())
# ---- RUL LABEL CREATION ----

# Find max cycle for each engine
max_cycles = train_df.groupby("engine_id")["cycle"].max()

# Create RUL column
train_df["RUL"] = train_df.apply(
    lambda row: max_cycles[row["engine_id"]] - row["cycle"],
    axis=1
)

print("RUL column added")
print(train_df[["engine_id", "cycle", "RUL"]].head())
# ---- FEATURE SELECTION ----

# Drop columns not useful for prediction
drop_cols = ["engine_id", "cycle"]
X = train_df.drop(columns=drop_cols + ["RUL"])
y = train_df["RUL"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaling done")
# ---- MODEL TRAINING ----

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
print("Model MAE:", mae)
# ---- SAVE MODEL & SCALER ----

import joblib

joblib.dump(model, "../model/rul_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")

print("Model and scaler saved successfully")