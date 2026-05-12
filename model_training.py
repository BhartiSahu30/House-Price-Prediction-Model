# =========================================
# HOUSE PRICE PREDICTION - MODEL TRAINING
# =========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------------------
# LOAD DATASET
# -----------------------------------------

df = pd.read_csv("train.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -----------------------------------------
# DATA CLEANING
# -----------------------------------------

# Numerical Columns
num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical Columns
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert Categorical Data
df = pd.get_dummies(df, drop_first=True)

# -----------------------------------------
# FEATURES & TARGET
# -----------------------------------------
X = df.drop("totalprice", axis=1)
y = df["totalprice"]
# -----------------------------------------
# TRAIN TEST SPLIT
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------------
# MODEL TRAINING
# -----------------------------------------


model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# -----------------------------------------
# PREDICTIONS
# -----------------------------------------

y_pred = model.predict(X_test)

# -----------------------------------------
# EVALUATION
# -----------------------------------------

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Evaluation")
print("---------------------")

print("R2 Score :", r2)
print("MAE :", mae)

# -----------------------------------------
# SAVE MODEL
# -----------------------------------------

with open("models/house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel Saved Successfully")

# -----------------------------------------
# ACTUAL VS PREDICTED PLOT
# -----------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.savefig("outputs/actual_vs_predicted.png")

plt.show()

# -----------------------------------------
# HEATMAP
# -----------------------------------------

plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.savefig("outputs/heatmap.png")

plt.show()

# -----------------------------------------
# SAMPLE PREDICTIONS
# -----------------------------------------

print("\nSample Predictions")
print(y_pred[:5])