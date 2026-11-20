from pathlib import Path
import pandas as pd

# Dataset location
DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")

# Check whether the dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())

print("\n========== COLUMN INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print(f"Duplicate rows: {df.duplicated().sum()}")

print("\n========== CHURN DISTRIBUTION ==========")
print(df["Churn"].value_counts())

print("\n========== CHURN PERCENTAGES ==========")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

print("\n========== TOTAL CHARGES ==========")
print("Original data type:", df["TotalCharges"].dtype)

total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Values that cannot be converted:", total_charges.isnull().sum())