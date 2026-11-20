from pathlib import Path
import pandas as pd

# File paths
RAW_PATH = Path("data/raw/Telco-Customer-Churn.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "Telco-Customer-Churn-Cleaned.csv"

# Load the original dataset
df = pd.read_csv(RAW_PATH)

print("Original dataset shape:", df.shape)

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values after conversion
print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"]).copy()

# Remove duplicate customer IDs, if any
df = df.drop_duplicates(subset=["customerID"])

# Create the processed-data folder
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Save the cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\n========== CLEANING SUMMARY ==========")
print("Cleaned dataset shape:", df.shape)
print("Remaining missing values:", df.isnull().sum().sum())
print("Duplicate customer IDs:", df["customerID"].duplicated().sum())
print("Saved cleaned dataset to:", OUTPUT_PATH)