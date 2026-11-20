from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
DATA_PATH = Path("data/processed/Telco-Customer-Churn-Cleaned.csv")
df = pd.read_csv(DATA_PATH)

# Create folder for charts
CHART_DIR = Path("reports/figures")
CHART_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(CHART_DIR / "churn_distribution.png")
plt.show()

# 2. Churn by contract type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(CHART_DIR / "churn_by_contract.png")
plt.show()

# 3. Monthly charges by churn
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig(CHART_DIR / "monthly_charges_by_churn.png")
plt.show()

# 4. Calculate churn rate by contract
contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .round(2)
)

print("\n========== CHURN RATE BY CONTRACT (%) ==========")
print(contract_churn)

# 5. Calculate churn rate by internet service
internet_churn = (
    df.groupby("InternetService")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .round(2)
)

print("\n========== CHURN RATE BY INTERNET SERVICE (%) ==========")
print(internet_churn)

print("\nCharts saved in:", CHART_DIR)