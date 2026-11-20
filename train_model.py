from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Load cleaned dataset
DATA_PATH = Path("data/processed/Telco-Customer-Churn-Cleaned.csv")
df = pd.read_csv(DATA_PATH)

# Separate target from input features
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# Preprocess numerical columns
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Preprocess categorical columns
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Split data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

for name, classifier in models.items():
    # Keep preprocessing inside the pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # Train model
    model.fit(X_train, y_train)
    
    if name == "Logistic Regression":
        import joblib 
        MODEL_DIR = Path("models")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "churn_model.joblib")
        print("Model saved successfully!")

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluate performance
    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    results.append(result)

    print(f"\n========== {name} ==========")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Stayed", "Churned"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

# Compare models
results_df = pd.DataFrame(results).sort_values(
    by="ROC-AUC",
    ascending=False,
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

# Save comparison results
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)

print("\nResults saved to reports/model_comparison.csv")







# Save the trained Logistic Regression model
import joblib
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Reload the data and train the model pipeline
# The easiest fix is to use the complete updated script instead.