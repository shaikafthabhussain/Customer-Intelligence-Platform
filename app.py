
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "Telco-Customer-Churn-Cleaned.csv"
)

st.title("📊 Customer Intelligence Platform")
st.subheader("Customer Churn Prediction Dashboard")

st.write(
    "Analyze customer data, understand historical churn patterns, "
    "and estimate which customers may leave the service."
)

# =====================================================
# 2. LOAD DATA AND MODEL
# =====================================================

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


df = load_data(str(DATA_PATH))
model = load_model(str(MODEL_PATH))

required_columns = [
    "Churn", "gender", "SeniorCitizen", "Partner",
    "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Missing dataset columns: {missing}")
    st.stop()

df["Churn"] = df["Churn"].astype(str).str.strip()

df["MonthlyCharges"] = pd.to_numeric(
    df["MonthlyCharges"], errors="coerce"
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["tenure"] = pd.to_numeric(
    df["tenure"], errors="coerce"
)

df = df.dropna(
    subset=["Churn", "MonthlyCharges", "TotalCharges", "tenure"]
)

if df.empty:
    st.error("No valid customer records found.")
    st.stop()

# Model input columns
features = [
    c for c in df.columns
    if c not in ["Churn", "customerID"]
]

if not hasattr(model, "predict_proba"):
    st.error("The saved model does not support probability predictions.")
    st.stop()

if not hasattr(model, "classes_"):
    st.error("The saved model does not expose its class labels.")
    st.stop()

# =====================================================
# 3. DASHBOARD METRICS
# =====================================================

total_customers = len(df)
churned = int((df["Churn"] == "Yes").sum())
retained = int((df["Churn"] == "No").sum())

churn_rate = (
    churned / total_customers * 100
    if total_customers else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Customers Churned", f"{churned:,}")
c3.metric("Customers Retained", f"{retained:,}")
c4.metric("Historical Churn Rate", f"{churn_rate:.2f}%")

st.divider()

# =====================================================
# 4. HISTORICAL CUSTOMER CHURN CHARTS
# =====================================================

st.header("Historical Customer Churn")

left, right = st.columns(2)

with left:
    st.subheader("Customer Distribution")

    counts = (
        df["Churn"]
        .value_counts()
        .reindex(["No", "Yes"], fill_value=0)
        .rename_axis("Churn")
        .reset_index(name="Customers")
    )

    fig1 = px.bar(
        counts,
        x="Churn",
        y="Customers",
        color="Churn",
        text="Customers",
        title="Customer Churn Distribution",
        color_discrete_map={
            "No": "#60A5FA",
            "Yes": "#F97373",
        },
    )

    fig1.update_layout(showlegend=False)
    fig1.update_yaxes(
        rangemode="tozero",
        tickformat=",d"
    )

    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("Churn by Contract Type")

    contract_data = (
        df.groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig2 = px.bar(
        contract_data,
        x="Contract",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Churn by Contract Type",
        color_discrete_map={
            "No": "#60A5FA",
            "Yes": "#F97373",
        },
    )

    fig2.update_yaxes(rangemode="tozero")

    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Monthly Charges by Churn")

fig3 = px.box(
    df,
    x="Churn",
    y="MonthlyCharges",
    color="Churn",
    title="Monthly Charges: Retained vs Churned",
    color_discrete_map={
        "No": "#60A5FA",
        "Yes": "#F97373",
    },
)

st.plotly_chart(fig3, use_container_width=True)

st.divider()

# =====================================================
# 5. CUSTOMER INPUT FORM
# =====================================================

st.header("🔍 Predict Individual Customer Churn")

st.write("Enter customer details and click Predict Churn.")


def options_for(column):
    return sorted(df[column].dropna().unique().tolist())


def selected_index(options, value):
    return options.index(value) if value in options else 0


with st.form("customer_form"):

    left, right = st.columns(2)

    with left:
        gender = st.selectbox(
            "Gender", options_for("gender")
        )

        senior = st.selectbox(
            "Senior Citizen",
            options_for("SeniorCitizen"),
            format_func=lambda x: "Yes" if int(x) == 1 else "No",
        )

        partner = st.selectbox(
            "Has Partner?", options_for("Partner")
        )

        dependents = st.selectbox(
            "Has Dependents?", options_for("Dependents")
        )

        tenure = st.slider(
            "Tenure (months)",
            0,
            int(df["tenure"].max()),
            min(12, int(df["tenure"].max())),
        )

        phone = st.selectbox(
            "Phone Service", options_for("PhoneService")
        )

        lines = st.selectbox(
            "Multiple Lines", options_for("MultipleLines")
        )

        internet = st.selectbox(
            "Internet Service", options_for("InternetService")
        )

        security = st.selectbox(
            "Online Security", options_for("OnlineSecurity")
        )

        backup = st.selectbox(
            "Online Backup", options_for("OnlineBackup")
        )

    with right:
        device = st.selectbox(
            "Device Protection", options_for("DeviceProtection")
        )

        support = st.selectbox(
            "Tech Support", options_for("TechSupport")
        )

        tv = st.selectbox(
            "Streaming TV", options_for("StreamingTV")
        )

        movies = st.selectbox(
            "Streaming Movies", options_for("StreamingMovies")
        )

        contract_options = options_for("Contract")

        contract = st.selectbox(
            "Contract Type",
            contract_options,
            index=selected_index(
                contract_options, "Month-to-month"
            ),
        )

        paperless = st.selectbox(
            "Paperless Billing", options_for("PaperlessBilling")
        )

        payment = st.selectbox(
            "Payment Method", options_for("PaymentMethod")
        )

        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=float(df["MonthlyCharges"].max()),
            value=float(df["MonthlyCharges"].median()),
        )

        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=max(1.0, float(df["TotalCharges"].max())),
            value=float(df["TotalCharges"].median()),
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
    )

# =====================================================
# 6. PREDICTION AND SHAP EXPLANATION
# =====================================================

if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    customer_df = pd.DataFrame(
        [customer]
    ).reindex(columns=features)

    # Match training data types
    for column in customer_df.columns:
        try:
            customer_df[column] = customer_df[column].astype(
                df[column].dtype
            )
        except (ValueError, TypeError):
            pass

    try:
        # ---------------------------------------------
        # MODEL PREDICTION
        # ---------------------------------------------

        prediction = model.predict(customer_df)[0]
        probabilities = model.predict_proba(customer_df)[0]

        classes = list(model.classes_)

        if "Yes" in classes:
            churn_index = classes.index("Yes")
        elif 1 in classes:
            churn_index = classes.index(1)
        else:
            churn_index = len(classes) - 1

        probability = float(probabilities[churn_index])

        st.divider()
        st.header("Prediction Result")

        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                "Estimated Churn Probability",
                f"{probability * 100:.2f}%",
            )
            st.progress(probability)

        with r2:
            if str(prediction) == "Yes" or prediction == 1:
                st.error(
                    "⚠️ Model predicts this customer may churn."
                )
            else:
                st.success(
                    "✅ Model predicts this customer may stay."
                )

        st.caption(
            "This is an estimate based on historical data, "
            "not a guarantee of future customer behavior."
        )

        # ---------------------------------------------
        # INDIVIDUAL CUSTOMER SHAP EXPLANATION
        # ---------------------------------------------

        st.divider()
        st.subheader("🔎 Why did the model make this prediction?")

        try:
            if not hasattr(model, "named_steps"):
                raise ValueError(
                    "Saved model is not a preprocessing pipeline."
                )

            preprocessor = model.named_steps.get("preprocessor")
            classifier = model.named_steps.get("classifier")

            if preprocessor is None or classifier is None:
                raise ValueError(
                    "Could not find pipeline steps named "
                    "'preprocessor' and 'classifier'."
                )

            # Transform raw categorical data into numeric features.
            background_raw = df[features].sample(
                n=min(30, len(df)),
                random_state=42,
            )

            background_transformed = preprocessor.transform(
                background_raw
            )

            customer_transformed = preprocessor.transform(
                customer_df
            )

            # Convert sparse matrices into dense arrays.
            if hasattr(background_transformed, "toarray"):
                background_transformed = background_transformed.toarray()

            if hasattr(customer_transformed, "toarray"):
                customer_transformed = customer_transformed.toarray()

            background_transformed = np.asarray(
                background_transformed,
                dtype=float
            )

            customer_transformed = np.asarray(
                customer_transformed,
                dtype=float
            )

            # Get names of encoded features.
            try:
                feature_names = list(
                    preprocessor.get_feature_names_out()
                )
            except Exception:
                feature_names = [
                    f"Feature {i + 1}"
                    for i in range(background_transformed.shape[1])
                ]

            if len(feature_names) != background_transformed.shape[1]:
                feature_names = [
                    f"Feature {i + 1}"
                    for i in range(background_transformed.shape[1])
                ]

            # Predict churn probability using transformed inputs.
            def predict_transformed_churn(data):
                data = np.asarray(data, dtype=float)

                return classifier.predict_proba(data)[:, churn_index]

            explainer = shap.Explainer(
                predict_transformed_churn,
                shap.maskers.Independent(background_transformed),
                algorithm="permutation",
            )

            with st.spinner("Calculating SHAP explanation..."):
                shap_result = explainer(
                    customer_transformed,
                    max_evals=(
                        2 * customer_transformed.shape[1] + 1
                    ),
                )

            shap_values = np.asarray(
                shap_result.values
            ).reshape(
                -1, customer_transformed.shape[1]
            )[0]

            explanation = pd.DataFrame({
                "Encoded Feature": feature_names,
                "Customer Value": customer_transformed[0],
                "SHAP Impact": shap_values,
            })

            explanation["Effect"] = np.where(
                explanation["SHAP Impact"] > 0,
                "Increases churn risk",
                "Decreases churn risk",
            )

            explanation["Absolute Impact"] = (
                explanation["SHAP Impact"].abs()
            )

            explanation = explanation.sort_values(
                "Absolute Impact",
                ascending=False,
            ).head(10)

            st.write(
                "These are the strongest individual feature contributions "
                "to this customer's predicted churn probability. "
                "Categorical features are shown in encoded form."
            )

            st.dataframe(
                explanation[
                    [
                        "Encoded Feature",
                        "Customer Value",
                        "SHAP Impact",
                        "Effect",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            # SHAP impact chart
            fig_shap = px.bar(
                explanation.sort_values("SHAP Impact"),
                x="SHAP Impact",
                y="Encoded Feature",
                color="Effect",
                orientation="h",
                title="Individual Customer SHAP Explanation",
                color_discrete_map={
                    "Increases churn risk": "#F97373",
                    "Decreases churn risk": "#60A5FA",
                },
            )

            fig_shap.update_layout(
                xaxis_title="Contribution to predicted churn probability",
                yaxis_title="Encoded model feature",
            )

            fig_shap.add_vline(
                x=0,
                line_dash="dash",
                line_color="gray",
            )

            st.plotly_chart(
                fig_shap,
                use_container_width=True,
            )

            st.caption(
                "Positive SHAP values increase the model's predicted "
                "churn probability relative to the background baseline. "
                "Negative SHAP values decrease it. These values explain "
                "model behaviour, not real-world cause and effect."
            )

        except Exception as shap_error:
            st.warning(
                f"SHAP explanation could not be generated: {shap_error}"
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# =====================================================
# 7. FOOTER
# =====================================================

st.divider()

st.caption(
    "Customer Intelligence Platform | Python | Pandas | "
    "Scikit-learn | SHAP | Streamlit | Plotly"
)
