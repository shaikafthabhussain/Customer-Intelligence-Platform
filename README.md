# Customer Intelligence Platform
### Machine Learning-Based Customer Churn Prediction & Explainable Analytics

<p align="center">
  <img src="https://img.shields.io/badge/Python-Machine%20Learning-blue?logo=python">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange">
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-purple">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-red?logo=streamlit">
</p>

---

##  Live Application

[**Launch Customer Intelligence Platform**](https://customer-intelligence-platform-afthab.streamlit.app)

---

##  Overview

Customer Intelligence Platform is an end-to-end machine learning application designed to analyse customer behaviour, identify churn patterns, and predict the likelihood of customers leaving a service.

The platform integrates data preprocessing, exploratory data analysis, machine learning classification, and Explainable AI to transform customer data into meaningful insights.

Through an interactive dashboard, users can explore customer trends, analyse churn behaviour, evaluate individual customer risk, and understand the factors influencing predictions.

The objective is to support data-driven decision-making and help businesses develop proactive customer retention strategies.

---

##  System Architecture

The platform follows an end-to-end machine learning workflow:

1. **Data Ingestion** – Load customer data from the source dataset.
2. **Data Inspection & Cleaning** – Identify inconsistencies, handle missing values, and prepare data.
3. **Exploratory Data Analysis** – Analyse customer behaviour and churn patterns.
4. **Data Preprocessing** – Transform numerical and categorical features.
5. **Model Training** – Train machine learning classification models.
6. **Model Evaluation** – Evaluate predictive performance.
7. **Model Persistence** – Save the trained model for inference.
8. **Prediction** – Generate churn predictions and probabilities for customer inputs.
9. **Explainability** – Use SHAP to understand individual prediction contributions.
10. **Visualization** – Present insights and predictions through an interactive Streamlit dashboard.

---

##  Dataset

**IBM Telco Customer Churn Dataset**

Source: [IBM Telco Customer Churn – GitHub](https://github.com/IBM/telco-customer-churn-on-icp4d)

The dataset contains customer information related to demographics, service subscriptions, contract details, billing, and churn status.

### Data Categories

- Customer demographics
- Tenure and customer relationship duration
- Phone and internet services
- Service subscriptions
- Contract types
- Payment methods
- Monthly and total charges
- Customer churn status

The dataset is used to explore customer behaviour and develop a classification model to identify customers who may discontinue their services.

---

##  Features

###  Customer Analytics Dashboard

- Customer overview and retention insights
- Historical churn distribution
- Churn analysis by contract type
- Monthly charges analysis
- Interactive visualizations
- Customer behaviour exploration

###  Customer Churn Prediction

- Interactive customer input form
- Individual customer risk prediction
- Churn probability estimation
- Stay-or-churn prediction output
- Real-time inference through the web dashboard

###  Explainable AI

- Individual prediction explanations using SHAP
- Feature contribution analysis
- Identification of factors increasing churn risk
- Identification of factors reducing churn risk
- Interactive SHAP visualizations

---

##  Machine Learning

### Logistic Regression

A supervised classification algorithm used to estimate the probability of customer churn by learning relationships between customer attributes and churn outcomes.

### Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to capture patterns and interactions between customer features.

### Model Evaluation

The trained models are evaluated using classification evaluation techniques to understand their predictive behaviour and suitability for identifying customers at risk of churn.

---

##  Explainable AI — SHAP

SHAP (SHapley Additive exPlanations) is used to interpret machine learning predictions.

It explains how different customer attributes contribute to the model's prediction by showing which features increase or decrease the estimated churn probability.

### SHAP provides:

- Individual customer prediction explanations
- Feature contribution values
- Positive and negative prediction impacts
- Ranked contributing features
- Visual explanation of model behaviour

This helps make machine learning predictions more transparent and understandable for business users.

---

##  Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data processing and analysis |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Exploratory analysis |
| Scikit-learn | Machine learning |
| Logistic Regression | Classification |
| Random Forest | Ensemble learning |
| SHAP | Model explainability |
| Joblib | Model persistence |
| Plotly | Interactive charts |
| Streamlit | Web application |
| Git | Version control |
| GitHub | Source code management and collaboration |

---

##  Data Pipeline

**Raw Customer Data → Data Cleaning → Exploratory Analysis → Feature Preprocessing → Model Training → Model Evaluation → Model Persistence → Prediction → SHAP Explanation → Interactive Dashboard**

The pipeline maintains consistency between model training and prediction by using a reusable preprocessing and machine learning workflow.

---

##  Key Challenges & Solutions

### Data Quality

Handled missing, invalid, and inconsistent data values to ensure the dataset was suitable for analysis and model training.

### Feature Preprocessing

Applied appropriate preprocessing techniques to numerical and categorical customer attributes for machine learning compatibility.

### Model Interpretability

Integrated SHAP to explain individual predictions and understand how customer characteristics influence model outputs.

### Training–Inference Consistency

Used a unified machine learning pipeline to ensure customer inputs undergo the same transformations used during training.

### Interactive Prediction

Developed an interactive dashboard that allows users to enter customer information and obtain predictions without directly interacting with the underlying Python code.

---

##  Business Applications

Customer churn prediction can be applied across various industries where customer retention is important.

- **Telecommunications** – Identify subscribers likely to discontinue services.
- **Streaming Platforms** – Analyse subscription cancellation risk.
- **E-commerce** – Identify customers with declining engagement.
- **SaaS Businesses** – Monitor subscription retention.
- **Financial Services** – Understand customer attrition patterns.
- **Subscription Businesses** – Support targeted retention initiatives.

The platform can help businesses prioritise customer engagement, investigate churn risk, and make data-informed retention decisions.

---

##  Future Enhancements

- Large-scale data processing
- Customer segmentation using clustering
- Advanced machine learning models
- Automated retention recommendations
- Real-time data integration
- Automated model monitoring
- Scheduled model retraining
- Enhanced analytics and filtering

---

##  Author

**Shaik Afthab Hussain**

---

<p align="center">
  <i>Turning customer data into insights, predictions, and smarter retention decisions.</i>
</p>