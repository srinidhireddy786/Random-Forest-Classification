import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Random Forest Classification",
    layout="wide"
)

# =========================
# Modern CSS
# =========================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

/* Title */
h1 {
    text-align: center;
    font-size: 58px !important;
    font-weight: 800;
    color: #00d4ff;
    text-shadow: 0px 0px 15px rgba(0,212,255,0.5);
}

/* Headers */
h2, h3 {
    color: #7ed6ff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
}

/* Sidebar Labels */
section[data-testid="stSidebar"] label {
    color: white !important;
    font-size: 18px !important;
}

/* Inputs */
div[data-baseweb="input"] input {
    background-color: rgba(255,255,255,0.08);
    color: white;
    border-radius: 12px;
    font-size: 18px !important;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 16px;
    border: none;
    font-size: 22px !important;
    font-weight: bold;
    color: white;
    background: linear-gradient(to right, #00c6ff, #0072ff);
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
}

/* Metric Value */
[data-testid="stMetricValue"] {
    color: #00ffcc;
    font-size: 34px !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Load Dataset
# =========================
data = load_wine()

X = data.data
y = data.target

feature_names = data.feature_names

target_names = data.target_names

# =========================
# DataFrame
# =========================
df = pd.DataFrame(
    X,
    columns=feature_names
)

df["target"] = y

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Load Model & Scaler
# =========================
model = joblib.load(
    "models/random_forest_classifier.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# =========================
# Scale Data
# =========================
X_test_scaled = scaler.transform(X_test)

# =========================
# Predictions
# =========================
y_pred = model.predict(X_test_scaled)

# =========================
# Metrics
# =========================
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

# =========================
# Title
# =========================
st.title("Random Forest Classification App")

st.write("""
Wine classification using Random Forest Classifier.
""")

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("Enter Wine Features")

inputs = []

for feature in feature_names:

    value = st.sidebar.number_input(
        feature,
        value=float(df[feature].mean()),
        format="%.2f"
    )

    inputs.append(value)

# =========================
# Prediction
# =========================
if st.sidebar.button("Predict Wine Type"):

    input_data = np.array([inputs])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    model_name = "Random Forest Classifier"

    predicted_class = target_names[prediction[0]]

    st.success(
        f"Model : {model_name}"
    )

    st.success(
        f"Predicted Wine Class : {predicted_class}"
    )

# =========================
# Statistics
# =========================
st.subheader("Model Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "Precision",
    f"{precision * 100:.2f}%"
)

col3.metric(
    "Recall",
    f"{recall * 100:.2f}%"
)

col4.metric(
    "F1 Score",
    f"{f1 * 100:.2f}%"
)

# =========================
# Dataset Preview
# =========================
st.subheader("Dataset Preview")

st.dataframe(df.head(15))

# =========================
# Confusion Matrix
# =========================
st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    columns=target_names,
    index=target_names
)

st.dataframe(cm_df)

# =========================
# Feature Importance
# =========================
st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

fig, ax = plt.subplots(figsize=(8, 6))

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax.set_title("Feature Importance")

st.pyplot(fig)

# =========================
# Footer
# =========================
st.markdown("""
---
### Built with Streamlit & Scikit-learn
""")