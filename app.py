import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

# -------------------------
# Load CSS
# -------------------------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# -------------------------
# Load Model
# -------------------------
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("breast-cancer.csv")

df = load_data()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("🩺 Navigation")

page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 Visualization",
        "🤖 Prediction",
        "📂 Batch Prediction"
    ]
)

# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("🩺 Breast Cancer Classification")

    st.markdown("---")

    st.markdown("""
### Welcome

This application predicts whether a breast tumor is:

- ✅ Benign
- ❌ Malignant

using a Logistic Regression model.

The dashboard also contains:

- Dataset Preview
- Correlation Heatmap
- Outlier Detection
- Single Prediction
- Batch Prediction
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Features", df.shape[1] - 2)

# ============================================================
# DATASET
# ============================================================

elif page == "📊 Dataset":

    st.title("Dataset")

    st.dataframe(df)

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Missing Values")

    st.write(df.isnull().sum())

    st.subheader("Statistics")

    st.dataframe(df.describe())

# ============================================================
# VISUALIZATION
# ============================================================

elif page == "📈 Visualization":

    st.title("Data Visualization")

    st.subheader("Correlation Heatmap")

    corr = df.drop(columns=["id"]).corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        corr,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Diagnosis Distribution")

    fig2, ax2 = plt.subplots()

    sns.countplot(
        x=df["diagnosis"],
        palette="Blues",
        ax=ax2
    )

    st.pyplot(fig2)

# ============================================================
# PREDICTION
# ============================================================

elif page == "🤖 Prediction":

    st.title("🩺 Breast Cancer Prediction")

    feature_names = list(df.drop(columns=["id", "diagnosis"]).columns)

    inputs = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):

        default_value = float(df[feature].mean())

        if i % 2 == 0:
            inputs[feature] = col1.number_input(
                feature,
                value=default_value,
                format="%.5f"
            )
        else:
            inputs[feature] = col2.number_input(
                feature,
                value=default_value,
                format="%.5f"
            )

    if st.button("🔍 Predict"):

        input_df = pd.DataFrame([inputs])

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0]

        confidence = np.max(probability) * 100

        if prediction == 1:

            st.error("## 🔴 Malignant")

        else:

            st.success("## 🟢 Benign")

        st.progress(float(confidence / 100))

        st.write(f"### Confidence : {confidence:.2f}%")

# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "📂 Batch Prediction":

    st.title("📂 Batch Prediction")

    uploaded = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded is not None:

        data = pd.read_csv(uploaded)

        st.subheader("Preview")

        st.dataframe(data.head())

        if "id" in data.columns:
            ids = data["id"]
            data = data.drop(columns=["id"])
        else:
            ids = None

        predictions = model.predict(data)

        probability = model.predict_proba(data)

        confidence = probability.max(axis=1)

        labels = encoder.inverse_transform(predictions)

        result = data.copy()

        if ids is not None:
            result.insert(0, "id", ids)

        result["Prediction"] = labels
        result["Confidence"] = confidence

        st.success("Prediction Completed")

        st.dataframe(result)

        csv = result.to_csv(index=False).encode()

        st.download_button(
            "⬇ Download Prediction",
            csv,
            "prediction.csv",
            "text/csv"
        )
