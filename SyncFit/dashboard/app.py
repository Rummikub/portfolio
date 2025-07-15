import streamlit as st
import pandas as pd
import joblib
from src.feature_builder import build_user_features

st.set_page_config(page_title="SyncFit Dashboard", layout="wide")
st.title("🏃‍♀️ SyncFit – Wearable Churn Analytics")

st.header("📥 Upload Synthetic Wearable Logs")
uploaded_file = st.file_uploader("Upload a CSV file (must include steps, heart_rate, etc.)", type="csv")

if uploaded_file:
    path = "data/user_uploaded.csv"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        model = joblib.load("models/syncfit_churn_model.pkl")
        X, _ = build_user_features(path)
        predictions = model.predict(X)
        X['predicted_churn'] = predictions
        X['user_id'] = X.index

        st.success("Churn Prediction Complete ✅")
        st.dataframe(X[['user_id', 'predicted_churn']])
        st.bar_chart(X['predicted_churn'].value_counts())

    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Upload a file to begin churn prediction.")
