import streamlit as st
import pandas as pd
import joblib
from src.feature_builder import build_user_features
from src.guardian_alerts import generate_guardian_alerts
from src.llm_alerts import generate_alert_message  # Optional OpenAI integration

st.set_page_config(page_title="SyncFit Dashboard", layout="wide")
st.title("SyncFit -Wearable Health Monitoring")

st.sidebar.title("Navigation")
view = st.sidebar.radio("Choose a view:", ["Churn Prediction", "Guardian Alerts"])

if view == "Churn Prediction":
    st.header("Upload Synthetic Wearable Logs")
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

elif view == "Guardian Alerts":
    st.header("Guardian Alert System")
    try:
        df = generate_guardian_alerts("data/synthetic_wearable_logs.csv")
        alert_df = df[df['alert'] != '✅ Normal']

        st.subheader("Users Requiring Attention")
        st.dataframe(alert_df)

        st.bar_chart(alert_df['alert'].value_counts())

        if st.checkbox("Generate Alerts - LLM Based"):
            for _, row in alert_df.iterrows():
                message = generate_alert_message(
                    user_id=row['user_id'],
                    deviation=row['deviation'],
                    hours=row['inactive_hours']
                )
                st.markdown(f"**User {row['user_id']}**: {message}")

    except Exception as e:
        st.error(f"Failed to load guardian alerts: {e}")
