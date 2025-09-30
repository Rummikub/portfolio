import streamlit as st
import pandas as pd
import joblib
from src.feature_builder import build_user_features
from src.guardian_alerts import generate_guardian_alerts

# Try Anthropic first, fallback to OpenAI if needed
try:
    from src.llm_alerts_anthropic import generate_alert_message
    llm_provider = "Anthropic Claude"
except ImportError:
    try:
        from src.llm_alerts import generate_alert_message
        llm_provider = "OpenAI GPT"
    except ImportError:
        # Fallback function if no LLM is available
        def generate_alert_message(user_id, deviation, hours):
            return f"[No LLM configured] User {user_id}: {int(deviation*100)}% decline, {int(hours)}h inactive"
        llm_provider = "Fallback System"

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

        if st.checkbox(f"Generate AI Health Alerts (Using: {llm_provider})"):
            st.info(f"🤖 Using {llm_provider} for intelligent health recommendations")
            
            # Generate alerts for critical users
            for _, row in alert_df.head(5).iterrows():  # Limit to top 5 to save API calls
                with st.expander(f"Alert for {row['user_id']} - {row['alert']}", expanded=False):
                    with st.spinner(f"Generating AI alert..."):
                        message = generate_alert_message(
                            user_id=row['user_id'],
                            deviation=row['deviation'],
                            hours=row['inactive_hours']
                        )
                    st.markdown(message)
                    
                    # Show metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Activity Drop", f"{int(row['deviation']*100)}%")
                    with col2:
                        st.metric("Inactive", f"{row['inactive_hours']:.1f}h")

    except Exception as e:
        st.error(f"Failed to load guardian alerts: {e}")
