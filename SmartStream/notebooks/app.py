import streamlit as st
import pandas as pd
import joblib
from src.feature_engineering import engineer_features

st.set_page_config(page_title="SmartStream Dashboard", layout="wide")
st.title("📊 SmartStream: Personalized ML Insights")

option = st.selectbox(
    "Choose a use case to explore:",
    ["🎵 SongMood Classifier", "🚕 Mobility Trip Duration", "📺 Content Recommender"]
)

if option == "SongMood Classifier":
    st.header("Classify Songs by Mood")
    uploaded = st.file_uploader("Upload a song feature CSV", type="csv", key="music")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        try:
            model = joblib.load("models/smartstream_model.pkl")
            X, _ = engineer_features(df)  # assumes it works for songs too
            predictions = model.predict(X)
            df['Predicted Mood'] = predictions
            st.success("Prediction Complete!")
            st.dataframe(df[['song_id', 'Predicted Mood']])
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif option == "🚕 Mobility Trip Duration":
    st.header("Predict Trip Duration")
    uploaded = st.file_uploader("Upload a mobility trip CSV", type="csv", key="mobility")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        try:
            model = joblib.load("models/smartstream_model.pkl")
            X, _ = engineer_features(df)
            predictions = model.predict(X)
            df['Predicted Duration'] = predictions
            st.success("Prediction Complete!")
            st.dataframe(df[['trip_id', 'Predicted Duration']])
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif option == "📺 Content Recommender":
    st.header("🎬 Recommender System")
    st.info("Coming soon: Personalized movie/show recommendations based on embeddings or user history.")
