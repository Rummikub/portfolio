import streamlit as st
import pandas as pd

st.title("🎬 SmartStream: Personalized Recommendations")

st.write("Select a domain to explore personalized ML outputs:")
option = st.selectbox("Choose:", ["Spotify Mood Classifier", "Uber Trip Duration", "Netflix Recommender"])

if option == "MUsic Mood Classifier":
    st.success("🎵 Classify songs by mood using audio features")
    uploaded = st.file_uploader("Upload song features CSV")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        st.write("[Mock prediction results here]")

elif option == "Trip Duration":
    st.success("🚕 Predict trip durations from location & time")
    uploaded = st.file_uploader("Upload Uber trip data")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        st.write("[Mock prediction results here]")

elif option == "Contents Recommender":
    st.success("📽️ Recommend movies based on user taste")
    st.write("Coming soon...")
