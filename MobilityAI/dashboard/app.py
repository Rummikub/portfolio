import streamlit as st 
import pandas as pd
import os


st.set_page_config(page_title="MobilityAI Dashboard", layout='wide')
st.title("MobilityAI : Ridership Prediction & Exploration")


st.header("Sample NYC Taxi Data")

if os.path.exists("data/yellow_trip_sample.parquet"):
    df = pd.read_parquet("data/yellow_trip_sample.parquet")
    st.dataframe(df.head())

    st.map(df[['pickup_latitude', 'pickup_longitude']].dropna().rename(
        columns={"pickup_latitude": "lat", "pickup_longitude": "lon"}))
else:
    st.warning("No TLC data found. Place `yellow_trip_sample.parquet` under `data/` folder.")


st.header("Predicted Ridership")
uploaded_file = st.file_uploader("Upload a CSV file with new taxi trips", type="csv")

if uploaded_file:
    new_df = pd.read_csv(uploaded_file)
    new_df.to_csv("data/new_trips.csv", index=False)

    try:
        from src.model_predict import predict_new_trip
        prediction_df = predict_new_trip("data/new_trips.csv")

        st.success("Prediction completed!")
        st.dataframe(prediction_df)

        st.bar_chart(prediction_df['predicted_duration_sec'])

    except Exception as e:
        st.error(f"Failed to generate predictions: {e}")