import streamlit as st
import pandas as pd
import pydeck as pdk
import os
from src.model_predict import predict_new_trip

st.set_page_config(page_title='MobilityAI Dashboard', layout='wide')
st.title("MobilityAI - Personalized Trip Duration Prediction")
st.header("NYC Trip Taxi Data")

if os.path.exists("data/yellow_tripdata_2021-01.parquet"):
    df = pd.read_parquet("data/yellow_tripdata_2021-01.parquet")
    st.dataframe(df.head())

    st.subheader("Pickup Locations Map (# 1,000)")
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=40.75, longitude=-73.95, zoom=10, pitch=50
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df[['pickup_lat', 'pickup_lon']].dropna().sample(1000),
                get_position='[pickup_lon, pickup_lat]',
                get_color='[255, 165, 0, 160]',
                get_radius=100,
            ),
        ],
    ))

else:
    st.warning("No sample TLC data found. Please place `yellow_trip_sample.parquet` under the `data/` folder.")

# Prediction (at this point the user can add new dataset to predict)
st.header("Predict Duration on New Trips")
uploaded_file = st.file_uploader("Upload a CSV file with new taxi trip data", type="csv")

if uploaded_file:
    new_df = pd.read_csv(uploaded_file)
    new_df.to_csv("data/new_trips.csv", index=False)

    try:
        prediction_df = predict_new_trip("data/new_trips.csv")
        st.success("Prediction completed!")

        st.dataframe(prediction_df)

        st.subheader("Trip Duration Distribution (sec)")
        st.bar_chart(prediction_df['predicted_duration'])

    except Exception as e:
        st.error(f"Warning: Failed to generate predictions: {e}")