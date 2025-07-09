import streamlit as st
import pandas as pd
import joblib
from src.feature_extraction import preprocess_logs


st.set_page_config(page_title='BioSec Dashboard', layout='wide')
st.title("BioSec - Insider Threat Detection Prediction")

# Load and Process
st.header("Access Log Overview")
try:
    df_raw = pd.read_csv("data/synthetic_logs.csv")
    st.dataframe(df_raw.sample(10))

    model = joblib.load("models/biosec_iforest.pkl")

    X, _ = preprocess_logs()
    preds = model.predict(X)

    df_raw['Anomaly Score'] = preds
    df_raw['Anomaly Label'] = ['Anomaly' if p == 1 else 'Normal' for p in preds]
    
    st.header("Anomaly Detection Results")
    st.dataframe(df_raw[['user','ip','resource','action','timestamp','Anomaly Label']].tail(20))

    st.bar_chart(df_raw['Anomaly Label'].value_counts())
except Exception as e:
    st.error("Error Loading Data / Model",e)