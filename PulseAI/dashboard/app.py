import streamlit as st
import pandas as pd

st.title("PulseAI – Real-Time User Session Insights")

try:
    df = pd.read_csv("data/features.csv")
    st.write("### Latest Aggregated Features")
    st.dataframe(df)
    st.bar_chart(df.set_index("user_id"))
except FileNotFoundError:
    st.warning("No data available. Run the pipeline first.")