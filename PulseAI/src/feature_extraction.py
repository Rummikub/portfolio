import pandas as pd
import json

def load_jsonl(path):
    with open(path) as f:
        return pd.DataFrame([json.loads(line) for line in f])

def extract_session_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['timestamp'].dt.hour
    df['event_code'] = df['event_type'].astype("category").cat.codes
    df['device_code'] = df['device'].astype("category").cat.codes

    agg = df.groupby("user_id").agg({
        "event_code": ["nunique", "mean"],
        "device_code": "nunique",
        "hour": "nunique"
    })
    agg.columns = ["event_type_diversity", "avg_event_code", "device_diversity", "active_hours"]
    return agg.reset_index()
