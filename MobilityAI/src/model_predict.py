import pandas as pd
import joblib
from features_engineering import engineer_features

def predict_new_trip(input_csv="data/*.csv"):
    df = pd.read_csv(input_csv)
    X, _ = engineer_features(df)
    model = joblib.load('momodels/GDB_model.pkl')
    predictions = model.predict(X)
    df['predicted_duration'] = predictions
    return df[['pickup_lat','pickup_lon','predicted_duration']]