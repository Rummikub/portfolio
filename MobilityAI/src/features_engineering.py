import numpy as np
import pandas as pd
from src.features_engineering import engineer_features
X, y = engineer_features(df)
def havesine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth.

    This function uses the Haversine formula to compute the shortest distance
    over the earth's surface between two points specified by their latitude
    and longitude in decimal degrees.

    Parameters:
    lat1, lon1 -- Latitude and longitude of the first point in decimal degrees.
    lat2, lon2 -- Latitude and longitude of the second point in decimal degrees.

    Returns:
    Distance between the two points in kilometers.
    """

    R = 6371 # Radius of the earth in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi/2) * np.sin(delta_phi/2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2) * np.sin(delta_lambda/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c

def engineer_features(df):
    """
    Engineer features for a taxi trip dataset.

    This function calculates additional features from the input DataFrame, such as the distance between pickup and dropoff points, 
    the hour of the pickup, and the day of the week. It returns the engineered features and labels for model training.

    Parameters:
    df (pd.DataFrame): DataFrame containing the raw taxi trip data with columns 'pickup_latitude', 'pickup_longitude', 
                       'dropoff_latitude', 'dropoff_longitude', 'pickup_datetime', and 'trip_duration'.

    Returns:
    features (pd.DataFrame): DataFrame containing the engineered features 'distance', 'hour', and 'weekday'.
    labels (pd.Series): Series containing the target variable 'trip_duration'.
    """
    df['distance'] = havesine(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['weekday'] = df['pickup_datetime'].dt.dayofweek
    
    features = df[['distance', 'hour', 'weekday']]
    labels = df['trip_duration']
    return features, labels