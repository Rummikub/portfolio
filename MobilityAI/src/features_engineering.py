import numpy as np
import pandas as pds

def havesine(lat1, lon1, lat2, lon2):
    R = 6371 # Radius of the earth in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi/2) * np.sin(delta_phi/2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2) * np.sin(delta_lambda/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c

def engineer_features(df):
    df['distance'] = havesine(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
    df['hour'] = pd.to_datetime(df['pickup_datetime']).dt.hour
    df['weekday'] = pd.to_datetime(df['pickup_datetime']).dt.dayofweek
    
    features = df[['distance', 'hour', 'weekday']]
    labels = df['trip_duration']
    return features, labels