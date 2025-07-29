import pandas as pd
def engineer_features(df):
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    X = df[['hour', 'day_of_week', 'feature1', 'feature2']]  # replace with actual features
    y = df['label']
    return X, y