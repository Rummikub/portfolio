import pandas as pd
from src.data_ingestion import load_data
df = load_data("data/yellow_tripdata_2025-01.parquet")

def load_data(path):
    df = pd.read_parquet(path)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])    
    df['trip_duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    
    df.rename(columns={
        'pickup_longitude': 'pickup_lon',
        'pickup_latitude': 'pickup_lat',
        'dropoff_longitude': 'dropoff_lon',
        'dropoff_latitude': 'dropoff_lat'
    }, inplace=True)
    
    return df