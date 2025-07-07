import pandas as pd

def load_data(path):
    df = pd.read_parquet(path)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])    
    df['trip_duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    return df

def main():
    df = load_data("../data/yellow_tripdata_2021-01.parquet")
    # Peek!
    df.head()
    
if __name__ == "__main__":
    main()