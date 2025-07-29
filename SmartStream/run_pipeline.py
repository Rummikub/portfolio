from src.data_ingestion import load_data
from src.feature_engineering import engineer_features
from src.train import train_model

def run_all():
    print(" Loading data...")
    df = load_data("data/mobility_data_sample.csv")

    print(" Engineering features...")
    X, y = engineer_features(df)

    print(" Training model...")
    train_model(X, y)

if __name__ == "__main__":
    run_all()