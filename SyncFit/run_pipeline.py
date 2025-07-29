# run_pipeline.py
from src.data_simulator import generate_data
from src.feature_builder import build_user_features
from src.churn_model import train_churn_model
from src.guardian_alerts import run_guardian_alerts

def main():
    print("Generating synthetic data...")
    generate_data("data/synthetic_wearable_logs.csv")

    print("Building features...")
    X, y = build_user_features("data/synthetic_wearable_logs.csv")

    print("Training churn model...")
    train_churn_model(X, y)

    print("Running Guardian Alert System...")
    run_guardian_alerts("data/synthetic_wearable_logs.csv")

if __name__ == "__main__":
    main()
