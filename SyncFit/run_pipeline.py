# run_pipeline.py
from src.data_simulator import simulate_wearable_data
from src.feature_builder import build_user_features
from src.churn_model import train_churn_model
from src.guardian_alerts import generate_guardian_alerts
import os

def main():
    print("="*60)
    print("SYNCFIT ML PIPELINE - ORIGINAL VERSION")
    print("Airflow DAG + OpenAI LLM Integration")
    print("="*60)
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    print("\n1. Generating synthetic wearable data...")
    simulate_wearable_data(n_users=100, n_days=30)
    print("   ✓ Generated data for 100 users over 30 days")

    print("\n2. Building features...")
    X, y = build_user_features("data/synthetic_wearable_logs.csv")
    print(f"   ✓ Built features for {len(X)} users")
    print(f"   ✓ Features: {list(X.columns)[:5]}...")

    print("\n3. Training churn model...")
    print("   Classification Report:")
    model = train_churn_model()  # This function doesn't take parameters
    print("   ✓ XGBoost model trained and saved to models/syncfit_churn_model.pkl")

    print("\n4. Running Guardian Alert System...")
    alerts_df = generate_guardian_alerts("data/synthetic_wearable_logs.csv")
    
    # Show alert summary
    alert_counts = alerts_df['alert'].value_counts()
    print("\n   Alert Distribution:")
    for alert_type, count in alert_counts.items():
        if "Normal" not in alert_type:
            print(f"   - {alert_type}: {count} users")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. View alerts: Check data/guardian_alerts.csv")
    print("2. Run Streamlit dashboard: streamlit run dashboard/app.py")
    print("3. View model: models/syncfit_churn_model.pkl")
    print("\nTo see Airflow dashboard simulation:")
    print("   python airflow_dashboard_demo.py")

if __name__ == "__main__":
    main()
