# dags/syncfit_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.feature_builder import build_user_features
from src.churn_model import train_churn_model
from src.guardian_alerts import run_guardian_alerts

default_args = {
    'start_date': datetime(2025, 1, 1),
    'catchup': False,
}

with DAG("syncfit_pipeline",
         default_args=default_args,
         schedule_interval="@daily",
         description="SyncFit DAG without synthetic generation",
         tags=["syncfit", "wearables", "churn", "ml"]) as dag:

    def train_wrapper():
        X, y = build_user_features("data/synthetic_wearable_logs.csv")
        train_churn_model(X, y)

    t1 = PythonOperator(
        task_id="train_churn_model",
        python_callable=train_wrapper
    )

    t2 = PythonOperator(
        task_id="guardian_alerts",
        python_callable=run_guardian_alerts,
        op_args=["data/synthetic_wearable_logs.csv"]
    )

    t1 >> t2
