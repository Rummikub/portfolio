from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

from src.data_ingestion import load_data
from src.features_engineering import engineer_features
from src.model_training import train_model

import os

DATA_PATH = "data/yellow_tripdata_2025-01.parquet"

default_args = {
    'owner': 'bianca',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_pipeline():
    print("Loading TLC trip data...")
    df = load_data(DATA_PATH)
    
    print("Engineering features...")
    X, y = engineer_features(df)

    print("Training model...")
    model = train_model(X, y)

    print("Model training complete and saved.")

with DAG(
    'mobilityai_pipeline',
    default_args=default_args,
    description='MobilityAI model training pipeline',
    schedule_interval=None,  # manual run
    catchup=False,
) as dag:
    run_pipeline_task = PythonOperator(
        task_id='run_training_pipeline',
        python_callable=run_pipeline
    )