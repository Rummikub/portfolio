from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.data_ingestion import load_data
from src.feature_engineering import engineer_features
from src.model_training import train_model


def run_pipeline():
    df = load_data('/opt/airflow/data/uber_data.csv')
    df = engineer_features(df)
    train_model(df.drop(columns=['trip_duration']), df['trip_duration'])


default_args = {
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

dag = DAG(
    'smartstream_pipeline',
    default_args=default_args,
    schedule_interval='@weekly',
    catchup=False
)

pipeline_task = PythonOperator(
    task_id='run_smartstream_pipeline',
    python_callable=run_pipeline,
    dag=dag
)