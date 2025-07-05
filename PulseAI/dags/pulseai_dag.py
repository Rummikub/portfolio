from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.data_streaming import stream_user_events
from src.feature_extraction import load_streamed_events, extract_features

def run_stream_to_features():
    stream_user_events("/opt/airflow/data/stream.jsonl", num_events=50)
    df = load_streamed_events("/opt/airflow/data/stream.jsonl")
    features = extract_features(df)
    features.to_csv("/opt/airflow/data/features.csv", index=False)

default_args = {'start_date': datetime(2024, 1, 1)}

dag = DAG(
    'pulseai_pipeline',
    default_args=default_args,
    schedule_interval='@hourly',
    catchup=False
)

pipeline_task = PythonOperator(
    task_id='run_stream_to_features',
    python_callable=run_stream_to_features,
    dag=dag
)