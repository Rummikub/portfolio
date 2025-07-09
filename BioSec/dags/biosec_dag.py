from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.data_generator import generate_data
from src.anomaly_model import train_pycaret_model

default_args = {
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}


# Genearte Daily DAG
dag = DAG(
    dag_id = "biosec_pipeline",
    default_args = default_args,
    schedule_interval = "@daily",
    catchup = False
)

generate_logs = PythonOperator(
    task_id = "generate_logs",
    python_callable = generate_data,
    dag = dag
)

train_model = PythonOperator(
    task_id = "train_model",
    python_callable = train_pycaret_model,
    dag = dag
)

generate_logs >> train_model