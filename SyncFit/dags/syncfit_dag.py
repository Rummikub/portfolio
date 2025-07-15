from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.data_simulator import simulate_wearable_data
from src.churn_model import train_churn_model

default_args = {
    "start_date": datetime(2024, 1, 1),
    "retries": 1
}

dag = DAG(
    dag_id="syncfit_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False
)

generate_data = PythonOperator(
    task_id="simulate_data",
    python_callable=simulate_wearable_data,
    dag=dag
)

train_model = PythonOperator(
    task_id="train_churn_model",
    python_callable=train_churn_model,
    dag=dag
)

generate_data >> train_model
