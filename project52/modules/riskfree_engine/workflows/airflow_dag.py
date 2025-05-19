# File: workflows/airflow_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from api.main import get_curve

with DAG('riskfree_engine', start_date=datetime(2025, 1, 1), schedule_interval='@daily') as dag:
    task_build = PythonOperator(
        task_id='build_curve',
        python_callable=lambda: get_curve(date=datetime.now().strftime('%Y-%m-%d'))
    )