from airflow.models.dag import DAG
from airflow.providers.standard.operators.bash import BashOperator
# from airflow.providers.standard.operators.python import BashOperator
from datetime import datetime
import os

# --- Configuration Constants ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
AIRFLOW_SCRIPTS_DIR = "opt/airflow_folder/scripts"
DATA_DIR = "/opt/airflow_folder/data/titanic"
TRAINING_SCRIPT_PATH = os.path.join(AIRFLOW_SCRIPTS_DIR, "train_local.py")
TRAINING_SCRIPT_PATH = os.path.join(AIRFLOW_SCRIPTS_DIR, "train_local.py")
PREPROCESS_SCRIPT_PATH = os.path.join(AIRFLOW_SCRIPTS_DIR, "preprocess_local.py")
RAW_DATA_PATH = os.path.join(DATA_DIR, "titanic.csv") 
FINAL_DATA_PATH = os.path.join(DATA_DIR, "titanic_cleaned.csv") 


# --- Default Arguments for the DAG ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    dag_id='local_ml_pipeline',
    default_args=default_args,
    description='A simple ML pipeline running on the Airflow worker',
    schedule=None, # This DAG is manually triggered
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['ml', 'local'],
) as dag:
    
    preprocess_task = BashOperator(
        task_id='preprocessing_data_task',
        bash_command=(
            f"python {PREPROCESS_SCRIPT_PATH} "
            f"--input-path {RAW_DATA_PATH} "
            f"--output-path {FINAL_DATA_PATH} "
        )
    )
    
    train_model_task = BashOperator(
        task_id='train_random_forest_model',
        bash_command=(
            f"python {TRAINING_SCRIPT_PATH} "
            f"--n-estimators 150 "
            f"--max-depth 8 "
            f"--input-path {FINAL_DATA_PATH} "
            f"--mlflow-tracking-uri {MLFLOW_TRACKING_URI}"
        )
    )

    preprocess_task >> train_model_task