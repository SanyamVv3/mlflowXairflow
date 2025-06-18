# This file contains two parts:
# 1. The training script that will be executed by Airflow ('train_local.py').
# 2. The Airflow DAG definition that orchestrates the workflow ('ml_dag_local.py').

# ==============================================================================
# Part 1: The Training Script (train_local.py)
# ==============================================================================
# FILE LOCATION: Place this script in a directory accessible by your Airflow instance,
# for example: /opt/airflow/scripts/train_local.py

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train_model():
    """
    A function to train a model, logging everything to MLflow.
    This function is designed to be called by an Airflow task.
    """
    # --- 1. Argument Parsing (getting info from Airflow) ---
    # We use argparse to make the script configurable. Airflow will pass these.
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum depth of the trees.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True, help="MLflow server URI.")
    
    args = parser.parse_args()

    # --- 2. MLflow Setup ---
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("Local_Airflow_Training")
    
    with mlflow.start_run() as run:
        print(f"Starting MLflow Run: {run.info.run_id}")
        mlflow.set_tag("orchestrator", "Airflow")

        # --- 3. Log Parameters ---
        print(f"Logging parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # --- 4. Data Loading and Preparation ---
        print(f"Loading data from {args.input_path}")
        iris_df = pd.read_csv(args.input_path)
        X = iris_df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
        y = iris_df['variety']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- 5. Model Training ---
        print("Training RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # --- 6. Evaluation and Logging Metrics ---
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"Test Accuracy: {accuracy}")
        mlflow.log_metric("accuracy", accuracy)

        # --- 7. Log Model ---
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(model, "model", registered_model_name="iris-classifier-local")
        
        print("Training script finished successfully.")

if __name__ == "__main__":
    train_model()

# ==============================================================================
# Part 2: The Airflow DAG (ml_dag_local.py)
# ==============================================================================
# FILE LOCATION: Place this script in your Airflow DAGs folder: /opt/airflow/dags/ml_dag_local.py

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

# --- Configuration Constants ---
MLFLOW_TRACKING_URI = "http://<your-mlflow-server-ip>:5000"
AIRFLOW_SCRIPTS_DIR = "/opt/airflow/scripts"
DATA_DIR = "/opt/airflow/data"
TRAINING_SCRIPT_PATH = os.path.join(AIRFLOW_SCRIPTS_DIR, "train_local.py")
RAW_DATA_PATH = os.path.join(DATA_DIR, "iris.csv") # Assume iris.csv is placed here

# --- Default Arguments for the DAG ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

def prepare_environment():
    """
    Creates directories needed for the workflow.
    This demonstrates a typical data preparation/setup step.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(AIRFLOW_SCRIPTS_DIR, exist_ok=True)
    print(f"Data directory '{DATA_DIR}' is ready.")
    print("In a real scenario, this task would download data from a source like a database or S3.")
    # For this example, manually place iris.csv in the DATA_DIR.
    # You could add a download command here:
    # os.system(f"curl -o {RAW_DATA_PATH} https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")


with DAG(
    dag_id='local_ml_pipeline',
    default_args=default_args,
    description='A simple ML pipeline running on the Airflow worker',
    schedule_interval=None, # This DAG is manually triggered
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['ml', 'local'],
) as dag:

    # --- Task 1: Prepare the environment and data ---
    prepare_env_task = PythonOperator(
        task_id='prepare_environment_and_data',
        python_callable=prepare_environment
    )

    # --- Task 2: Run the model training script ---
    # We use a BashOperator to show how Airflow executes the script as a command-line process.
    # This is a robust way to run isolated Python scripts.
    train_model_task = BashOperator(
        task_id='train_random_forest_model',
        bash_command=(
            f"python {TRAINING_SCRIPT_PATH} "
            f"--n-estimators 150 "
            f"--max-depth 8 "
            f"--input-path {RAW_DATA_PATH} "
            f"--mlflow-tracking-uri {MLFLOW_TRACKING_URI}"
        )
    )

    # --- Define Task Dependencies ---
    prepare_env_task >> train_model_task