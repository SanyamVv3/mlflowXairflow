# This file contains two parts:
# 1. The training script adapted to run on SageMaker ('train_sagemaker.py').
# 2. The Airflow DAG that triggers the SageMaker job ('ml_dag_sagemaker.py').

# ==============================================================================
# Part 1: The SageMaker Training Script (train_sagemaker.py)
# ==============================================================================
# FILE LOCATION: This script should be in a local directory from where Airflow will
# upload it to S3, for example: /opt/airflow/sagemaker_scripts/train_sagemaker.py

import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

def train_model_sagemaker():
    """
    This script is executed within the SageMaker training container.
    It reads environment variables and arguments set by SageMaker.
    """
    # --- 1. Argument Parsing ---
    # SageMaker passes hyperparameters as command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    
    # SageMaker-specific environment variables for data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    # We pass the MLflow URI via a custom environment variable
    parser.add_argument('--mlflow-tracking-uri', type=str, required=True)
    
    args = parser.parse_args()

    # --- 2. MLflow Setup ---
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("SageMaker_Airflow_Training")

    with mlflow.start_run() as run:
        mlflow.set_tag("orchestrator", "Airflow")
        mlflow.set_tag("compute_engine", "SageMaker")

        # --- 3. Log Parameters ---
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max-depth", args.max_depth)
        mlflow.log_param("sagemaker_job_name", os.environ.get("SAGEMAKER_JOB_NAME", "N/A"))

        # --- 4. Data Loading from SageMaker's Channel ---
        # SageMaker automatically copies data from S3 to the path specified by SM_CHANNEL_TRAIN
        input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
        if not input_files:
            raise ValueError(f"No training data files found in {args.train}. Check S3 input path.")
            
        raw_data = [pd.read_csv(file) for file in input_files]
        train_df = pd.concat(raw_data)
        
        X_train = train_df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
        y_train = train_df['variety']

        # --- 5. Model Training ---
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        model.fit(X_train, y_train)

        # --- 6. Log Metrics ---
        # For simplicity, we evaluate on the training set. In reality, you'd use a validation channel.
        accuracy = model.score(X_train, y_train)
        mlflow.log_metric("training_accuracy", accuracy)

        # --- 7. Log and Save the Model ---
        # Log to MLflow artifact store
        mlflow.sklearn.log_model(model, "model", registered_model_name="iris-classifier-sagemaker")
        
        # Save model in the format SageMaker expects for deployment
        # This model is saved to SM_MODEL_DIR, which SageMaker then zips and saves to S3.
        joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

if __name__ == "__main__":
    train_model_sagemaker()

# ==============================================================================
# Part 2: The Airflow DAG to trigger SageMaker (ml_dag_sagemaker.py)
# ==============================================================================
# FILE LOCATION: Place this script in your Airflow DAGs folder: /opt/airflow/dags/ml_dag_sagemaker.py

from airflow.models.dag import DAG
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python import PythonOperator
from sagemaker.sklearn.estimator import SKLearn
from datetime import datetime

# --- Configuration Constants ---
MLFLOW_TRACKING_URI = "http://<your-mlflow-server-ip>:5000"
S3_BUCKET = "<your-s3-bucket-name>" # Bucket for data, scripts, and model output
SAGEMAKER_ROLE_ARN = "arn:aws:iam::<your-aws-account-id>:role/<your-sagemaker-execution-role-name>"
LOCAL_SCRIPT_PATH = "/opt/airflow/sagemaker_scripts/train_sagemaker.py"

# S3 paths
S3_SCRIPT_KEY = "scripts/train_sagemaker.py"
S3_INPUT_DATA_KEY = "data/iris/iris.csv"
S3_TRAIN_DATA_URI = f"s3://{S3_BUCKET}/data/iris/"
S3_MODEL_OUTPUT_URI = f"s3://{S3_BUCKET}/models/"

# --- Default Arguments for the DAG ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
}

def upload_to_s3(file_path: str, bucket_name: str, key: str):
    """Uploads a file to an S3 bucket."""
    s3_hook = S3Hook()
    s3_hook.load_file(filename=file_path, key=key, bucket_name=bucket_name, replace=True)
    print(f"Successfully uploaded {file_path} to s3://{bucket_name}/{key}")

# --- Estimator Configuration for the SageMaker Operator ---
# This dictionary defines the entire SageMaker training job.
sagemaker_training_config = {
    "AlgorithmSpecification": {
        "TrainingImage": SKLearn.get_image_uri(version="1.2-1", instance_type="ml.m5.large"),
        "TrainingInputMode": "File",
    },
    "HyperParameters": {
        "n-estimators": "200", # Must be strings
        "max-depth": "12",
        "mlflow-tracking-uri": f'"{MLFLOW_TRACKING_URI}"', # Pass MLflow URI as hyperparameter
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_TRAIN_DATA_URI,
                }
            },
        }
    ],
    "OutputDataConfig": {"S3OutputPath": S3_MODEL_OUTPUT_URI},
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.m5.large",
        "VolumeSizeInGB": 10,
    },
    "RoleArn": SAGEMAKER_ROLE_ARN,
    "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
    # The 'entry_point' is specified within the source_dir tarball
    # We define it via the SageMakerTrainingOperator's parameters instead.
}

with DAG(
    dag_id='sagemaker_ml_pipeline',
    default_args=default_args,
    description='A pipeline to trigger SageMaker training jobs',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'sagemaker'],
) as dag:
    
    # Task to upload the training script to S3
    upload_script_task = PythonOperator(
        task_id='upload_sagemaker_script_to_s3',
        python_callable=upload_to_s3,
        op_kwargs={
            'file_path': LOCAL_SCRIPT_PATH,
            'bucket_name': S3_BUCKET,
            'key': S3_SCRIPT_KEY,
        },
    )

    # The main task: trigger the SageMaker training job
    trigger_sagemaker_job = SageMakerTrainingOperator(
        task_id='trigger_sagemaker_training_job',
        config=sagemaker_training_config,
        aws_conn_id='aws_default', # The Airflow connection to AWS
        # The operator needs to know where to find the entrypoint script in S3
        sagemaker_model_config={
             "source_dir": f"s3://{S3_BUCKET}/{S3_SCRIPT_KEY.rsplit('/', 1)[0]}/",
             "entry_point": S3_SCRIPT_KEY.rsplit('/', 1)[1],
        },
        wait_for_completion=True,
    )

    # --- Define Task Dependencies ---
    upload_script_task >> trigger_sagemaker_job

