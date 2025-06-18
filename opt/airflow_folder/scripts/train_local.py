import mlflow
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train_model():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-estimators', dest='n_estimators', type = int ,default=100, help="The number of estimators that would be used by CART")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum depth of the trees.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True, help="MLflow server URI.")
    
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("Local_Airflow_Training")

    with mlflow.start_run() as run:
        print(f"Starting MLflow Run: {run.info.run_id}")
        mlflow.set_tag("orchestrator", "Airflow")

        print(f"Logging parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        data = pd.read_csv(args.input_path)

        X = data.drop(['Survived'], axis=1)
        y = data['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"Test Accuracy: {accuracy}")
        mlflow.log_metric("accuracy", accuracy)

        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(model, "model", registered_model_name="titanic-classifier-local")
        
        print("Training script finished successfully.")

if __name__ == "__main__":
    train_model()

        
