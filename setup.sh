export MLFLOW_TRACKING_URI=file:///mlflow.db
mlflow server --backend-store-uri file:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000 &
