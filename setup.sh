export MLFLOW_TRACKING_URI=file:///mlruns/1
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000 &
