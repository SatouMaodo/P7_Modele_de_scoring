export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000 &
