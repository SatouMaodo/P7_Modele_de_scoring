#web: mlflow server --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root $MLFLOW_ARTIFACT_ROOT --host 0.0.0.0 --port $PORT
web: mlflow models serve -m "runs:/e168c89c821b4680a3a92d7ee0ed2e28/Scoring_model" --no-conda -h 0.0.0.0 -p $PORT
