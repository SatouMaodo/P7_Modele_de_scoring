#!/bin/bash
export MLFLOW_TRACKING_URI="mlruns/1/e168c89c821b4680a3a92d7ee0ed2e28/artifacts/Scoring_model"
# Remplacez par l'URI de votre serveur MLflow si nécessaire
mlflow models serve -m models:mlruns/1/e168c89c821b4680a3a92d7ee0ed2e28/artifacts/Scoring_model -h 0.0.0.0 -p 8080 &
