#!/bin/bash
#export MLFLOW_TRACKING_URI="file:///mlruns"
# Remplacez par l'URI de votre serveur MLflow si nécessaire
#mlflow models serve -m models:file:///mlruns -h 0.0.0.0 -p 8080 &
# Démarrer le serveur de modèle MLflow
mlflow models serve -m "runs:/e168c89c821b4680a3a92d7ee0ed2e28/Scoring_model" -h 0.0.0.0 -p 8080 &
