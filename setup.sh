#!/bin/bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
# Remplacez par l'URI de votre serveur MLflow si nécessaire
mlflow models serve -m models:sqlite:///mlflow.db -h 0.0.0.0 -p 8080 &
