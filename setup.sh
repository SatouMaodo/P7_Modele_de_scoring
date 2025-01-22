#!/bin/bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
# Remplacez par l'URI de votre serveur MLflow si nécessaire
mlflow models serve -m models:/Scoring_model/y -h 0.0.0.0 -p 8080 &
