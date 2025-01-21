#!/bin/bash
export MLFLOW_TRACKING_URI=https://f88e-34-138-30-203.ngrok-free.app"
# Remplacez par l'URI de votre serveur MLflow si nécessaire
mlflow models serve -m models:/Scoring_model/y -h 0.0.0.0 -p 8080 &
