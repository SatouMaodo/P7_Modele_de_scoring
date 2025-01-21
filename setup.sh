#!/bin/bash
export MLFLOW_TRACKING_URI=https://d483-34-106-155-85.ngrok-free.app
# Remplacez par l'URI de votre serveur MLflow si nécessaire
mlflow models serve -m models:/Scoring_model/y -h 0.0.0.0 -p 8080 &
