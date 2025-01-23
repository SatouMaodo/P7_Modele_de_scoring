#!/bin/bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5313"
# Remplacez par l'URI de votre serveur MLflow si nécessaire
mlflow models serve -m models:https://98fd-34-168-184-236.ngrok-free.app -h 0.0.0.0 -p 8080 &
