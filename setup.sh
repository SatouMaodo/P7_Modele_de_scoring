#!/bin/bash
export MLFLOW_TRACKING_URI="./mlruns/1/e168c89c821b4680a3a92d7ee0ed2e28/artifacts"
#export MLFLOW_MODEL_URI="models:/Scoring_model"
#export MLFLOW_TRACKING_URI="https://5451-34-16-135-108.ngrok-free.app"
# Remplacez par l'URI de votre serveur MLflow si nécessaire
#mlflow models serve -m models:file:///mlruns -h 0.0.0.0 -p 8080 &
# Démarrer le serveur de modèle MLflow
#mlflow models serve -m "runs:/e168c89c821b4680a3a92d7ee0ed2e28/Scoring_model" -h 0.0.0.0 -p 8080 &



# Définissez les variables d'environnement pour votre application
#export MLFLOW_MODEL_PATH='models:/Projet7_Scoring_model/Production'

