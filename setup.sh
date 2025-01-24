#!/bin/bash
#export MLFLOW_TRACKING_URI="file:///mlruns"

#export MLFLOW_TRACKING_URI="https://5451-34-16-135-108.ngrok-free.app"
# Remplacez par l'URI de votre serveur MLflow si nécessaire
#mlflow models serve -m models:file:///mlruns -h 0.0.0.0 -p 8080 &
# Démarrer le serveur de modèle MLflow
#mlflow models serve -m "runs:/e168c89c821b4680a3a92d7ee0ed2e28/Scoring_model" -h 0.0.0.0 -p 8080 &

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install mlflow

# Téléchargez le modèle MLflow
mlflow models download -m "$MLFLOW_TRACKING_URI" -o model

# Définissez les variables d'environnement pour votre application
export MLFLOW_MODEL_PATH=best_model.joblib # Chemin vers le modèle dans l'image Docker

