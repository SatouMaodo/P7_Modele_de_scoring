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

# Enregistrer le modèle
model_info = mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="mlruns/1/e168c89c821b4680a3a92d7ee0ed2e28/artifacts",
    registered_model_name="Projet7_Scoring_model"
)
# Transitionner le modèle en production
client = mlflow.tracking.MlflowClient()

# Get the latest version of the registered model
latest_versions = client.get_latest_versions(name="Projet7_Scoring_model")

# the latest version to production
latest_version = latest_versions[0].version

client.transition_model_version_stage(
    name="Projet7_Scoring_model",
    version=latest_version,  
    stage="Production",
    archive_existing_versions=False
)


# Définissez les variables d'environnement pour votre application
export MLFLOW_MODEL_PATH='models:/Projet7_Scoring_model/Production'

