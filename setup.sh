# Définissez l'URI de suivi MLflow (si vous utilisez un serveur MLflow)
export MLFLOW_TRACKING_URI="http://127.0.0.1:5313" 

# Définissez le chemin du modèle (utilisez le chemin relatif ou absolu)
MODEL_PATH="mlruns/1/e168c89c821b4680a3a92d7ee0ed2e28/artifacts/Scoring_model"

# Démarrez le serveur de modèle MLflow 
mlflow models serve -m "$MODEL_PATH" -h 0.0.0.0 -p 5313 &
