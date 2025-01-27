import mlflow
import mlflow.pyfunc
import os

    # Définir le chemin d’accès au modèle
MODEL_PATH = os.path.join(os.getcwd(), "mlruns/1")

    # Charger le modèle
model = mlflow.pyfunc.load_model(MODEL_PATH)

 
