import os
import joblib
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# Charger le modèle (assurez-vous que le modèle est copié dans l'image Docker lors de la construction)
model = joblib.load('best_model.joblib')

# Charger le fichier CSV de données une seule fois au démarrage
# Cela permet d'éviter de charger à chaque requête
test_df = pd.read_csv('test_df.csv')
test_df = test_df.drop(columns=['TARGET'])

@app.post("/predict")
async def predict(data: dict):
    num_client = data['client_id']
    
    # Charger uniquement les données nécessaires pour un client spécifique
    input_df = test_df[test_df['SK_ID_CURR'] == num_client]

    # Effectuer la prédiction avec le modèle
    prediction = model.predict_proba(input_df)[0, 1]

    return {"prediction": round(prediction, 3)}

@app.post("/interpretabilite_locale")
async def shap_local(data: dict):
    # Obtenir l'importance locale des caractéristiques via SHAP
    num_client = data['client_id']
    
    input_df = test_df[test_df['SK_ID_CURR'] == num_client]
    
    # Créer un explicateur SHAP pour l'arbre de décision
    explainer = shap.TreeExplainer(model)
    
    # Obtenir les valeurs SHAP pour les caractéristiques
    shap_values_explanation = explainer(input_df)
    
    # Retourner les valeurs SHAP
    return {
        "shap_values": shap_values_explanation.values.tolist(),
        "base_values": shap_values_explanation.base_values.tolist()
    }

# Si ce fichier est exécuté directement, démarrer l'application FastAPI sur le port Heroku
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Utiliser le port dynamique de Heroku
    uvicorn.run(app, host="0.0.0.0", port=port)
