import signal
import sys
import os
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import shap



app = FastAPI()

# Charger le modèle au démarrage de l'application
model = joblib.load('best_model.joblib')

# Charger les données de test au démarrage pour éviter les chargements répétés
test_df = pd.read_csv('test_df.csv', sep=';')


# Gérer les valeurs manquantes dans le DataFrame
test_df = test_df.replace([np.inf, -np.inf], np.nan)
for col in test_df.select_dtypes(include=np.number).columns:
    test_df[col] = test_df[col].fillna(test_df[col].median())  # Correction ici


@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API FastAPI ! Utilisez /docs pour voir la documentation."}

@app.post("/predict")
async def predict(data: dict):
    # Vérifier si client_id est fourni et est un entier
    client_id = data.get("client_id")
    if client_id is None:
        raise HTTPException(status_code=422, detail="client_id est requis")
    if not isinstance(client_id, int):
        raise HTTPException(status_code=404, detail="client_id doit être un entier")

    # Vérifier si le client existe dans le dataset
    input_df = test_df[test_df['SK_ID_CURR'] == client_id]
    if input_df.empty:
        raise HTTPException(status_code=404, detail="Client non trouvé dans les données")

    # Prédiction avec le modèle
    prediction = model.predict_proba(input_df)[0, 1]

    return {"prediction": round(prediction, 3)}

@app.post("/interpretabilite_locale")
async def shap_local(data: dict):
    num_client = data['client_id']
    
    # Vérifier si le client existe dans le dataset
    input_df = test_df[test_df['SK_ID_CURR'] == num_client]
    
    if input_df.empty:
        raise HTTPException(status_code=404, detail="Client non trouvé dans les données")
    
    # Explainer SHAP
    feature_names = test_df.columns
    explainer = shap.TreeExplainer(model)
    shap_values_explanation = explainer.shap_values(input_df)
    
    # Retourner les valeurs SHAP sous forme de liste
    return {
        "shap_values": shap_values_explanation[0].tolist(),
        "base_values": explainer.expected_value.tolist(),  # Correction ici
        "feature_names": feature_names.tolist()
    }
