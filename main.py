from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import shap
import os

app = FastAPI()

# Charger le modèle au démarrage de l'application
model = joblib.load('best_model.joblib')

# Charger les données de test au démarrage pour éviter les chargements répétés
test_df = pd.read_csv('test_df.csv')

# Supprimer la colonne 'TARGET' (elle ne doit pas être utilisée dans les prédictions)
test_df = test_df.drop(columns=['TARGET'])

# Gérer les valeurs manquantes dans le DataFrame
test_df = test_df.replace([np.inf, -np.inf], np.nan)
for col in test_df.select_dtypes(include=np.number).columns:
    test_df[col] = test_df[col].fillna(test_df[col].median())

@app.post("/predict")
async def predict(data: dict):
    num_client = data['client_id']
    
    # Vérifier si le client existe dans le dataset
    input_df = test_df[test_df['SK_ID_CURR'] == num_client]
    
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
        "base_values": shap_values_explanation.base_values[0].tolist()
    }

# Important: Configurez l'application pour écouter sur le bon port Heroku
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
