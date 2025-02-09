from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

# Charger le modèle
model = joblib.load('best_model.joblib')

# Créer une instance FastAPI
app = FastAPI()


# Définir le point de terminaison pour la prédiction
@app.post("/predict")
async def predict(data: dict):
    # Créer un DataFrame à partir des données d'entrée
    input_df = pd.DataFrame([data])

    # Sélectionner les colonnes spécifiées
    selected_features = ['DAYS_EMPLOYED', 'OWN_CAR_AGE', 'NAME_EDUCATION_TYPE_Higher education',
                         'PAYMENT_RATE', 'AMT_GOODS_PRICE', 'EXT_SOURCE_1', 'CODE_GENDER',
                         'EXT_SOURCE_2', 'EXT_SOURCE_3']
    input_data = input_df[selected_features]

    # Remplacer les valeurs infinies par des NaN
    input_data = input_data.replace([np.inf, -np.inf], np.nan)

    # Imputation des valeurs manquantes par la médiane
    for col in input_data.select_dtypes(include=np.number).columns:
        input_data[col] = input_data[col].fillna(input_data[col].median())

    # Effectuer la prédiction
    prediction = model.predict(input_data)[0]

    # Retourner la prédiction
    return {"prediction": prediction}
