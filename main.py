from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import shap
app = FastAPI()

# Charger le modèle
model = joblib.load('best_model.joblib')

test_df= pd.read_csv('test_df.csv')
test_df = test_df.drop(columns=['TARGET'])
@app.post("/predict")
async def predict(data: dict):
    num_client = data['client_id']
    test_df = pd.read_csv('test_df.csv')  # Charger à chaque fois la donnée nécessaire
    test_df = test_df.drop(columns=['TARGET'])
    input_df = test_df[test_df['SK_ID_CURR'] == num_client]

    prediction = model.predict_proba(input_df)[0, 1]

    return {"prediction": round(prediction, 3)}

@app.post("/interpretabilite_locale")
async def shap_local(data:dict):
    # Feature importance locale avec shap.plots.waterfall
    num_client = data['client_id']
    input_df = test_df[test_df['SK_ID_CURR'] == num_client]
    feature_names = test_df.columns
    explainer = shap.TreeExplainer(model)
    # Get SHAP values as an Explanation object
    shap_values_explanation = explainer(input_df)
    print(shap_values_explanation[0])
    return {"shap_values": shap_values_explanation.values.tolist(),"base_values":shap_values_explanation.base_values.tolist()}
