import mlflow
    import pandas as pd
    from fastapi import FastAPI
    
    
    app = FastAPI()
    
    # Charger le modèle MLflow
    model_uri = "app/best_model.joblib"  # Le chemin vers le modèle enregistré
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    @app.get("/")
    async def root():
        return {"message": "Bienvenue sur l'API de Scoring"}
    
    @app.post("/predict")
    async def predict(input_data: dict):
        # Convertir les données d'entrée en DataFrame Pandas
        input_df = pd.DataFrame([input_data])
    
        # Faire une prédiction
        prediction = loaded_model.predict(input_df)
    
        return {"prediction": prediction.tolist()[0]}
