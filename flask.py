from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle
model = joblib.load('best_model.joblib')  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])  # Convertir les données JSON en DataFrame
    prediction = model.predict_proba(df)[0][1]  # Obtenir la probabilité de la classe 1
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)  # Exécuter l'API sur le port 8080
