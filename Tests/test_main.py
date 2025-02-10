from fastapi.testclient import TestClient
import sys
import os

# Obtenir le répertoire racine du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Ajouter le répertoire racine au chemin de recherche
sys.path.insert(0, project_root)

from main import app

test_client = TestClient(app)

# Test de l'endpoint /predict avec un client_id valide
def test_predict():
    response = test_client.post("/predict", json={"client_id": 100001})
    assert response.status_code == 200
    assert response.json()['prediction'] == 0.296

# Test de l'endpoint /predict sans fournir de client_id
def test_predict_missing_client_id():
    response = test_client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity car client_id est requis

# Test de l'endpoint /predict avec un client_id invalide
def test_predict_invalid_client_id():
    response = test_client.post("/predict", json={"client_id": "invalid"})
    assert response.status_code == 422  # Erreur de validation FastAPI
