from fastapi.testclient import TestClient
import sys
import os

# Obtenir le répertoire du fichier courant
current_dir = os.path.dirname(os.path.abspath(__file__))

# Ajouter le répertoire parent au chemin de recherche des modules
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from main import app
test_client=TestClient(app)


def test_predict():
    reponse=test_client.post("/predict", json={"client_id":100001})
    assert reponse.status_code==200
    assert reponse.json()['prediction']==0.060
