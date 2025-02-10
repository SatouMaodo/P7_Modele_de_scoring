from fastapi.testclient import TestClient
import sys
import os

import sys
import os

# Obtenir le répertoire racine du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Ajouter le répertoire racine au chemin de recherche
sys.path.insert(0, project_root)

from main import app
test_client=TestClient(app)


from fastapi.testclient import TestClient
from main import app
test_client=TestClient(app)


def test_predict():
    reponse=test_client.post("/predict", json={"client_id":100001})
    assert reponse.status_code==200
    assert reponse.json()['prediction']==0.060
