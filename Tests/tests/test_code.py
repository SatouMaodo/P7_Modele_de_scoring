from fastapi.testclient import TestClient
from my_function import app
test_client=TestClient(app)


def test_predict():
    reponse=test_client.post("/predict", json={"client_id":100001})
    assert reponse.status_code==200
    assert reponse.json()['prediction']==0.060
