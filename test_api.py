from fastapi.testclient import TestClient
from main import app,model
client=TestClient(app)

def test_home():
    response=client.get("/")
    assert response.status_code==200
    assert "welcome" in response.json()["message"]

def test_health():
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()["status"]=="healthy"
    assert response.json()["model_loaded"] is True

def test_predict():
    input_data={"tenure":12,"monthly_charges":85.0,"total_charges":1027.0}
    response=client.post("/predict",json=input_data)
    assert response.status_code==200
    result=response.json()
    assert "churn_pred" in result
    assert "churn_prob" in result
    assert "risk_level" in result
    assert result["churn_pred"] in [0,1]
    assert 0<=result["churn_prob"]<=100
    assert result["risk_level"] in ["high risk","low risk"]

def test_invalid_input():
    input_data={"tenure":-5,"monthly_charges":85.0,"total_charges":1027.0}
    response=client.post("/predict",json=input_data)
    assert response.status_code==422
     