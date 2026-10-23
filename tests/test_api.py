
from fastapi.testclient import TestClient

from src.api import app, FEATURES

client = TestClient(app)


def sample_transaction():
    return {
        feature: 0.0
        for feature in FEATURES
    }


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to FraudShield AI"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={"transaction": sample_transaction()},
    )

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["label"] in ["Fraudulent", "Legitimate"]
    assert 0 <= result["fraud_probability"] <= 1
    assert result["model"] == "Random Forest"


def test_missing_feature_is_rejected():
    transaction = sample_transaction()
    transaction.pop(FEATURES[0])

    response = client.post(
        "/predict",
        json={"transaction": transaction},
    )

    assert response.status_code == 422


def test_unexpected_feature_is_rejected():
    transaction = sample_transaction()
    transaction["unexpected_feature"] = 123.0

    response = client.post(
        "/predict",
        json={"transaction": transaction},
    )

    assert response.status_code == 422
