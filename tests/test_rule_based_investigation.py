
from src.investigation import generate_rule_based_investigation

import src.api as api
from fastapi.testclient import TestClient

def test_fraud_prediction_produces_review_recommendations():
    prediction = {
        "prediction": 1,
        "label": "Fraudulent",
        "fraud_probability": 0.87,
        "threshold": 0.5,
        "model": "Random Forest",
    }

    result = generate_rule_based_investigation(
        transaction={"Amount": 100.0, "Time": 20.0},
        prediction=prediction,
    )

    assert "summary" in result
    assert "evidence_interpretation" in result
    assert result["recommended_actions"]
    assert result["limitations"]
    assert result["confidence_note"]

    # The investigation must not change the model prediction.
    assert prediction["prediction"] == 1
    assert prediction["fraud_probability"] == 0.87


def test_legitimate_prediction_is_not_called_proven_safe():
    prediction = {
        "prediction": 0,
        "label": "Legitimate",
        "fraud_probability": 0.12,
        "threshold": 0.5,
        "model": "Random Forest",
    }

    result = generate_rule_based_investigation(
        transaction={"Amount": 50.0},
        prediction=prediction,
    )

    assert "summary" in result
    assert "does not guarantee" in result["evidence_interpretation"]
    assert result["recommended_actions"]


def test_anonymized_features_are_not_given_real_world_meanings():
    prediction = {
        "prediction": 1,
        "label": "Fraudulent",
        "fraud_probability": 0.87,
        "threshold": 0.5,
        "model": "Random Forest",
    }

    result = generate_rule_based_investigation(
        transaction={"V1": -2.3, "V2": 1.5},
        prediction=prediction,
    )

    explanation = str(result).lower()

    assert "anonymized v1-v28 features" in explanation
    assert "merchant details" in explanation
def test_investigate_endpoint_works_without_openai_credits():
    client = TestClient(api.app)

    # Create a valid transaction containing every required feature.
    transaction = {
        feature: 0.0
        for feature in api.FEATURES
    }

    response = client.post(
        "/investigate",
        json={"transaction": transaction},
    )

    assert response.status_code == 200

    data = response.json()

    # The ML model remains responsible for the prediction.
    assert data["decision_authority"] == "Random Forest"
    assert data["prediction"]["model"] == "Random Forest"

    # The free investigation explanation is returned.
    assert data["investigation"]["summary"]
    assert data["investigation"]["limitations"]
    assert data["investigation"]["recommended_actions"]

    assert "rule-based explanation" in data["disclaimer"]