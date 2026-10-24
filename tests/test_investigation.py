import src.api as api
from fastapi.testclient import TestClient
import json
from types import SimpleNamespace
import pytest
import src.investigation as investigation_service

client = TestClient(api.app)


def test_investigation_preserves_ml_prediction(monkeypatch):
    original_prediction = {
        "prediction": 1,
        "label": "Fraudulent",
        "fraud_probability": 0.87,
        "threshold": 0.5,
        "model": "Random Forest",
        "disclaimer": "Educational prediction only.",
    }

    def fake_predict(request):
        return original_prediction

    def fake_investigation(transaction, prediction):
        # The assistant receives the original ML result.
        assert prediction is original_prediction

        return {
            "summary": "The model flagged this transaction.",
            "evidence_interpretation": (
                "The model probability exceeds its threshold."
            ),
            "limitations": [
                "The available evidence does not prove fraud."
            ],
            "recommended_actions": [
                "Review additional transaction evidence."
            ],
            "confidence_note": (
                "This is a model estimate, not proof of fraud."
            ),
        }

    monkeypatch.setattr(api, "predict_fraud", fake_predict)
    monkeypatch.setattr(
        api, "generate_investigation", fake_investigation
    )

    response = client.post(
        "/investigate",
        json={"transaction": {"Amount": 100.0}},
    )

    assert response.status_code == 200

    data = response.json()

    # The original prediction must remain unchanged.
    assert data["prediction"] == original_prediction
    assert data["prediction"]["prediction"] == 1
    assert data["prediction"]["label"] == "Fraudulent"
    assert data["prediction"]["fraud_probability"] == 0.87

    assert data["decision_authority"] == "Random Forest"
    assert "investigation" in data


def test_investigation_failure_returns_503(monkeypatch):
    def fake_predict(request):
        return {
            "prediction": 0,
            "label": "Legitimate",
            "fraud_probability": 0.12,
            "threshold": 0.5,
            "model": "Random Forest",
            "disclaimer": "Educational prediction only.",
        }

    def fake_investigation(transaction, prediction):
        raise RuntimeError("Simulated AI service failure")

    monkeypatch.setattr(api, "predict_fraud", fake_predict)
    monkeypatch.setattr(
        api, "generate_investigation", fake_investigation
    )

    response = client.post(
        "/investigate",
        json={"transaction": {"Amount": 100.0}},
    )

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]


def test_predict_works_without_investigation(monkeypatch):
    def investigation_must_not_run(*args, **kwargs):
        raise AssertionError(
            "The /predict endpoint must not call the LLM."
        )

    monkeypatch.setattr(
        api, "generate_investigation", investigation_must_not_run
    )

    transaction = {
        feature: 0.0
        for feature in api.FEATURES
    }

    response = client.post(
        "/predict",
        json={"transaction": transaction},
    )

    assert response.status_code == 200
    assert response.json()["model"] == "Random Forest"
    assert response.json()["prediction"] in (0, 1)

def test_invalid_ai_json_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="This is not valid JSON"
                )
            )
        ]
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return fake_response

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=FakeCompletions()
            )

    monkeypatch.setattr(
        investigation_service, "OpenAI", FakeOpenAI
    )

    prediction = {
        "prediction": 1,
        "label": "Fraudulent",
        "fraud_probability": 0.87,
        "threshold": 0.5,
        "model": "Random Forest",
    }

    try:
        investigation_service.generate_investigation(
            {"Amount": 100.0},
            prediction,
        )
        assert False, "Invalid JSON should have been rejected"
    except RuntimeError as exc:
        assert "invalid JSON" in str(exc)


def test_unexpected_ai_response_structure_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    invalid_result = {
        "summary": "A transaction was flagged."
    }

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(invalid_result)
                )
            )
        ]
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return fake_response

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=FakeCompletions()
            )

    monkeypatch.setattr(
        investigation_service, "OpenAI", FakeOpenAI
    )

    prediction = {
        "prediction": 1,
        "label": "Fraudulent",
        "fraud_probability": 0.87,
        "threshold": 0.5,
        "model": "Random Forest",
    }

    try:
        investigation_service.generate_investigation(
            {"Amount": 100.0},
            prediction,
        )
        assert False, "Unexpected response structure should be rejected"
    except RuntimeError as exc:
        assert "unexpected response structure" in str(exc)


from types import SimpleNamespace


def mock_openai_response(monkeypatch, content):
    """Replace OpenAI with a fake response for testing."""

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content)
            )
        ]
    )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: fake_response
            )
        )
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    monkeypatch.setattr(
        investigation_service,
        "OpenAI",
        lambda **kwargs: fake_client,
    )


@pytest.fixture
def sample_prediction():
    return {
        "prediction": 1,
        "label": "Fraudulent",
        "fraud_probability": 0.87,
        "threshold": 0.5,
        "model": "Random Forest",
    }


def test_invalid_json_is_rejected(
    monkeypatch, sample_prediction
):
    mock_openai_response(
        monkeypatch,
        "This is not valid JSON",
    )

    with pytest.raises(
        RuntimeError,
        match="invalid JSON",
    ):
        investigation_service.generate_investigation(
            transaction={"Amount": 100.0},
            prediction=sample_prediction,
        )


def test_missing_required_field_is_rejected(
    monkeypatch, sample_prediction
):
    incomplete_response = {
        "summary": "The model flagged this transaction.",
        "evidence_interpretation": "More review is needed.",
        "limitations": [],
        "recommended_actions": [],
        # confidence_note is deliberately missing
    }

    mock_openai_response(
        monkeypatch,
        json.dumps(incomplete_response),
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected response structure",
    ):
        investigation_service.generate_investigation(
            transaction={"Amount": 100.0},
            prediction=sample_prediction,
        )


def test_invalid_field_type_is_rejected(
    monkeypatch, sample_prediction
):
    invalid_response = {
        "summary": ["This should be a string"],
        "evidence_interpretation": "More review is needed.",
        "limitations": [],
        "recommended_actions": [],
        "confidence_note": "Model estimate only.",
    }

    mock_openai_response(
        monkeypatch,
        json.dumps(invalid_response),
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid AI response field: summary",
    ):
        investigation_service.generate_investigation(
            transaction={"Amount": 100.0},
            prediction=sample_prediction,
        )
        
def test_investigate_success_with_mocked_ai(monkeypatch):
    import src.api as api
    from fastapi.testclient import TestClient

    def fake_generate_investigation(transaction, prediction):
        return {
            "summary": "The model flagged this transaction for review.",
            "evidence_interpretation": (
                "The model's prediction is the basis for this review."
            ),
            "limitations": (
                "A model prediction alone does not prove fraud."
            ),
            "recommended_actions": [
                "Review the transaction using verified records."
            ],
            "confidence_note": (
                "This is an automated assessment, not a final decision."
            ),
        }

    monkeypatch.setattr(
        api,
        "generate_investigation",
        fake_generate_investigation,
    )

    test_client = TestClient(api.app)
    transaction = {
        feature: 0.0 for feature in api.FEATURES
    }

    response = test_client.post(
        "/investigate",
        json={"transaction": transaction},
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "investigation" in result
    assert result["decision_authority"] == "Random Forest"

    assert result["investigation"]["summary"] == (
        "The model flagged this transaction for review."
    )
    assert "fraud_probability" in result["prediction"]
    assert result["disclaimer"]
