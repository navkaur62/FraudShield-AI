
from pathlib import Path
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAIError
from src.investigation import generate_investigation

# Locate the saved model relative to this source file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. "
        "Run src/train_model.py first."
    )

# Load the model once when the API starts.
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
FEATURES = model_bundle["features"]
THRESHOLD = float(model_bundle.get("threshold", 0.5))


app = FastAPI(
    title="FraudShield AI API",
    description=(
        "An API for predicting potentially fraudulent "
        "credit card transactions using a trained ML model."
    ),
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    transaction: dict[str, float] = Field(
        ...,
        description=(
            "Transaction features. Include every model feature, "
            "such as Time, V1-V28, and Amount."
        ),
    )


@app.get("/")
def home():
    return {
        "message": "Welcome to FraudShield AI",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_count": len(FEATURES),
    }


@app.post("/predict")
def predict_fraud(request: PredictionRequest):
    transaction = request.transaction

    missing_features = sorted(set(FEATURES) - set(transaction))
    unexpected_features = sorted(set(transaction) - set(FEATURES))

    if missing_features or unexpected_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Transaction features do not match the model.",
                "missing_features": missing_features,
                "unexpected_features": unexpected_features,
            },
        )

    if not all(math.isfinite(value) for value in transaction.values()):
        raise HTTPException(
            status_code=422,
            detail="All transaction feature values must be finite numbers.",
        )

    # Arrange features in exactly the order used during training.
    input_df = pd.DataFrame(
        [[transaction[name] for name in FEATURES]],
        columns=FEATURES,
    )

    probabilities = model.predict_proba(input_df)[0]
    fraud_probability = float(probabilities[list(model.classes_).index(1)])

    predicted_class = int(fraud_probability >= THRESHOLD)

    return {
        "prediction": predicted_class,
        "label": (
            "Fraudulent" if predicted_class == 1 else "Legitimate"
        ),
        "fraud_probability": round(fraud_probability, 6),
        "threshold": THRESHOLD,
        "model": "Random Forest",
        "disclaimer": (
            "Educational prediction only; not a final financial decision."
        ),
    }
@app.post("/investigate")
def investigate_transaction(request: PredictionRequest):
    # Reuse the existing validation and ML prediction logic.
    prediction = predict_fraud(request)

    try:
        investigation = generate_investigation(
            transaction=request.transaction,
            prediction=prediction,
        )

    except (OpenAIError, RuntimeError) as error:
        print(
            f"Investigation error: "
            f"{type(error).__name__}: {error}"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "The investigation assistant is temporarily "
                "unavailable. The ML prediction endpoint "
                "remains available."
            ),
        )

    return {
        "prediction": prediction,
        "investigation": investigation,
        "decision_authority": "Random Forest",
        "disclaimer": (
            "The investigation is AI-generated and is not proof "
            "of fraud. The ML prediction remains authoritative."
        ),
    }