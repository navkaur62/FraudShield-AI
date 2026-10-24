import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


SYSTEM_PROMPT = """
You are FraudShieldAI's transaction investigation assistant.

Your job is to explain an existing machine-learning prediction,
not to make a new fraud decision.

STRICT RULES:
1. The supplied ML prediction is authoritative. Never change it.
2. Do not recalculate the prediction or fraud probability.
3. Treat V1 through V28 as anonymized features with unknown meanings.
   Never assign real-world meanings to these features.
4. Do not claim that a transaction is fraudulent merely because
   the model predicts fraud.
5. Use only the evidence provided in the input.
6. Clearly state when the available evidence is insufficient.
7. Do not invent merchant details, locations, devices, cardholder
   behavior, transaction history, or reasons for a prediction.
8. Recommendations must be reasonable human-review steps.
9. Return valid JSON only, with exactly these keys:
   summary, evidence_interpretation, limitations,
   recommended_actions, confidence_note.

The summary and evidence_interpretation must be strings.
The limitations and recommended_actions must be arrays of strings.
The confidence_note must be a string.

Explain model probability as a model estimate, not as proof of fraud.
Do not describe the probability as a calibrated real-world likelihood
unless calibration has been independently established.
"""


def build_investigation_evidence(transaction, prediction):
    """Prepare verified, minimal evidence for the LLM."""

    evidence = {
        "ml_prediction": prediction["prediction"],
        "ml_label": prediction["label"],
        "fraud_probability": prediction["fraud_probability"],
        "decision_threshold": prediction["threshold"],
        "model": prediction["model"],
        "available_transaction_fields": [],
        "anonymized_feature_count": sum(
            name.startswith("V") and name[1:].isdigit()
            for name in transaction
        ),
    }

    # Include only fields whose names have known meanings.
    # Do not send the anonymized V1-V28 values to the LLM.
    for field in ("Time", "Amount"):
        if field in transaction:
            evidence["available_transaction_fields"].append(
                {
                    "name": field,
                    "value": transaction[field],
                }
            )

    return evidence


def generate_investigation(transaction, prediction):
    """
    Generate an explanation of an existing ML prediction.

    This function never runs the ML model or changes its decision.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Check your .env file."
        )

    if not model_name:
        raise RuntimeError(
            "OPENAI_MODEL is missing. Check your .env file."
        )

    evidence = build_investigation_evidence(
        transaction,
        prediction,
    )

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(evidence),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_completion_tokens=600,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "The AI assistant returned an empty response."
        )

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The AI assistant returned invalid JSON."
        ) from exc

    expected_keys = {
        "summary",
        "evidence_interpretation",
        "limitations",
        "recommended_actions",
        "confidence_note",
    }

    if not isinstance(result, dict) or set(result) != expected_keys:
        raise RuntimeError(
            "The AI assistant returned an unexpected response structure."
        )

    for field in (
        "summary",
        "evidence_interpretation",
        "confidence_note",
    ):
        if not isinstance(result[field], str):
            raise RuntimeError(
                f"Invalid AI response field: {field}"
            )

    for field in ("limitations", "recommended_actions"):
        if (
            not isinstance(result[field], list)
            or not all(
                isinstance(item, str) for item in result[field]
            )
        ):
            raise RuntimeError(
                f"Invalid AI response field: {field}"
            )

    return result