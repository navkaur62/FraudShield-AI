
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

MODEL_PATH = Path("models/fraud_detection_model.joblib")
DATA_PATH = Path("data/raw/creditcard.csv")
REPORT_DIR = Path("reports/model")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]

    df = pd.read_csv(DATA_PATH)
    X = df[features]
    y = df["Class"]

    # Recreate the same held-out test split used during training.
    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    probabilities = model.predict_proba(X_test)[:, 1]
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "true_fraud_detected": int(tp),
            "missed_fraud": int(fn),
            "false_alarms": int(fp),
            "true_negatives": int(tn),
        })

    results_df = pd.DataFrame(results)

    print("\n========== THRESHOLD COMPARISON ==========")
    print(results_df.round(4).to_string(index=False))

    output_path = REPORT_DIR / "threshold_comparison.csv"
    results_df.to_csv(output_path, index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(
        results_df["threshold"],
        results_df["precision"],
        marker="o",
        label="Precision",
    )
    plt.plot(
        results_df["threshold"],
        results_df["recall"],
        marker="o",
        label="Recall",
    )
    plt.plot(
        results_df["threshold"],
        results_df["f1_score"],
        marker="o",
        label="F1-score",
    )
    plt.xlabel("Fraud Probability Threshold")
    plt.ylabel("Score")
    plt.title("Fraud Detection: Threshold Comparison")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    chart_path = REPORT_DIR / "threshold_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nSaved comparison: {output_path.resolve()}")
    print(f"Saved chart: {chart_path.resolve()}")


if __name__ == "__main__":
    main()
