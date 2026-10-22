
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/raw/creditcard.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/model")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
    }

    print(f"\n========== {name} ==========")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    ))
    print("Average precision:", round(
        metrics["average_precision"], 4
    ))

    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Legitimate", "Fraud"],
    )
    display.plot(values_format="d")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / f"{name.lower().replace(' ', '_')}_confusion.png",
        dpi=150,
    )
    plt.close()

    return metrics


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' was not found.")

    X = df.drop(columns="Class")
    y = df["Class"]

    # Keep a final test set separate from model training.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Fraud cases in training:", int(y_train.sum()))
    print("Fraud cases in test:", int(y_test.sum()))

    results = []

    # Baseline: always predicts the majority class.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    results.append(
        evaluate_model("Dummy Baseline", baseline, X_test, y_test)
    )

    # First real ML model.
    # Balanced class weights give greater importance to fraud examples.
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining Random Forest; this may take a few minutes...")
    model.fit(X_train, y_train)

    results.append(
        evaluate_model("Random Forest", model, X_test, y_test)
    )

    # Save the model and the exact feature order used during training.
    model_path = MODEL_DIR / "fraud_detection_model.joblib"
    joblib.dump(
        {
            "model": model,
            "features": X.columns.tolist(),
            "threshold": 0.5,
        },
        model_path,
    )

    metrics_path = REPORT_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n========== MODEL COMPARISON ==========")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSaved model to: {model_path.resolve()}")
    print(f"Saved metrics to: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
