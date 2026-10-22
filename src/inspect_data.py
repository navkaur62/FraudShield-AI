
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/creditcard.csv")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n=== FraudShield AI: Dataset Inspection ===")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:", df.isnull().sum().sum())

    print("\nClass distribution:")
    print(df["Class"].value_counts().sort_index())

    print(f"\nFraud percentage: {df['Class'].mean() * 100:.4f}%")


if __name__ == "__main__":
    main()
