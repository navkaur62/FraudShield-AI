
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path("data/raw/creditcard.csv")
OUTPUT_DIR = Path("reports/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    # 1. Basic dataset information
    print("\n========== DATASET OVERVIEW ==========")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum():,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    # 2. Fraud class distribution
    print("\n========== CLASS DISTRIBUTION ==========")
    counts = df["Class"].value_counts().sort_index()
    print(counts.rename(index={0: "Legitimate", 1: "Fraud"}))

    print("\nClass percentages:")
    print((df["Class"].value_counts(normalize=True) * 100).round(4))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="Class", hue="Class",
                  palette=["#278C75", "#D94F70"],
                  legend=False, ax=ax)
    ax.set_title("Legitimate vs Fraudulent Transactions")
    ax.set_xlabel("Class (0 = Legitimate, 1 = Fraud)")
    ax.set_ylabel("Number of Transactions")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Transaction amount analysis
    print("\n========== AMOUNT SUMMARY ==========")
    print(df.groupby("Class")["Amount"].describe().round(2))

    # A log-scaled horizontal axis makes large amounts easier to compare.
    # log1p safely handles zero-valued amounts.
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, name, color in [
        (0, "Legitimate", "#278C75"),
        (1, "Fraud", "#D94F70"),
    ]:
        values = df.loc[df["Class"] == label, "Amount"]
        sns.histplot(
            x=values,
            bins=50,
            stat="density",
            element="step",
            fill=False,
            log_scale=(True, False),
            label=name,
            color=color,
            ax=ax,
        )

    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Transaction Amount (log scale)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "amount_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Dataset time feature
    # Time is elapsed time in the dataset, not a clock time of day.
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=df,
        x="Time",
        hue="Class",
        bins=60,
        element="step",
        stat="count",
        common_norm=False,
        ax=ax,
    )
    ax.set_title("Transaction Counts Across Dataset Time")
    ax.set_xlabel("Elapsed Time (seconds)")
    ax.set_ylabel("Transaction Count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "time_distribution.png", dpi=150)
    plt.close(fig)

    # 5. Features most correlated with the fraud label
    correlations = (
        df.corr(numeric_only=True)["Class"]
        .drop("Class")
        .sort_values(key=abs, ascending=False)
        .head(10)
        .sort_values()
    )

    print("\n========== TOP CLASS CORRELATIONS ==========")
    print(correlations.round(4))

    fig, ax = plt.subplots(figsize=(9, 6))
    correlations.plot(kind="barh", ax=ax, color="#6875D9")
    ax.set_title("Top 10 Feature Correlations with Fraud Label")
    ax.set_xlabel("Pearson Correlation")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_correlations.png", dpi=150)
    plt.close(fig)

    print("\nEDA completed successfully!")
    print(f"Charts saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
