# 🛡️ FraudShield AI

### Intelligent Online Payment Fraud Detection Using Machine Learning

FraudShield AI is a machine learning project that identifies potentially fraudulent credit card transactions using transaction data and a Random Forest classifier. The project explores imbalanced datasets, evaluates fraud detection performance, and compares probability thresholds to balance fraud detection against false alarms.

> **Project status:** Machine learning pipeline completed. API and Generative AI investigation assistant are planned next.

## ✨ Features

- 📊 Dataset inspection and exploratory data analysis
- 🤖 Random Forest fraud classification
- ⚖️ Baseline comparison using a Dummy Classifier
- 📈 Precision, recall, F1-score, and average precision evaluation
- 🎚️ Fraud probability threshold comparison
- 🧾 Exportable evaluation results and charts

## 🧰 Tech Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualization:** Matplotlib
- **Model persistence:** Joblib
- **Development:** VS Code, Git, GitHub

## 📁 Project Structure

```text
FraudShield-AI/
├── data/
│   ├── raw/                  # Local dataset; not committed
│   └── processed/
├── models/                   # Locally generated model files
├── notebooks/
├── reports/
│   ├── eda/
│   └── model/
├── src/
│   ├── inspect_data.py
│   ├── eda.py
│   ├── train_model.py
│   └── tune_threshold.py
├── tests/
├── .gitignore
└── README.md
```

## 📊 Initial Model Results

The Random Forest model was evaluated on a held-out test split containing 56,962 transactions, including 98 fraudulent transactions.

| Metric | Random Forest |
|---|---:|
| Precision | 92.77% |
| Recall | 78.57% |
| F1-score | 85.08% |
| Average precision | 0.8669 |

### What do these results mean?

- **Precision:** About 92.77% of transactions flagged as fraud were actually fraudulent.
- **Recall:** The model detected about 78.57% of the fraudulent transactions in the test set.
- **F1-score:** Summarizes the balance between precision and recall.
- **Average precision:** Measures precision-recall performance across different classification thresholds.

These results are from the initial model and should not be interpreted as production performance.

## 🎚️ Threshold Analysis

The initial threshold comparison produced these results:

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.20 | 74.78% | 87.76% | 80.75% |
| 0.30 | 84.00% | 85.71% | 84.85% |
| 0.40 | 87.23% | 83.67% | 85.42% |
| 0.50 | 92.77% | 78.57% | 85.08% |
| 0.60 | 94.94% | 76.53% | 84.75% |
| 0.70 | 96.10% | 75.51% | 84.57% |

A lower threshold can detect more fraud but may produce more false alarms. A higher threshold tends to reduce false alarms but may miss more fraudulent transactions.

The threshold of 0.40 had the highest F1-score among the thresholds tested. A final threshold should be selected using validation data before evaluating the final model on an untouched test set.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/navkaur62/FraudShield-AI.git
cd FraudShield-AI
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install pandas numpy scikit-learn matplotlib joblib
```

### 4. Add the dataset

Place the `creditcard.csv` dataset in:

```text
data/raw/creditcard.csv
```

The dataset is not included in this repository. Obtain it from an authorized source and follow its applicable terms of use.

### 5. Inspect the data and train the model

```powershell
python .\src\inspect_data.py
python .\src\eda.py
python .\src\train_model.py
python .\src\tune_threshold.py
```

Generated models, reports, and charts are saved locally according to the project scripts and `.gitignore` rules.

## 🗺️ Roadmap

- [x] Dataset inspection
- [x] Exploratory data analysis script
- [x] Initial Random Forest model
- [x] Baseline evaluation
- [x] Probability threshold comparison
- [ ] Validation-based threshold selection
- [ ] Fraud prediction API using FastAPI
- [ ] API testing and input validation
- [ ] Generative AI fraud investigation assistant
- [ ] Interactive dashboard
- [ ] Automated tests and deployment

## ⚠️ Disclaimer

This is an educational project and is not intended to make real-world financial decisions. Model predictions can be incorrect, and any practical deployment would require additional validation, security controls, monitoring, and appropriate human oversight.

## 👩‍💻 Author

**Navdeep Kaur**

GitHub: [@navkaur62](https://github.com/navkaur62)
