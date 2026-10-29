# 🛡️ FraudShield AI
### Intelligent Credit Card Fraud Detection & Investigation

**FraudShield AI** is a machine learning project that detects potentially fraudulent credit card transactions using a trained Random Forest classifier. It provides a FastAPI backend for predictions and a free, rule-based investigation assistant that explains the model's existing result.

The Random Forest model remains the decision authority. The investigation assistant provides context and review recommendations without changing the prediction or treating it as proof of fraud.

## ✨ Key Features

- **Fraud Prediction:** Classifies transactions as potentially fraudulent or legitimate.
- **Fraud Probability:** Returns the model's fraud score and configured decision threshold.
- **REST API:** Provides prediction, health-check, and investigation endpoints through FastAPI.
- **Input Validation:** Rejects missing features, unexpected features, and non-finite numeric values.
- **Free Investigation Assistant:** Produces a structured explanation locally without requiring paid OpenAI API credits.
- **Responsible Predictions:** Clearly communicates model limitations and recommends appropriate review.
- **Automated Testing:** Includes tests for API behaviour, input validation, investigation logic, and mocked AI responses.

## 🧰 Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Machine Learning | Scikit-learn, Random Forest |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| API | FastAPI, Uvicorn |
| Validation | Pydantic |
| Optional AI Integration | OpenAI Python SDK |
| Configuration | python-dotenv |
| Testing | Pytest |

## 🏗️ System Architecture

```text
                 Transaction Input
                        |
                        v
                 FastAPI Validation
                        |
                        v
                 Random Forest Model
                        |
                        v
             Prediction + Fraud Score
                        |
               +--------+--------+
               |                 |
               v                 v
            /predict         /investigate
                                   |
                                   v
                         Rule-Based Explanation
                                   |
                                   v
                       Summary, Limitations,
                      and Review Recommendations
```

**Design principle:** The model produces the prediction. The investigation layer explains that result and cannot override it.

## 🚀 Getting Started

### 1. Clone the repository

Replace the URL below with your actual GitHub repository URL.

```powershell
git clone YOUR_GITHUB_REPOSITORY_URL
cd FraudShieldAI
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Python is already installed and your virtual environment exists, activate the existing environment instead.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Verify the trained model

The API expects the trained model at:

```text
models/fraud_detection_model.joblib
```

The repository must contain this model file before you start the API. If you need to regenerate it, use the project's training script:

```powershell
python -m src.train_model
```

Run this only if the training script supports module execution in your current project setup; otherwise, use the project's documented script path.

### 5. Start the API

```powershell
python -m uvicorn src.api:app --reload
```

Open the following URLs in your browser:

- API base URL: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

The free rule-based investigation endpoint does not require OpenAI API credits.

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Displays basic API information |
| GET | `/health` | Reports API and model status |
| POST | `/predict` | Predicts the transaction's fraud class and score |
| POST | `/investigate` | Returns the prediction with a structured explanation |

### Example Request

The model expects every feature included in its trained feature list. The example below is illustrative; it is not a known fraudulent or legitimate real-world transaction.

```json
{
  "transaction": {
    "Time": 0.0,
    "V1": 0.0,
    "V2": 0.0,
    "V3": 0.0,
    "V4": 0.0,
    "V5": 0.0,
    "V6": 0.0,
    "V7": 0.0,
    "V8": 0.0,
    "V9": 0.0,
    "V10": 0.0,
    "V11": 0.0,
    "V12": 0.0,
    "V13": 0.0,
    "V14": 0.0,
    "V15": 0.0,
    "V16": 0.0,
    "V17": 0.0,
    "V18": 0.0,
    "V19": 0.0,
    "V20": 0.0,
    "V21": 0.0,
    "V22": 0.0,
    "V23": 0.0,
    "V24": 0.0,
    "V25": 0.0,
    "V26": 0.0,
    "V27": 0.0,
    "V28": 0.0,
    "Amount": 0.0
  }
}
```

Send this JSON to either endpoint using `POST`:

- `/predict`
- `/investigate`

You can try both endpoints through the interactive Swagger UI at `/docs`.

### Prediction Response

The `/predict` endpoint returns:

- Predicted class and human-readable label
- Fraud probability score
- Configured threshold
- Model name
- Educational-use disclaimer

### Investigation Response

The `/investigate` endpoint returns the model prediction together with:

- Investigation summary
- Interpretation of the available transaction fields and score
- Limitations
- Recommended review actions
- Confidence note

The explanation is rule-based and does not require a live generative AI service. An anonymized feature such as `V1` or `V2` is not assigned an invented real-world meaning.

## 🧪 Testing

Run the complete test suite:

```powershell
python -m pytest -v
```

**Latest verified result: 18 tests passed.**

The tests cover:

- Home and health endpoints
- Successful fraud predictions
- Missing and unexpected feature validation
- Investigation behaviour and preservation of the model prediction
- Investigation error handling
- Validation of mocked AI responses
- Rule-based investigation recommendations and limitations
- End-to-end `/investigate` endpoint behaviour without OpenAI credits

Run the tests again after making code changes to confirm the result remains current.

## 🔐 Responsible Use and Limitations

- A predicted fraudulent transaction is not proof of criminal activity.
- A legitimate classification does not guarantee that a transaction is safe.
- The model's fraud score is not necessarily a calibrated real-world probability.
- The anonymized `V1`–`V28` features are not given assumed real-world meanings.
- The rule-based investigation is an explanation, not an independent fraud investigation.
- The model's performance depends on its training data, preprocessing, and decision threshold.
- This project is intended for education and demonstration, not as a standalone financial decision system.

## 🗺️ Future Improvements

- Add a confusion matrix and documented precision, recall, and false-positive metrics.
- Add model explainability using suitable feature-attribution methods.
- Build a transaction monitoring dashboard.
- Add structured logging and improve operational error handling.
- Add deployment instructions and a hosted API demonstration.
- Evaluate the model on additional data and validate its probability calibration.

## 👩‍💻 Project Status

The project includes a trained Random Forest model, a FastAPI prediction service, a free rule-based investigation endpoint, and an automated test suite with **18 passing tests in the latest reported run**.

Live generative AI investigation remains optional and is not required for the current rule-based workflow.

---

**FraudShield AI — Detect risk. Explain evidence. Support informed review.**
