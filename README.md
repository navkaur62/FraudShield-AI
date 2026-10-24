# 🛡️ FraudShieldAI
### Intelligent Credit Card Fraud Detection & Investigation

FraudShieldAI is a machine learning project that predicts potentially fraudulent credit card transactions and provides an optional AI-assisted investigation summary. It combines a Random Forest classifier with a FastAPI backend and an OpenAI-powered explanation layer.

The machine learning model remains the authority for predictions. The AI assistant is designed to explain the available evidence, not override the model or treat a prediction as proof of fraud.

## ✨ Features

- **Fraud Prediction:** Predicts whether a transaction is potentially fraudulent using a trained Random Forest model.
- **Probability Scoring:** Returns the model's estimated fraud probability and configured decision threshold.
- **REST API:** Exposes prediction and investigation endpoints through FastAPI.
- **Input Validation:** Rejects missing, unexpected, or non-finite transaction feature values.
- **AI Investigation Assistant:** Generates a structured explanation of the existing prediction when the OpenAI API is configured and available.
- **Safe Failure Handling:** Keeps the prediction endpoint independent of AI-service availability.
- **Automated Testing:** Tests prediction, validation, investigation errors, and mocked AI responses.

## 🧰 Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Machine Learning | Scikit-learn, Random Forest |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| API | FastAPI, Uvicorn |
| Validation | Pydantic |
| AI Integration | OpenAI Python SDK |
| Configuration | python-dotenv |
| Testing | Pytest |

## 🏗️ Architecture

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
 Prediction + Probability
       |
       +----------------------+
       |                      |
       v                      v
   /predict              /investigate
                              |
                              v
                     AI Explanation Layer
                              |
                              v
                    Structured Investigation
```

The AI investigation layer is optional and depends on OpenAI API availability. It does not replace the Random Forest prediction.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd FraudShieldAI
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file if you plan to use the live AI integration:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=your_supported_model_name
```

Use a supported model name available to your API account. Never commit your actual API key.

**Note:** Live AI investigations require an API account with available credits. Prediction and mocked AI tests can be run without purchasing API credits.

### 5. Start the API

```powershell
python -m uvicorn src.api:app --reload
```

The API runs locally at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Basic API information |
| GET | `/health` | Health check |
| POST | `/predict` | Predict transaction fraud risk |
| POST | `/investigate` | Return a prediction with an AI-generated investigation |

### Prediction Response

The `/predict` endpoint returns the predicted class, human-readable label, fraud probability, decision threshold, model name, and disclaimer.

The request must contain the transaction features expected by the trained model.

### Investigation Response

The `/investigate` endpoint returns the machine learning prediction together with a structured investigation containing a summary, evidence interpretation, limitations, recommended actions, and confidence note.

The investigation may return HTTP `503` if the AI service is unavailable. The `/predict` endpoint remains available independently.

## 🧪 Testing

Run the automated test suite:

```powershell
python -m pytest -v
```

The current test suite contains **14 passing tests** covering:

- Home and health endpoints
- Successful prediction requests
- Missing and unexpected features
- Preservation of the machine learning prediction
- AI-service failure handling
- AI response validation
- Successful investigation using a mocked AI response

Mocked AI tests do not make live OpenAI API calls.

## 🔐 Responsible Use

- A predicted fraudulent transaction is not proof of criminal activity.
- Predictions should be reviewed alongside appropriate, verified evidence.
- The AI assistant must not invent transaction details or override the model's result.
- The model's performance depends on the training data, preprocessing, and decision threshold.
- This project is intended for educational and demonstration purposes, not as a standalone financial decision system.

## 🗺️ Future Improvements

- Add model evaluation metrics and a confusion matrix.
- Document precision, recall, and false-positive performance.
- Add explainability tools for model features.
- Develop a transaction monitoring dashboard.
- Improve structured logging and API error handling.
- Add deployment instructions and API usage examples.

## 👩‍💻 Project Status

The current backend has a working prediction endpoint and an optional AI investigation endpoint. The automated test suite currently passes all 14 tests. Live AI generation has not been verified successfully because the configured OpenAI API account has exhausted its credits.

---

**FraudShieldAI — Detect risk. Explain evidence. Support informed review.**
