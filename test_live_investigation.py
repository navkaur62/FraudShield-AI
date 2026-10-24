
import json
import urllib.request
import urllib.error

import joblib

MODEL_PATH = "models/fraud_detection_model.joblib"
API_URL = "http://127.0.0.1:8000/investigate"

bundle = joblib.load(MODEL_PATH)

# Build a request containing every required model feature.
transaction = {
    feature: 0.0
    for feature in bundle["features"]
}

payload = json.dumps({
    "transaction": transaction
}).encode("utf-8")

request = urllib.request.Request(
    API_URL,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=90) as response:
        print("HTTP status:", response.status)
        result = json.loads(response.read().decode("utf-8"))
        print(json.dumps(result, indent=2))

except urllib.error.HTTPError as error:
    print("HTTP status:", error.code)
    print("Response details:")
    print(error.read().decode("utf-8"))

except Exception as error:
    print("Request failed:", str(error))
