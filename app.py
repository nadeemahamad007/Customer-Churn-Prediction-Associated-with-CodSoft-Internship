from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "churn_model.pkl"

app = Flask(__name__)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get("/")
def healthcheck():
    return jsonify(
        {
            "message": "Customer Churn Prediction API is running.",
            "prediction_endpoint": "/predict",
            "model_loaded": model is not None,
        }
    )


@app.post("/predict")
def predict():
    if model is None:
        return jsonify(
            {
                "error": "Model file not found. Train the model first with `python src/train.py`."
            }
        ), 400

    payload = request.get_json(silent=True)

    if not payload:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    if "features" in payload:
        features = payload["features"]
    else:
        features = payload

    if not isinstance(features, dict):
        return jsonify({"error": "Provide customer data as a JSON object."}), 400

    frame = pd.DataFrame([features])
    prediction = int(model.predict(frame)[0])
    response = {"churn_prediction": prediction}

    if hasattr(model, "predict_proba"):
        response["churn_probability"] = round(float(model.predict_proba(frame)[0][1]), 4)

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
