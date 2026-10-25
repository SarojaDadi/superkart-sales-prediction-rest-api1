from pathlib import Path
import io
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "SuperKart_prediction_model_v1_0.joblib"
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "SuperKart_prediction_model_v1_0"})


@app.post("/v1/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send one JSON object containing the feature columns."}), 400
    try:
        features = pd.DataFrame([payload])
        prediction = float(model.predict(features)[0])
        return jsonify({"predicted_sales": prediction})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/v1/predictbatch")
def predict_batch():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "Upload a CSV using the 'file' form field."}), 400
    try:
        data = pd.read_csv(uploaded)
        predictions = model.predict(data)
        result = data.copy()
        result["Predicted_Product_Store_Sales_Total"] = predictions
        return result.to_json(orient="records")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
