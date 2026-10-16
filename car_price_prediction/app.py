from pathlib import Path
import pickle
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"

app = Flask(__name__, static_folder="static", static_url_path="/static")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file is missing. Run: python train_model.py"
        )
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


MODEL = load_model()


@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    required = [
        "Car_Name", "Year", "Present_Price", "Driven_kms",
        "Fuel_Type", "Selling_type", "Transmission", "Owner"
    ]
    missing = [field for field in required if field not in data or data[field] in ("", None)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        year = int(data["Year"])
        present_price = float(data["Present_Price"])
        driven_kms = float(data["Driven_kms"])
        owner = int(data["Owner"])

        if not 1980 <= year <= 2035:
            raise ValueError("Year must be between 1980 and 2035.")
        if present_price <= 0:
            raise ValueError("Present price must be greater than 0.")
        if driven_kms < 0:
            raise ValueError("Driven kilometers cannot be negative.")
        if owner not in (0, 1, 2, 3):
            raise ValueError("Owner must be 0, 1, 2, or 3.")

        row = {
            "Car_Name": str(data["Car_Name"]).strip(),
            "Year": year,
            "Present_Price": present_price,
            "Driven_kms": driven_kms,
            "Fuel_Type": str(data["Fuel_Type"]).strip(),
            "Selling_type": str(data["Selling_type"]).strip(),
            "Transmission": str(data["Transmission"]).strip(),
            "Owner": owner,
        }

        prediction = float(MODEL.predict([row])[0])
        prediction = max(0.0, prediction)

        return jsonify({
            "predicted_price": round(prediction, 2),
            "currency": "lakh INR",
            "message": "Prediction generated successfully."
        })
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the input values and trained model."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
