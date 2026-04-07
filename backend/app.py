from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import gdown

app = Flask(__name__,
            template_folder="templates",
            static_folder="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "rul_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "model", "scaler.pkl")

# Google Drive File IDs
MODEL_ID = "1Y4oLNpd2c2EysWpV-Bd2EKKdSzULS-qb"
SCALER_ID = "1A7XoVTwKn0QL0cuiQWBZedaDqEA5MW42"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.join(BASE_DIR, "..", "model"), exist_ok=True)
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

if not os.path.exists(SCALER_PATH):
    url = f"https://drive.google.com/uc?id={SCALER_ID}"
    gdown.download(url, SCALER_PATH, quiet=False)

# Load model
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/turbofan")
def turbofan():
    return render_template("turbofan.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = []

    for i in range(1, 25):
        value = float(request.form[f"f{i}"])
        features.append(value)

    scaled_data = scaler.transform([features])
    prediction = model.predict(scaled_data)[0]

    health = "Healthy"
    suggestion = "Motor operating normally."

    if prediction < 60:
        health = "Critical"
        suggestion = "Immediate inspection recommended."

    elif prediction < 120:
        health = "Warning"
        suggestion = "Schedule maintenance soon."

    return render_template(
        "turbofan.html",
        prediction=round(float(prediction), 2),
        health=health,
        suggestion=suggestion
    )