from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Correct path to model folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "rul_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "model", "scaler.pkl")

# Load model and scaler
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

if __name__ == "__main__":
    app.run(debug=False)

from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([list(data.values())])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)