from flask import Flask, request, jsonify, render_template
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("../model/rul_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/turbofan")
def turbofan():
    return render_template("turbofan.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = []

    for i in range(1,25):
        value = float(request.form[f"f{i}"])
        features.append(value)

    scaled_data = scaler.transform([features])
    prediction = model.predict(scaled_data)[0]

    health = "Healthy"
    suggestion = "Engine operating normally."

    if prediction < 60:
        health = "Critical"
        suggestion = "Immediate inspection recommended."

    elif prediction < 120:
        health = "Warning"
        suggestion = "Schedule maintenance soon."

    return render_template(
        "turbofan.html",
        prediction=round(float(prediction),2),
        health=health,
        suggestion=suggestion
    )

if __name__ == "__main__":
    app.run(debug=True)