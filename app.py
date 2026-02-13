from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model & features
model = pickle.load(open("heart_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])

def predict():
    print("Received request:", request.json)

    data = request.json

    input_data = [float(data[f]) for f in features]
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][prediction]

    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    return jsonify({
        "prediction": result,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
