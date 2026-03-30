# model.py

from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Dummy training data
# [cases, vaccination_rate, population]
X = np.array([
    [100, 80, 1000],
    [500, 30, 2000],
    [300, 50, 1500],
    [800, 20, 3000],
])

# Risk: 0 = Low, 1 = High
y = np.array([0, 1, 0, 1])

model = LogisticRegression()
model.fit(X, y)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    cases = data.get("cases")
    vaccination = data.get("vaccination")
    population = data.get("population")

    input_data = np.array([[cases, vaccination, population]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "High Risk" if prediction == 1 else "Low Risk"

    return jsonify({
        "risk": result,
        "confidence": float(probability)
    })


if __name__ == "__main__":
    app.run(port=5001)