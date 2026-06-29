from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        rainfall = float(request.form["rainfall"])
        cloud = float(request.form["cloud"])
        season = int(request.form["season"])

        data = np.array([[rainfall, cloud, season]])

        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "⚠️ HIGH FLOOD RISK"
        else:
            result = "✅ LOW FLOOD RISK"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)