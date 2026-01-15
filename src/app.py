from flask import Flask, render_template, request
import os
from predict_pipeline import load_artifacts, predict_churn
import pandas as pd

# Set templates folder explicitly
app = Flask(
    __name__, 
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../templates")
)

# Load trained model, scaler, and training columns
model, scaler, training_columns = load_artifacts()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "gender": request.form.get("gender"),
        "SeniorCitizen": int(request.form.get("SeniorCitizen", 0)),
        "Partner": request.form.get("Partner", "No"),
        "Dependents": request.form.get("Dependents", "No"),
        "tenure": int(request.form.get("tenure", 0)),
        "PhoneService": request.form.get("PhoneService", "No"),
        "PaperlessBilling": request.form.get("PaperlessBilling", "No"),
        "MonthlyCharges": float(request.form.get("MonthlyCharges", 0)),
        "TotalCharges": float(request.form.get("TotalCharges", 0)),
        "MultipleLines": request.form.get("MultipleLines", "No"),
        "InternetService": request.form.get("InternetService", "No"),
        "OnlineSecurity": request.form.get("OnlineSecurity", "No"),
        "OnlineBackup": request.form.get("OnlineBackup", "No"),
        "DeviceProtection": request.form.get("DeviceProtection", "No"),
        "TechSupport": request.form.get("TechSupport", "No"),
        "StreamingTV": request.form.get("StreamingTV", "No"),
        "StreamingMovies": request.form.get("StreamingMovies", "No"),
        "Contract": request.form.get("Contract", "Month-to-month"),
        "PaymentMethod": request.form.get("PaymentMethod", "Electronic check")
    }

    df = pd.DataFrame([data])
    preds, probs = predict_churn(df, model, scaler, training_columns)

    return render_template(
        "index.html",
        prediction=int(preds[0][0]),
        probability=float(probs[0][0])
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

