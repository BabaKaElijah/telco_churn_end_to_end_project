import pandas as pd
from predict_pipeline import load_artifacts, predict_churn

# Define new customer data
new_customers = pd.DataFrame([
    {
        "customerID": "0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.5,
        "TotalCharges": 400.5
    },
    {
        "customerID": "0002",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 60.2,
        "TotalCharges": 1450.2
    }
])

# Load artifacts
model, scaler, training_columns = load_artifacts()

# Predict churn
preds, preds_proba = predict_churn(new_customers, model, scaler, training_columns)

# Add predictions to dataframe
new_customers["Predicted_Churn"] = preds
new_customers["Churn_Probability"] = preds_proba

# Display results
print(new_customers[["customerID", "Predicted_Churn", "Churn_Probability"]])
