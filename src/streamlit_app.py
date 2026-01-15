import streamlit as st
import pandas as pd
from src.predict_pipeline import load_artifacts, predict_churn

st.title("Telco Churn Prediction")

model, scaler, training_columns = load_artifacts()

st.sidebar.header("Customer Details")
customer = {
    "customerID": st.sidebar.text_input("Customer ID", "0001"),
    "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
    "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", [0, 1]),
    "Partner": st.sidebar.selectbox("Partner", ["Yes", "No"]),
    "Dependents": st.sidebar.selectbox("Dependents", ["Yes", "No"]),
    "tenure": st.sidebar.number_input("Tenure (months)", 0, 72, 12),
    "PhoneService": st.sidebar.selectbox("Phone Service", ["Yes", "No"]),
    "MultipleLines": st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"]),
    "InternetService": st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    "OnlineSecurity": st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"]),
    "OnlineBackup": st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
    "DeviceProtection": st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
    "TechSupport": st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
    "StreamingTV": st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
    "StreamingMovies": st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
    "Contract": st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": st.sidebar.selectbox("Paperless Billing", ["Yes", "No"]),
    "PaymentMethod": st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
    "MonthlyCharges": st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0),
    "TotalCharges": st.sidebar.number_input("Total Charges", 0.0, 5000.0, 840.0)
}

if st.button("Predict Churn"):
    df = pd.DataFrame([customer])
    preds, preds_proba = predict_churn(df, model, scaler, training_columns)
    st.write(f"Predicted Churn: {preds[0][0]}")
    st.write(f"Churn Probability: {preds_proba[0][0]:.2f}")
