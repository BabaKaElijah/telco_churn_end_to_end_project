import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def load_artifacts(model_path="artifacts/churn_model.h5",
                   scaler_path="artifacts/scaler.pkl",
                   columns_path="artifacts/training_columns.pkl"):
    """
    Load saved model, scaler, and training columns
    """
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        training_columns = joblib.load(columns_path)
        logger.info("Model, scaler, and training columns loaded successfully")
        return model, scaler, training_columns
    except Exception as e:
        raise CustomException("Failed to load artifacts", e)


def preprocess_new_data(df: pd.DataFrame, scaler, training_columns):
    """
    Preprocess new customer data to match training features
    """
    try:
        # Drop customerID if present
        if "customerID" in df.columns:
            df = df.drop("customerID", axis=1)

        # Binary encoding
        binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Female": 1, "Male": 0})

        # One-hot encode multi-category columns
        multi_cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                          "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                          "Contract", "PaymentMethod"]
        df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

        # Add missing columns and ensure same order as training
        for col in training_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[training_columns]

        # Scale numeric columns
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        return df

    except Exception as e:
        raise CustomException("Error in preprocessing new data", e)


def predict_churn(df: pd.DataFrame, model, scaler, training_columns):
    """
    Predict churn for new customer data
    """
    try:
        X = preprocess_new_data(df, scaler, training_columns)
        preds_proba = model.predict(X)
        preds = (preds_proba >= 0.5).astype(int)
        return preds, preds_proba
    except Exception as e:
        raise CustomException("Prediction failed", e)


if __name__ == "__main__":
    from data_ingestion import load_data

    # Load model, scaler, and training columns
    model, scaler, training_columns = load_artifacts()

    # Example: predict on new data (5 sample rows)
    df_new = load_data("telco_churn.csv").sample(5, random_state=42)
    preds, preds_proba = predict_churn(df_new, model, scaler, training_columns)

    df_new["Predicted_Churn"] = preds
    df_new["Churn_Probability"] = preds_proba

    print(df_new[["Predicted_Churn", "Churn_Probability"]])
