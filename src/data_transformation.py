import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from logger import get_logger
from exception import CustomException

logger = get_logger(__name__)

def preprocess_data(df: pd.DataFrame, artifacts_dir="artifacts"):
    try:
        # Fill missing TotalCharges
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
        logger.info(f"Missing TotalCharges handled. Nulls now: {df['TotalCharges'].isna().sum()}")

        # Drop customerID
        if "customerID" in df.columns:
            df = df.drop("customerID", axis=1)

        # Encode target
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        # Binary categorical encoding
        binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Female": 1, "Male": 0})

        # One-hot encode multi-category columns
        multi_cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                          "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                          "Contract", "PaymentMethod"]
        df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

        # Scale numeric columns
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Split features and target
        X = df.drop("Churn", axis=1)
        y = df["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Data split into train and test sets: X_train {X_train.shape}, X_test {X_test.shape}")

        # Save scaler and training columns
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
        joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
        joblib.dump(X_train.columns.tolist(), os.path.join(artifacts_dir, "training_columns.pkl"))
        logger.info(f"Scaler and training columns saved in {artifacts_dir}")

        return X_train, X_test, y_train, y_test, scaler

    except Exception as e:
        raise CustomException("Error in data preprocessing", e)


if __name__ == "__main__":
    from data_ingestion import load_data

    df = load_data("telco_churn.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print("Preprocessing complete")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train distribution:\n", y_train.value_counts())
