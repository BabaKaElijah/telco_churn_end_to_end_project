import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from logger import get_logger
from exception import CustomException
from data_ingestion import load_data
from data_transformation import preprocess_data

logger = get_logger(__name__)

def build_model(input_dim: int) -> Sequential:
    """
    Build a simple feedforward neural network for binary classification
    """
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    logger.info("Model compiled successfully")
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Train the model
    """
    history = model.fit(X_train, y_train, validation_split=0.2,
                        epochs=epochs, batch_size=batch_size)
    logger.info("Model training complete")
    return history

def save_artifacts(model, scaler, training_columns, artifacts_dir="artifacts"):
    """
    Save model, scaler, and training columns for later use
    """
    try:
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)

        model_path = os.path.join(artifacts_dir, "churn_model.h5")
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        columns_path = os.path.join(artifacts_dir, "training_columns.pkl")

        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(training_columns, columns_path)

        logger.info(f"Model saved at {model_path}")
        logger.info(f"Scaler saved at {scaler_path}")
        logger.info(f"Training columns saved at {columns_path}")

    except Exception as e:
        raise CustomException("Failed to save artifacts", e)


if __name__ == "__main__":
    try:
        # Load and preprocess data
        df = load_data("telco_churn.csv")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)  # capture scaler

        # Build and train model
        model = build_model(X_train.shape[1])
        history = train_model(model, X_train, y_train, epochs=50, batch_size=32)

        # Save artifacts including training columns
        save_artifacts(model, scaler, X_train.columns.tolist())

    except CustomException as ce:
        logger.error(ce)
