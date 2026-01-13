import os
import pandas as pd
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def load_data(file_name: str) -> pd.DataFrame:
    """
    Load CSV dataset using absolute path based on current script location.
    """
    try:
        # Get the folder where this script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build absolute path to CSV
        file_path = os.path.join(base_dir, "../data", file_name)
        
        if not os.path.exists(file_path):
            message = f"File not found: {file_path}"
            logger.error(message)
            raise CustomException(message)
        
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    except Exception as e:
        raise CustomException(f"Failed to load data from {file_name}", e)


if __name__ == "__main__":
    df = load_data("telco_churn.csv")
    print(df.head())
