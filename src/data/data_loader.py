import pandas as pd
import yaml
import os
import logging

logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(data_path):
    """
    Load data from a given CSV path.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    return pd.read_csv(data_path)

def validate_data(df, required_columns):
    """
    Validate that the dataframe contains all required columns.
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True
