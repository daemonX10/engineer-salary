import joblib
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import logging
import json
import os

logger = logging.getLogger(__name__)

def save_object(obj, file_path):
    """Saves a Python object using joblib."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving object to {file_path}: {e}", exc_info=True)
        raise

def load_object(file_path):
    """Loads a Python object using joblib."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"No such file or directory: '{file_path}'")
        obj = joblib.load(file_path)
        logger.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {file_path}: {e}", exc_info=True)
        raise

def save_dataframe(df, file_path):
    """Saves a pandas DataFrame to parquet format."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, file_path)
        logger.info(f"DataFrame saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {file_path}: {e}", exc_info=True)
        raise

def load_dataframe(file_path):
    """Loads a pandas DataFrame from parquet format."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"No such file or directory: '{file_path}'")
        table = pq.read_table(file_path)
        df = table.to_pandas()
        logger.info(f"DataFrame loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {file_path}: {e}", exc_info=True)
        raise

def save_json(data, file_path):
    """Saves data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}", exc_info=True)
        raise

def ensure_json_file_exists(file_path):
    """Ensures a JSON file exists, creating an empty one if it doesn't."""
    if not os.path.exists(file_path):
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Create an empty JSON file (empty dict)
            with open(file_path, 'w') as f:
                f.write('{}')
            logger.info(f"Created empty JSON file at {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating empty JSON file at {file_path}: {e}")
            return False
    return True

def load_json(file_path):
    """Loads a JSON file into a dictionary."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            ensure_json_file_exists(file_path)
            return {}
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return {}

def check_or_create_dir(directory):
    """Checks if a directory exists, creates it if not."""
    try:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directory checked/created: {directory}")
    except OSError as e:
        logger.error(f"Error creating directory {directory}: {e}", exc_info=True)
        raise 