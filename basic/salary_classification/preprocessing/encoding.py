import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder
import logging
import numpy as np
from utils.helpers import save_object, load_object
import config
import os

logger = logging.getLogger(__name__)

def label_encode_target(df, target_column):
    """Applies Label Encoding to the target variable."""
    le = LabelEncoder()
    if target_column in df.columns:
        # Handle potential NaNs or invalid values
        if df[target_column].isna().any():
            logger.warning(f"Target column '{target_column}' contains {df[target_column].isna().sum()} NaN values. These will be filled with 'Unknown'.")
            df[target_column] = df[target_column].fillna('Unknown')
        
        # Convert to string to handle potential numeric categories
        df[target_column] = df[target_column].astype(str)
        
        # Fit and transform
        y = le.fit_transform(df[target_column])
        logger.info(f"Target column '{target_column}' label encoded. {len(le.classes_)} unique classes found.")
        
        # Save encoder and class mappings for reference
        save_object(le, config.LABEL_ENCODER_FILE)
        
        # Save mapping dictionary for debugging
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        pd.DataFrame({'original': le.classes_, 'encoded': range(len(le.classes_))}).to_csv(
            os.path.join(config.OUTPUT_DIR, 'label_encoding_map.csv'), index=False
        )
        logger.info(f"Label encoder and mapping saved.")
        
        return y, le
    else:
        logger.warning(f"Target column '{target_column}' not found for label encoding.")
        return None, None

def apply_label_encoder(y_pred_numeric, le_path=config.LABEL_ENCODER_FILE):
    """Applies a saved LabelEncoder to decode predictions."""
    try:
        # Ensure y_pred_numeric is a numpy array
        y_pred_numeric = np.asarray(y_pred_numeric)
        
        # Load the encoder
        le = load_object(le_path)
        if le is None:
            raise ValueError("Loaded label encoder is None")
        
        # Validate that the encoder has classes_
        if not hasattr(le, 'classes_') or len(le.classes_) == 0:
            raise AttributeError("Label encoder does not have valid classes_")
        
        # Ensure predictions are within valid range
        valid_range = range(len(le.classes_))
        invalid_values = np.setdiff1d(y_pred_numeric, valid_range)
        
        if len(invalid_values) > 0:
            logger.warning(f"Found {len(invalid_values)} invalid class indices. Clipping to valid range.")
            y_pred_numeric = np.clip(y_pred_numeric, 0, len(le.classes_) - 1)
        
        # Decode predictions
        y_pred_labels = le.inverse_transform(y_pred_numeric)
        logger.info(f"Predictions decoded from {len(y_pred_numeric)} values to original labels.")
        
        return y_pred_labels
    
    except FileNotFoundError:
        logger.error(f"Label encoder file not found at {le_path}")
        # Convert to string as fallback
        return [f"class_{p}" for p in y_pred_numeric]
    
    except Exception as e:
        logger.error(f"Error applying label encoder: {e}", exc_info=True)
        # Try to extract classes directory if the encoder was loaded but inverse_transform failed
        try:
            if 'le' in locals() and hasattr(le, 'classes_'):
                logger.warning("Attempting manual decoding using classes_")
                return [le.classes_[min(int(p), len(le.classes_) - 1)] for p in y_pred_numeric]
        except Exception as e2:
            logger.error(f"Manual decoding failed: {e2}")
        
        # Final fallback if everything else fails
        return [f"class_{p}" for p in y_pred_numeric]


def target_encode_feature(X_train, y_train, X_val, X_test, column, smoothing=1.0, min_samples_leaf=1):
    """Applies Target Encoding to a specific feature."""
    logger.info(f"Applying Target Encoding to column: {column}")
    encoder = TargetEncoder(cols=[column], smoothing=smoothing, min_samples_leaf=min_samples_leaf)

    # Fit on training data
    X_train[f'{column}_encoded'] = encoder.fit_transform(X_train[[column]], y_train)

    # Transform validation and test data
    X_val[f'{column}_encoded'] = encoder.transform(X_val[[column]])
    X_test[f'{column}_encoded'] = encoder.transform(X_test[[column]])

    # Handle potential NaNs introduced by unseen categories in validation/test
    # Use the global mean from the training encoding
    global_mean = y_train.mean() # Get the mean of the encoded target
    X_val[f'{column}_encoded'].fillna(global_mean, inplace=True)
    X_test[f'{column}_encoded'].fillna(global_mean, inplace=True)

    logger.info(f"Target Encoding applied to {column}. Global mean fallback: {global_mean:.4f}")
    save_object(encoder, config.MODELS_DIR / f"{column}_target_encoder.joblib")

    return X_train, X_val, X_test, encoder


def apply_saved_target_encoder(df, column):
    """Applies a saved Target Encoder to a dataframe."""
    encoder_path = config.MODELS_DIR / f"{column}_target_encoder.joblib"
    try:
        encoder = load_object(encoder_path)
        df[f'{column}_encoded'] = encoder.transform(df[[column]])
        # Handle NaNs using a reasonable default or saved global mean if possible
        # For simplicity, using 0.5 (assuming binary or scaled target initially)
        # A better approach would be to save the global mean during training.
        df[f'{column}_encoded'].fillna(0.5, inplace=True)
        logger.info(f"Applied saved Target Encoder for {column}.")
    except FileNotFoundError:
        logger.warning(f"Target encoder file not found for {column} at {encoder_path}. Filling with 0.5.")
        df[f'{column}_encoded'] = 0.5
    except Exception as e:
        logger.error(f"Error applying saved target encoder for {column}: {e}. Filling with 0.5.", exc_info=True)
        df[f'{column}_encoded'] = 0.5
    return df

def boolean_to_int(df, columns):
    """Converts boolean columns (or columns that can be interpreted as bool) to integers (0/1)."""
    for col in columns:
        if col in df.columns:
            # Handle potential non-boolean types safely
            try:
                df[col] = df[col].astype(bool).astype(int)
            except Exception as e:
                logger.warning(f"Could not convert column {col} to boolean int: {e}. Skipping.")
    logger.info(f"Boolean columns converted to int: {columns}")
    return df 