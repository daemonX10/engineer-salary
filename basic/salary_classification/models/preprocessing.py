import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging
from config import CAT_COLUMNS, NUM_COLUMNS, ORDINAL_COLUMNS, ORDINAL_MAPPINGS

logger = logging.getLogger(__name__)

def get_categorical_info(df, cat_columns):
    """
    Get categorical feature information for TabNet
    
    Args:
        df: DataFrame containing the categorical columns
        cat_columns: List of categorical column names
    
    Returns:
        cat_dims: List of dimensions for each categorical feature
        cat_idxs: List of indices for categorical features
    """
    cat_dims = []
    for col in cat_columns:
        if col in df.columns:
            cat_dims.append(df[col].nunique())
        else:
            cat_dims.append(2)  # Default if column not found
    
    # Create list of indices for categorical features
    cat_idxs = [i for i, col in enumerate(df.columns) if col in cat_columns]
    
    return cat_dims, cat_idxs

def preprocess_for_tabnet(X_train, X_val=None, cat_columns=None, num_columns=None):
    """
    Preprocess data for TabNet model
    
    Args:
        X_train: Training data
        X_val: Validation data (optional)
        cat_columns: List of categorical column names
        num_columns: List of numerical column names
    
    Returns:
        X_train_processed: Processed training data
        X_val_processed: Processed validation data (if provided)
        cat_dims: List of dimensions for each categorical feature
        cat_idxs: List of indices for categorical features
    """
    if cat_columns is None:
        cat_columns = CAT_COLUMNS
        
    if num_columns is None:
        num_columns = NUM_COLUMNS
    
    # Create a copy to avoid modifying original data
    X_train = X_train.copy()
    
    # For categorical features: Label encode
    label_encoders = {}
    for col in cat_columns:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].fillna('Unknown'))
            label_encoders[col] = le
    
    # For numerical features: Impute and standardize
    num_imputer = SimpleImputer(strategy='mean')
    num_scaler = StandardScaler()
    
    num_cols_present = [col for col in num_columns if col in X_train.columns]
    if num_cols_present:
        X_train[num_cols_present] = num_imputer.fit_transform(X_train[num_cols_present])
        X_train[num_cols_present] = num_scaler.fit_transform(X_train[num_cols_present])
    
    # Process validation data if provided
    if X_val is not None:
        X_val = X_val.copy()
        
        # Apply same transformations to validation data
        for col in cat_columns:
            if col in X_val.columns and col in label_encoders:
                # Handle unseen categories
                X_val[col] = X_val[col].fillna('Unknown')
                X_val[col] = X_val[col].map(lambda x: x if x in label_encoders[col].classes_ else 'Unknown')
                X_val[col] = label_encoders[col].transform(X_val[col])
        
        if num_cols_present:
            X_val[num_cols_present] = num_imputer.transform(X_val[num_cols_present])
            X_val[num_cols_present] = num_scaler.transform(X_val[num_cols_present])
        
        # Get categorical info for TabNet
        cat_dims, cat_idxs = get_categorical_info(X_train, cat_columns)
        
        return X_train, X_val, cat_dims, cat_idxs
    
    # Get categorical info for TabNet
    cat_dims, cat_idxs = get_categorical_info(X_train, cat_columns)
    
    return X_train, cat_dims, cat_idxs 