import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

def impute_numerical(df, columns):
    """Imputes numerical columns using the median."""
    imputer = SimpleImputer(strategy='median')
    for col in columns:
        if col in df.columns:
            # SimpleImputer expects 2D array
            df[col] = imputer.fit_transform(df[[col]])
            if df[col].isnull().sum() > 0:
                 logger.warning(f"Imputation failed for {col}, filling remaining NaNs with 0")
                 df[col] = df[col].fillna(0)
    logger.info(f"Numerical imputation completed for columns: {columns}")
    return df

def impute_categorical(df, columns, fill_value='Unknown'):
    """Imputes categorical columns with a specified fill value."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    logger.info(f"Categorical imputation completed for columns: {columns}")
    return df

def impute_boolean(df, columns, fill_value=False):
    """Imputes boolean columns with a specified fill value (typically False -> 0)."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    logger.info(f"Boolean imputation completed for columns: {columns}")
    return df

def impute_job_desc(df, columns, fill_value=0):
    """Imputes job description features with a specified fill value (typically 0)."""
    for col in columns:
         if col in df.columns:
            # Convert to numeric first, coercing errors
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(fill_value)
    logger.info(f"Job description imputation completed.")
    return df 