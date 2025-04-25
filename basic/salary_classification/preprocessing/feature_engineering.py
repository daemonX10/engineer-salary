import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from utils.helpers import save_object, load_object
import config
import joblib

# Add UMAP import
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    
logger = logging.getLogger(__name__)

def engineer_date_features(df, date_column):
    """Engineers features from a date column (YYYY/MM format)."""
    if date_column not in df.columns:
        logger.warning(f"Date column '{date_column}' not found.")
        return df

    # Fill NaNs with a placeholder far in the past
    df[date_column] = df[date_column].fillna('1900/01')
    df[date_column] = df[date_column].astype(str) # Ensure string type

    # Extract year and month safely
    def extract_year(date_str):
        try:
            return int(str(date_str)[:4])
        except (TypeError, ValueError, IndexError):
            logger.debug(f"Could not parse year from '{date_str}', using 1900")
            return 1900

    def extract_month(date_str):
        try:
            parts = str(date_str).split('/')
            if len(parts) > 1:
                return int(parts[1])
            return 1 # Default month
        except (TypeError, ValueError, IndexError):
            logger.debug(f"Could not parse month from '{date_str}', using 1")
            return 1

    df['job_posted_year'] = df[date_column].apply(extract_year)
    df['job_posted_month'] = df[date_column].apply(extract_month)

    # Create cyclical features for month
    df['month_sin'] = np.sin(2 * np.pi * df['job_posted_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['job_posted_month'] / 12)

    # Create feature for recency (months since a reference point)
    max_year = df['job_posted_year'].max() if df['job_posted_year'].max() > 1900 else 2024
    reference_year = max_year
    reference_month = df.loc[df['job_posted_year'] == max_year, 'job_posted_month'].max() if max_year > 1900 else 12

    df['job_recency_months'] = (reference_year - df['job_posted_year']) * 12 + (reference_month - df['job_posted_month'])
    # Clip negative recency just in case
    df['job_recency_months'] = df['job_recency_months'].clip(lower=0)

    # Normalize year relative to mean
    mean_year = df.loc[df['job_posted_year'] > 1900, 'job_posted_year'].mean()
    mean_year = mean_year if pd.notna(mean_year) else 2020 # Fallback
    df['job_posted_year_norm'] = df['job_posted_year'] - mean_year

    logger.info("Date features engineered: year, month, sin/cos month, recency, normalized year.")
    return df.drop(columns=[date_column], errors='ignore') # Drop original date

def engineer_job_title_features(df, column='job_title'):
    """Engineers features based on job title keywords and rarity."""
    if column not in df.columns:
        logger.warning(f"Job title column '{column}' not found.")
        return df

    df[column] = df[column].fillna('Unknown').astype(str).str.lower()

    # Keyword features
    df['is_senior'] = df[column].str.contains('senior|sr|lead|principal|mgr|manager', regex=True).astype(int)
    df['is_junior'] = df[column].str.contains('junior|jr|intern|associate|entry|grad', regex=True).astype(int)
    df['is_engineer'] = df[column].str.contains('engineer|developer|programmer|software|swe', regex=True).astype(int)
    df['is_analyst'] = df[column].str.contains('analyst|analysis', regex=True).astype(int)
    df['is_data_scientist'] = df[column].str.contains('data scientist|scientist', regex=True).astype(int)
    df['is_manager'] = df[column].str.contains('manager|mgr|director|vp|head', regex=True).astype(int)

    # Rarity (group infrequent titles)
    threshold = 10
    title_counts = df[column].value_counts()
    rare_titles = title_counts[title_counts < threshold].index
    df[f'{column}_processed'] = df[column].apply(lambda x: 'Other_Title' if x in rare_titles else x)

    logger.info("Job title features engineered: keywords (senior, junior, etc.), processed title.")
    return df

def engineer_job_desc_aggregates(df, job_desc_cols):
    """Creates aggregate statistics from job description features."""
    valid_cols = [col for col in job_desc_cols if col in df.columns]
    if not valid_cols:
        logger.warning("No valid job description columns found for aggregation.")
        return df

    job_desc_data = df[valid_cols].fillna(0)

    with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings
        df['job_desc_mean'] = job_desc_data.mean(axis=1)
        df['job_desc_std'] = job_desc_data.std(axis=1)
        df['job_desc_sum'] = job_desc_data.sum(axis=1)
        df['job_desc_median'] = job_desc_data.median(axis=1)
        df['job_desc_max'] = job_desc_data.max(axis=1)
        df['job_desc_min'] = job_desc_data.min(axis=1)
        df['job_desc_non_zero_count'] = (job_desc_data != 0).sum(axis=1)
        df['job_desc_non_zero_ratio'] = df['job_desc_non_zero_count'] / len(valid_cols)

        # Quantiles
        df['job_desc_q25'] = job_desc_data.quantile(0.25, axis=1)
        df['job_desc_q75'] = job_desc_data.quantile(0.75, axis=1)
        df['job_desc_iqr'] = df['job_desc_q75'] - df['job_desc_q25']

    # Fill NaNs potentially created by std/quantiles on all-zero rows
    agg_cols = ['job_desc_mean', 'job_desc_std', 'job_desc_sum', 'job_desc_median',
                'job_desc_max', 'job_desc_min', 'job_desc_non_zero_count',
                'job_desc_non_zero_ratio', 'job_desc_q25', 'job_desc_q75', 'job_desc_iqr']
    for col in agg_cols:
        if col in df.columns:
             df[col] = df[col].fillna(0)

    logger.info("Job description aggregate features engineered.")
    return df

def apply_pca_job_desc(df_train, df_test, job_desc_cols, n_components):
    """Applies PCA to job description features."""
    valid_cols = [col for col in job_desc_cols if col in df_train.columns]
    if not valid_cols or len(valid_cols) < n_components:
        logger.warning(f"Not enough valid job description columns ({len(valid_cols)}) for PCA with {n_components} components.")
        return df_train, df_test, None

    pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
    pca_train = pca.fit_transform(df_train[valid_cols].fillna(0))
    pca_test = pca.transform(df_test[valid_cols].fillna(0))

    # Create DataFrames for PCA components
    pca_cols = [f'job_desc_pca_{i}' for i in range(n_components)]
    pca_train_df = pd.DataFrame(pca_train, index=df_train.index, columns=pca_cols)
    pca_test_df = pd.DataFrame(pca_test, index=df_test.index, columns=pca_cols)
    
    # Use concat instead of direct assignment to avoid fragmentation
    df_train = pd.concat([df_train, pca_train_df], axis=1)
    df_test = pd.concat([df_test, pca_test_df], axis=1)

    # Save the fitted PCA object
    pca_path = config.MODELS_DIR / "job_desc_pca.joblib"
    save_object(pca, pca_path)

    logger.info(f"Applied PCA to job description features, created {n_components} components.")
    return df_train, df_test, pca

def apply_saved_pca(df, job_desc_cols):
    """Applies a saved PCA model to job description features."""
    pca_path = config.MODELS_DIR / "job_desc_pca.joblib"
    valid_cols = [col for col in job_desc_cols if col in df.columns]

    try:
        pca = load_object(pca_path)
        n_components = pca.n_components_
        if not valid_cols or len(valid_cols) < n_components:
             logger.warning(f"Not enough valid job description columns ({len(valid_cols)}) for saved PCA. Skipping PCA.")
             return df

        pca_features = pca.transform(df[valid_cols].fillna(0))
        pca_cols = [f'job_desc_pca_{i}' for i in range(n_components)]
        
        # Use concat to avoid fragmentation
        pca_df = pd.DataFrame(pca_features, index=df.index, columns=pca_cols)
        df = pd.concat([df, pca_df], axis=1)
        
        logger.info(f"Applied saved PCA to job description features ({n_components} components).")
    except FileNotFoundError:
        logger.warning(f"PCA model not found at {pca_path}. Skipping PCA application.")
    except Exception as e:
        logger.error(f"Error applying saved PCA: {e}", exc_info=True)

    return df

def add_polynomial_features(X_train, X_test, degree, interaction_only=False):
    """Adds polynomial features using PolynomialFeatures."""
    logger.info(f"Adding polynomial features (degree={degree}, interaction_only={interaction_only}).")
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)

    # Select only numerical features for polynomial expansion
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    if not numerical_cols:
         logger.warning("No numerical columns found for polynomial features. Skipping.")
         return X_train, X_test, None

    logger.info(f"Applying PolynomialFeatures to {len(numerical_cols)} numerical columns.")

    # Fit on training data
    X_train_poly = poly.fit_transform(X_train[numerical_cols].fillna(0))
    X_test_poly = poly.transform(X_test[numerical_cols].fillna(0))

    # Get new feature names
    poly_feature_names = poly.get_feature_names_out(numerical_cols)

    # Create DataFrames with new feature names
    X_train_poly_df = pd.DataFrame(X_train_poly, index=X_train.index, columns=poly_feature_names)
    X_test_poly_df = pd.DataFrame(X_test_poly, index=X_test.index, columns=poly_feature_names)

    # Drop original numerical columns used in poly and concatenate
    X_train = pd.concat([X_train.drop(columns=numerical_cols), X_train_poly_df], axis=1)
    X_test = pd.concat([X_test.drop(columns=numerical_cols), X_test_poly_df], axis=1)

    # Save the transformer
    poly_path = config.MODELS_DIR / "polynomial_features.joblib"
    save_object(poly, poly_path)
    # Save the list of numerical columns used
    joblib.dump(numerical_cols, config.MODELS_DIR / "polynomial_features_cols.joblib")

    logger.info(f"Added polynomial features. New shape: {X_train.shape}")
    return X_train, X_test, poly

def boolean_to_int(df, columns):
    """Converts boolean-like columns to integers (0/1)."""
    for col in columns:
        if col in df.columns:
            # Handle boolean dtype directly
            if pd.api.types.is_bool_dtype(df[col]):
                 df[col] = df[col].astype(int)
            # Handle object dtype with True/False strings or 0/1
            elif pd.api.types.is_object_dtype(df[col]):
                # Map common boolean representations
                bool_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0, 1: 1, 0: 0, True: 1, False: 0}
                # Apply mapping, coercing errors to NaN, then fill NaNs
                try:
                    df[col] = df[col].astype(str).str.lower().map(bool_map).fillna(0).astype(int)
                except Exception as e:
                    logger.warning(f"Could not convert object column {col} to boolean int: {e}. Filling with 0.")
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            # Handle numeric types that might represent booleans
            elif pd.api.types.is_numeric_dtype(df[col]):
                 df[col] = (df[col] != 0).astype(int)

    logger.info(f"Boolean columns converted to int where possible: {columns}")
    return df 

def apply_umap(df_train, df_test, numerical_cols, n_components=15, n_neighbors=15, min_dist=0.1):
    """Apply UMAP dimensionality reduction to numerical features."""
    if not UMAP_AVAILABLE:
        logger.warning("UMAP not available. Install with 'pip install umap-learn'")
        return df_train, df_test, None
        
    logger.info(f"Applying UMAP with {n_components} components, n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    valid_cols = [col for col in numerical_cols if col in df_train.columns]
    if not valid_cols:
        logger.warning(f"No valid numerical columns for UMAP.")
        return df_train, df_test, None
        
    # Fit UMAP on training data
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=config.RANDOM_SEED
    )
    
    umap_train = reducer.fit_transform(df_train[valid_cols].fillna(0))
    # Transform test data
    umap_test = reducer.transform(df_test[valid_cols].fillna(0))
    
    # Create DataFrames for UMAP components
    umap_cols = [f'umap_{i}' for i in range(n_components)]
    umap_train_df = pd.DataFrame(umap_train, index=df_train.index, columns=umap_cols)
    umap_test_df = pd.DataFrame(umap_test, index=df_test.index, columns=umap_cols)
    
    # Use concat to avoid fragmentation
    df_train = pd.concat([df_train, umap_train_df], axis=1)
    df_test = pd.concat([df_test, umap_test_df], axis=1)
    
    # Save the fitted UMAP object
    umap_path = config.MODELS_DIR / "umap_reducer.joblib"
    save_object(reducer, umap_path)
    
    logger.info(f"Applied UMAP to numerical features, created {n_components} components")
    return df_train, df_test, reducer

def apply_saved_umap(df, numerical_cols):
    """Apply a saved UMAP transformation to data."""
    if not UMAP_AVAILABLE:
        logger.warning("UMAP not available. Install with 'pip install umap-learn'")
        return df
        
    umap_path = config.MODELS_DIR / "umap_reducer.joblib"
    valid_cols = [col for col in numerical_cols if col in df.columns]
    
    try:
        reducer = load_object(umap_path)
        n_components = reducer.n_components
        
        if not valid_cols:
            logger.warning(f"No valid numerical columns for UMAP application.")
            return df
            
        umap_features = reducer.transform(df[valid_cols].fillna(0))
        umap_cols = [f'umap_{i}' for i in range(n_components)]
        
        # Use concat to avoid fragmentation
        umap_df = pd.DataFrame(umap_features, index=df.index, columns=umap_cols)
        df = pd.concat([df, umap_df], axis=1)
        
        logger.info(f"Applied saved UMAP to numerical features ({n_components} components)")
    except FileNotFoundError:
        logger.warning(f"UMAP model not found at {umap_path}. Skipping UMAP application.")
    except Exception as e:
        logger.error(f"Error applying saved UMAP: {e}", exc_info=True)
        
    return df

def create_interaction_features(df, columns, max_interactions=10):
    """Create pairwise interaction features for most important columns."""
    if len(columns) < 2:
        logger.warning("Not enough columns for interaction features.")
        return df
    
    # If too many columns, select top ones to avoid explosion of features
    if len(columns) > max_interactions:
        columns = columns[:max_interactions]
    
    logger.info(f"Creating interaction features for {len(columns)} columns")
    
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            if col1 in df.columns and col2 in df.columns:
                # Create interaction feature
                interaction_name = f"{col1}_{col2}_interact"
                df[interaction_name] = df[col1] * df[col2]
                
    logger.info(f"Created interaction features between top columns")
    return df 