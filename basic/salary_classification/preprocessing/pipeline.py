import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging
import joblib

from preprocessing.imputation import impute_numerical, impute_categorical, impute_boolean, impute_job_desc
from preprocessing.encoding import label_encode_target, boolean_to_int, target_encode_feature
from preprocessing.feature_engineering import (
    engineer_date_features, engineer_job_title_features,
    engineer_job_desc_aggregates, apply_pca_job_desc, apply_umap,
    add_polynomial_features, boolean_to_int, create_interaction_features
)
from utils.helpers import save_object, load_object, save_dataframe

logger = logging.getLogger(__name__)

def run_preprocessing(train_df, test_df, config):
    """
    Runs the full preprocessing workflow:
    1. Initial cleaning and type conversion
    2. Feature Engineering
    3. Imputation
    4. Encoding
    5. Fits and applies the preprocessing pipeline
    6. Saves processed data and fitted objects
    """
    logger.info("Starting preprocessing workflow...")

    # --- Target Encoding ---
    if config.TARGET_COLUMN in train_df.columns:
        y_train, label_encoder = label_encode_target(train_df, config.TARGET_COLUMN)
        X_train = train_df.drop(columns=[config.TARGET_COLUMN] + config.DROP_COLS_INITIAL, errors='ignore')
        logger.info(f"Target variable '{config.TARGET_COLUMN}' processed. Label Encoder saved.")
    else:
        logger.error(f"Target column '{config.TARGET_COLUMN}' not found in training data!")
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' missing.")

    X_test = test_df.drop(columns=config.DROP_COLS_INITIAL, errors='ignore')

    # Align columns between train and test
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    logger.info(f"Initial train columns: {len(train_cols)}, Initial test columns: {len(test_cols)}")

    # Keep track of original job desc columns
    original_job_desc_cols = [col for col in config.JOB_DESC_COLS if col in X_train.columns]

    # --- Feature Engineering ---
    logger.info("Starting feature engineering...")
    
    # Date features
    X_train = engineer_date_features(X_train, config.DATE_FEATURE)
    X_test = engineer_date_features(X_test, config.DATE_FEATURE)

    # Job Title features
    X_train = engineer_job_title_features(X_train, 'job_title')
    X_test = engineer_job_title_features(X_test, 'job_title')

    # Boolean features (convert to 0/1)
    X_train = boolean_to_int(X_train, config.BOOLEAN_FEATURES)
    X_test = boolean_to_int(X_test, config.BOOLEAN_FEATURES)

    # Job Description aggregates
    X_train = engineer_job_desc_aggregates(X_train, original_job_desc_cols)
    X_test = engineer_job_desc_aggregates(X_test, original_job_desc_cols)

    # --- Imputation (Before Encoding/Scaling) ---
    logger.info("Starting imputation...")
    
    # Identify column types
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    boolean_cols = [col for col in config.BOOLEAN_FEATURES if col in X_train.columns]

    # Impute numerical (median)
    X_train = impute_numerical(X_train, numerical_cols)
    X_test = impute_numerical(X_test, numerical_cols)

    # Impute categorical ('Unknown')
    X_train = impute_categorical(X_train, categorical_cols, fill_value='Unknown')
    X_test = impute_categorical(X_test, categorical_cols, fill_value='Unknown')

    # Impute booleans (False/0)
    X_train = impute_boolean(X_train, boolean_cols, fill_value=False)
    X_test = impute_boolean(X_test, boolean_cols, fill_value=False)

    # Impute Job Desc original features (0)
    X_train = impute_job_desc(X_train, original_job_desc_cols, fill_value=0)
    X_test = impute_job_desc(X_test, original_job_desc_cols, fill_value=0)

    # --- Dimensionality Reduction ---
    if config.USE_PCA and original_job_desc_cols:
        logger.info("Applying PCA to job description features...")
        X_train, X_test, pca_model = apply_pca_job_desc(X_train, X_test, original_job_desc_cols, config.N_PCA_COMPONENTS)
        if pca_model:
             logger.info(f"PCA applied. Explained variance ratio: {np.sum(pca_model.explained_variance_ratio_):.4f}")
             # Update numerical_cols list
             pca_cols = [f'job_desc_pca_{i}' for i in range(config.N_PCA_COMPONENTS)]
             numerical_cols.extend([col for col in pca_cols if col not in numerical_cols and col in X_train.columns])

    # Drop original job description features after PCA or if not using them
    X_train = X_train.drop(columns=original_job_desc_cols, errors='ignore')
    X_test = X_test.drop(columns=original_job_desc_cols, errors='ignore')
    
    # Update numerical columns - remove dropped job_desc features
    numerical_cols = [col for col in numerical_cols if col in X_train.columns]
    
    # Apply UMAP for further dimensionality reduction on numerical features
    if config.USE_UMAP:
        logger.info("Applying UMAP for dimensionality reduction...")
        # Get updated numerical columns (without job desc features that were dropped)
        num_features_for_umap = X_train.select_dtypes(include=np.number).columns.tolist()
        
        # If there are too many numerical features, use the most important ones
        if len(num_features_for_umap) > 100:
            # Use simple correlation-based feature selection
            corr = pd.DataFrame(y_train, columns=['target']).join(
                X_train[num_features_for_umap]
            ).corr()['target'].abs().sort_values(ascending=False)
            # Get the top 100 features
            num_features_for_umap = corr.iloc[1:101].index.tolist()
            logger.info(f"Selected top {len(num_features_for_umap)} numerical features by correlation for UMAP")
            
        X_train, X_test, umap_model = apply_umap(
            X_train, X_test, 
            num_features_for_umap, 
            n_components=config.N_UMAP_COMPONENTS,
            n_neighbors=30,  # Higher values (30-100) preserve global structure
            min_dist=0.1     # Smaller values create tighter clusters
        )
        
        if umap_model:
            # Update numerical_cols list with new UMAP features
            umap_cols = [f'umap_{i}' for i in range(config.N_UMAP_COMPONENTS)]
            numerical_cols.extend([col for col in umap_cols if col not in numerical_cols and col in X_train.columns])
            logger.info(f"UMAP applied, created {len(umap_cols)} components")
            
    # Create interaction features for top numerical features
    logger.info("Creating interaction features...")
    # Find the most correlated features with the target
    top_numerical = []
    if len(numerical_cols) > 0:
        correlations = pd.DataFrame(y_train, columns=['target']).join(
            X_train[numerical_cols]
        ).corr()['target'].abs().sort_values(ascending=False)
        # Take the top 10 numerical features
        top_numerical = correlations.iloc[1:11].index.tolist()
        
    # Create interaction features
    X_train = create_interaction_features(X_train, top_numerical, max_interactions=10)
    X_test = create_interaction_features(X_test, top_numerical, max_interactions=10)
    logger.info(f"Created interaction features for top {len(top_numerical)} numerical features")
    
    # --- Advanced Feature Engineering (Polynomial) ---
    if config.USE_POLYNOMIAL_FEATURES:
        logger.info("Adding polynomial features...")
        X_train, X_test, poly = add_polynomial_features(
            X_train, X_test, 
            degree=config.POLYNOMIAL_DEGREE, 
            interaction_only=True  # Typically use interaction_only=True to reduce dimensionality
        )

    # --- Preprocessing Pipeline (Scaling, OHE) ---
    # Update column types after feature engineering
    final_numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    final_categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.info(f"Final numerical columns: {len(final_numerical_cols)}")
    logger.info(f"Final categorical columns: {len(final_categorical_cols)}")

    # Align columns between train and test
    train_cols_final = set(X_train.columns)
    test_cols_final = set(X_test.columns)
    missing_in_test = list(train_cols_final - test_cols_final)
    extra_in_test = list(test_cols_final - train_cols_final)

    # Handle missing columns
    if missing_in_test:
        logger.warning(f"Columns missing in test set, adding with zeros: {missing_in_test}")
        for col in missing_in_test:
            X_test[col] = 0
    if extra_in_test:
        logger.warning(f"Extra columns in test set, dropping: {extra_in_test}")
        X_test = X_test.drop(columns=extra_in_test)
    
    # Ensure same column order
    X_test = X_test[X_train.columns]

    # ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), final_numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), final_categorical_cols)
        ],
        remainder='passthrough'
    )

    logger.info("Fitting preprocessing pipeline...")
    X_train_processed = preprocessor.fit_transform(X_train)
    logger.info("Transforming test data...")
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        logger.warning(f"Could not get feature names from preprocessor: {e}. Using generic names.")
        feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]

    # Convert to DataFrame
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

    # Handle potential infinities
    X_train_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Impute any remaining NaNs
    X_train_processed.fillna(0, inplace=True)
    X_test_processed.fillna(0, inplace=True)

    logger.info(f"Preprocessing finished. Final shapes: X_train={X_train_processed.shape}, X_test={X_test_processed.shape}")

    # Save processed data and objects
    logger.info("Saving preprocessed data and pipeline...")
    save_dataframe(X_train_processed, config.PREPROCESSED_TRAIN_FILE)
    save_dataframe(pd.DataFrame({config.TARGET_COLUMN: y_train}), config.PREPROCESSED_DATA_DIR / "y_train.parquet")
    save_dataframe(X_test_processed, config.PREPROCESSED_TEST_FILE)
    save_object(preprocessor, config.PREPROCESSOR_FILE)
    joblib.dump(list(X_train_processed.columns), config.MODELS_DIR / "final_feature_names.joblib")

    logger.info("Preprocessing workflow completed successfully.")
    return X_train_processed, y_train, X_test_processed 