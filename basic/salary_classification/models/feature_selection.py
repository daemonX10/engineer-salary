import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from utils.helpers import save_object, load_object, save_dataframe
import config

logger = logging.getLogger(__name__)

def select_features(X_train, y_train, X_test, method='SelectFromModel', **kwargs):
    """Performs feature selection using the specified method."""
    logger.info(f"Starting feature selection using {method}...")
    
    if method == 'SelectFromModel':
        return select_features_sfm(X_train, y_train, X_test, **kwargs)
    else:
        logger.warning(f"Feature selection method '{method}' not supported. Using default SelectFromModel.")
        return select_features_sfm(X_train, y_train, X_test, **kwargs)

def select_features_sfm(X_train, y_train, X_test, estimator=None, threshold='median'):
    """Performs feature selection using SelectFromModel."""
    if estimator is None:
        # Use ExtraTrees as default estimator
        estimator = ExtraTreesClassifier(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1)
        logger.info("Using default ExtraTreesClassifier for SelectFromModel.")

    logger.info(f"Starting SelectFromModel with threshold: {threshold}")
    logger.info(f"Initial feature shape: {X_train.shape}")

    selector = SelectFromModel(estimator, threshold=threshold, prefit=False)

    # Fit the selector on training data
    try:
        selector.fit(X_train, y_train)
    except Exception as e:
         logger.error(f"Error fitting SelectFromModel: {e}. Skipping feature selection.", exc_info=True)
         return X_train, X_test, list(X_train.columns), None

    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()
    n_selected = len(selected_features)
    n_original = X_train.shape[1]

    logger.info(f"Selected {n_selected} features out of {n_original}.")

    # Transform the datasets
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Convert back to DataFrame with selected feature names
    X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
    X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)

    # Save the selector and selected features
    save_object(selector, config.MODELS_DIR / "feature_selector.joblib")
    
    # Extract feature importances if available
    if hasattr(selector.estimator_, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': selector.estimator_.feature_importances_
        }).sort_values(by='importance', ascending=False)
        save_dataframe(importances, config.OUTPUT_DIR / "feature_importances.csv")
        logger.info("Feature importances saved.")

    logger.info("Feature selection completed.")
    return X_train_selected_df, X_test_selected_df, selected_features, selector 