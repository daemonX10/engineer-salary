import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
import time
import os

# Import models conditionally to handle potential import errors gracefully
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

from models.architectures import get_mlp_classifier, get_tabnet_classifier, get_categorical_info
from models.ensembling import build_stacking_classifier
from utils.helpers import save_object, load_object
from utils.metrics import evaluate_model
import config

logger = logging.getLogger(__name__)

def get_model_instance(model_name, params=None, input_dim=None, output_dim=None):
    """Creates an instance of the specified model with given parameters."""
    if params is None:
        params = {} # Use defaults if no params provided
    
    # Add random_state to all models for reproducibility
    if 'random_state' not in params:
        params['random_state'] = config.RANDOM_SEED
    
    # Add n_jobs only to models that support parallel processing
    n_jobs_models = ['RandomForest', 'ExtraTrees', 'XGBoost', 'LightGBM']
    if 'n_jobs' not in params and model_name in n_jobs_models:
        params['n_jobs'] = -1
    
    logger.debug(f"Creating {model_name} instance with params: {params}")
    
    if model_name == 'RandomForest':
        model = RandomForestClassifier(**params)
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(**params)
    elif model_name == 'XGBoost':
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost is not installed. Please install with 'pip install xgboost'.")
            return None
        
        # Ensure required parameters are set
        if 'eval_metric' not in params:
            params['eval_metric'] = 'mlogloss' # Required for multi-class
        model = xgb.XGBClassifier(**params)
    elif model_name == 'LightGBM':
        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM is not installed. Please install with 'pip install lightgbm'.")
            return None
            
        # Add parameters to suppress warnings
        if 'feature_name' not in params:
            params['feature_name'] = 'auto' 
        if 'verbose' not in params:
            params['verbose'] = -1  # Suppress LightGBM console output
        if 'force_col_wise' not in params:
            params['force_col_wise'] = True  # Avoid overhead of testing
        if 'objective' not in params:
            params['objective'] = 'multiclass'  # Ensure multiclass is set
            
        model = lgb.LGBMClassifier(**params)
    elif model_name == 'CatBoost':
        if not CATBOOST_AVAILABLE:
            logger.error("CatBoost is not installed. Please install with 'pip install catboost'.")
            return None
            
        # Ensure required parameters
        if 'loss_function' not in params:
            params['loss_function'] = 'MultiClass'
        if 'verbose' not in params:
            params['verbose'] = 0  # Suppress output
        
        # Fix temporary directory if one is not provided
        if 'train_dir' not in params:
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix='catboost_')
            params['train_dir'] = temp_dir
            # Make sure the directory exists
            os.makedirs(temp_dir, exist_ok=True)
            
        model = cb.CatBoostClassifier(**params)
    elif model_name == 'ExtraTrees':
        model = ExtraTreesClassifier(**params)
    elif model_name == 'HistGradientBoosting':
        model = HistGradientBoostingClassifier(**params)
    elif model_name == 'MLP':
        if input_dim is not None:
            model = get_mlp_classifier(input_dim)
            # Update with any provided params
            model.set_params(**params)
        else:
            # If input_dim not provided, use default MLPClassifier with params
            model = MLPClassifier(**params)
    elif model_name == 'TabNet':
        if not TABNET_AVAILABLE:
            logger.error("TabNet is not installed. Please install with 'pip install pytorch-tabnet'.")
            return None
            
        if input_dim is not None and output_dim is not None:
            # For TabNet we need both input and output dimensions
            # If categorical features information is provided in params
            cat_idxs = params.pop('cat_idxs', None)
            cat_dims = params.pop('cat_dims', None)
            model = get_tabnet_classifier(input_dim, output_dim, cat_idxs, cat_dims)
            # Remaining params can be set using fit method
        else:
            # Default TabNet without specific dimensions
            model = TabNetClassifier(**params)
            logger.warning("TabNet created without input/output dimensions. Must be inferred during fit.")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

def train_single_model(model_name, X_train, y_train, X_test=None):
    """Trains a single model and returns the fitted model and CV accuracy."""
    logger.info(f"Training {model_name} model...")
    
    # Get number of classes
    output_dim = len(np.unique(y_train))
    
    # Check if model library is available before proceeding
    model_libraries = {
        'XGBoost': XGBOOST_AVAILABLE,
        'LightGBM': LIGHTGBM_AVAILABLE,
        'CatBoost': CATBOOST_AVAILABLE,
        'TabNet': TABNET_AVAILABLE
    }
    
    if model_name in model_libraries and not model_libraries[model_name]:
        logger.error(f"{model_name} library is not available. Skipping model training.")
        return None, 0.0
    
    # Handle TabNet separately
    if model_name == 'TabNet' and TABNET_AVAILABLE:
        # TabNet requires special handling for categorical features
        try:
            # Try to identify categorical features from the dataset
            cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_features:
                cat_idxs, cat_dims = get_categorical_info(X_train, cat_features)
                model = get_model_instance(model_name, input_dim=X_train.shape[1], output_dim=output_dim,
                                          params={'cat_idxs': cat_idxs, 'cat_dims': cat_dims})
                
                # TabNet does not support sklearn's cross_val_score, use custom validation
                X_train_val, X_val, y_train_val, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y_train
                )
                
                # Fit TabNet with early stopping
                model.fit(
                    X_train_val.values, y_train_val,
                    eval_set=[(X_val.values, y_val)],
                    max_epochs=100,
                    patience=10,
                    batch_size=1024
                )
                
                # Evaluate on validation set
                val_preds = model.predict(X_val.values)
                cv_accuracy = accuracy_score(y_val, val_preds)
                logger.info(f"TabNet validation accuracy: {cv_accuracy:.4f}")
            else:
                # No categorical features identified
                model = get_model_instance(model_name, input_dim=X_train.shape[1], output_dim=output_dim)
                if model is None:
                    logger.error(f"Failed to create {model_name} model instance.")
                    return None, 0.0
                    
                # Use the same validation approach
                X_train_val, X_val, y_train_val, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y_train
                )
                
                # Handle potential errors during fitting
                try:
                    model.fit(
                        X_train_val.values, y_train_val,
                        eval_set=[(X_val.values, y_val)],
                        max_epochs=100,
                        patience=10,
                        batch_size=1024
                    )
                    val_preds = model.predict(X_val.values)
                    cv_accuracy = accuracy_score(y_val, val_preds)
                except Exception as e:
                    logger.error(f"Error training TabNet model: {str(e)}")
                    return None, 0.0
        except Exception as e:
            logger.error(f"Error in TabNet training setup: {str(e)}")
            return None, 0.0
    else:
        # Standard sklearn models
        model = get_model_instance(model_name, input_dim=X_train.shape[1])
        if model is None:
            logger.error(f"Failed to create {model_name} model instance.")
            return None, 0.0
        
        # Cross-validation for evaluation
        cv = StratifiedKFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_SEED)
        cv_start_time = time.time()
        
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_time = time.time() - cv_start_time
            cv_accuracy = np.mean(cv_scores)
            
            logger.info(f"{model_name} CV Accuracy: {cv_accuracy:.4f} (std: {np.std(cv_scores):.4f})")
            logger.info(f"{model_name} cross-validation completed in {cv_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during cross-validation for {model_name}: {str(e)}")
            cv_accuracy = 0.0
            cv_time = time.time() - cv_start_time
            logger.warning(f"Cross-validation failed after {cv_time:.2f} seconds.")
        
        # Fit on entire training set
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            logger.info(f"{model_name} final training completed in {train_time:.2f} seconds")
            logger.info(f"{model_name} total training time: {cv_time + train_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error fitting {model_name} on full training data: {str(e)}")
            train_time = time.time() - start_time
            logger.warning(f"Model training failed after {train_time:.2f} seconds.")
            return None, 0.0
    
    # Save the model
    model_path = config.MODELS_DIR / f"{model_name}.joblib"
    try:
        save_object(model, model_path)
        logger.info(f"{model_name} model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving {model_name} model: {str(e)}")
    
    return model, cv_accuracy

def train_models(X_train, y_train, X_test=None, model_name=None):
    """Trains all models or a specific model if specified."""
    logger.info("Starting model training...")
    
    # Dictionary to store models and their CV accuracies
    models_info = {}
    
    # Train only the specified model, or all models if None
    model_names = [model_name] if model_name else config.MODELS_TO_TRAIN
    
    for name in model_names:
        if name not in config.MODELS_TO_TRAIN:
            logger.warning(f"Model {name} not in configured models. Skipping.")
            continue
        
        try:
            logger.info(f"--- Training {name} ---")
            model, cv_accuracy = train_single_model(name, X_train, y_train, X_test)
            if model is not None:
                models_info[name] = (model, cv_accuracy)
            else:
                logger.warning(f"Skipping {name} as model creation failed.")
        except Exception as e:
            logger.error(f"Error training {name}: {e}", exc_info=True)
    
    return models_info

def train_stacking_model(trained_models, X_train, y_train, X_test=None):
    """Trains a stacking ensemble model using pre-trained base models."""
    logger.info("Training stacking ensemble model...")
    
    # Create (name, model) tuples for the StackingClassifier
    base_estimators = [(name, model) for name, model in trained_models.items()]
    
    if not base_estimators:
        logger.error("No base estimators available for stacking. Check training results.")
        return None
    
    # Create stacking classifier
    stacking_model = build_stacking_classifier(base_estimators)
    if stacking_model is None:
        logger.error("Failed to build stacking classifier.")
        return None
    
    # Train the stacking model
    start_time = time.time()
    try:
        stacking_model.fit(X_train, y_train)
        train_time = time.time() - start_time
        logger.info(f"Stacking model training completed in {train_time:.2f} seconds")
        
        # Save the stacking model
        save_object(stacking_model, config.MODELS_DIR / "stacking_model.joblib")
        logger.info("Stacking model saved.")
        
        return stacking_model
    except Exception as e:
        train_time = time.time() - start_time
        logger.error(f"Error training stacking model after {train_time:.2f} seconds: {str(e)}")
        return None 