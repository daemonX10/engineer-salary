import optuna
import logging
import numpy as np
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import pandas as pd

from utils.helpers import save_json, load_json, ensure_json_file_exists
import config
from models.train import get_model_instance

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce verbosity

def create_study_visualization(study, model_name):
    """Creates and saves visualization plots for an Optuna study."""
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
        
        # Create output directory for plots
        plot_dir = config.OUTPUT_DIR / "optuna_plots" / model_name
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot optimization history
        fig = plot_optimization_history(study)
        fig.write_image(str(plot_dir / "optimization_history.png"))
        
        # Plot parameter importances
        if study.trials:
            fig = plot_param_importances(study)
            fig.write_image(str(plot_dir / "param_importances.png"))
        
        # Additional plots if there are enough trials
        if len(study.trials) > 5:
            # Try to create contour plots for some important parameters
            if study.best_params:
                param_names = list(study.best_params.keys())
                if len(param_names) >= 2:
                    for i in range(min(3, len(param_names))):
                        for j in range(i+1, min(4, len(param_names))):
                            try:
                                fig = plot_contour(study, params=[param_names[i], param_names[j]])
                                fig.write_image(str(plot_dir / f"contour_{param_names[i]}_{param_names[j]}.png"))
                            except Exception as e:
                                logger.debug(f"Could not create contour plot for {param_names[i]} vs {param_names[j]}: {e}")
        
        logger.info(f"Optuna visualizations saved to {plot_dir}")
    except ImportError:
        logger.warning("Could not create study visualizations. Required packages not installed.")
    except Exception as e:
        logger.warning(f"Error creating study visualizations: {e}")

def tune_hyperparameters(X_train, y_train, model_name=None):
    """
    Perform hyperparameter tuning for the specified model or all models.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_name: Name of the model to tune (if None, tune all models)
        
    Returns:
        Dictionary of best parameters for each model
    """
    models_to_tune = [model_name] if model_name else config.MODELS_TO_TRAIN
    best_params = {}
    
    # Try to load existing best parameters
    try:
        # Ensure the file exists before trying to load it
        ensure_json_file_exists(config.BEST_PARAMS_FILE)
        best_params = load_json(config.BEST_PARAMS_FILE)
        logger.info(f"Loaded existing best parameters for {len(best_params)} models")
    except Exception as e:
        logger.warning(f"Could not load existing parameters: {str(e)}. Starting fresh.")
        best_params = {}
    
    # For each model, run Optuna to find the best hyperparameters
    for model_name in models_to_tune:
        logger.info(f"Tuning hyperparameters for {model_name}...")
        
        # Skip if already tuned and in best_params, unless forced
        if model_name in best_params and not getattr(config, 'FORCE_RETUNE', False):
            logger.info(f"Using existing parameters for {model_name}. Set FORCE_RETUNE=True to retune.")
            continue
        
        # Create the objective function for this model
        def objective(trial):
            params = suggest_params(trial, model_name)
            
            try:
                # Get model instance with the trial parameters
                model = get_model_instance(model_name, params=params)
                
                # Cross-validation
                cv = StratifiedKFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_SEED)
                
                # Use try-except to catch model training errors
                try:
                    scores = cross_val_score(
                        model, X_train, y_train, 
                        scoring='accuracy', cv=cv, n_jobs=-1,
                        error_score='raise'  # Raise error for debugging
                    )
                    
                    mean_accuracy = np.mean(scores)
                    logger.debug(f"Trial {trial.number} for {model_name} - Score: {mean_accuracy:.5f}")
                    
                    return mean_accuracy
                except Exception as e:
                    logger.warning(f"Trial {trial.number} for {model_name} failed with error: {str(e)}")
                    # Return a very low score to deprioritize this parameter set
                    return 0.0
                    
            except Exception as e:
                logger.warning(f"Could not create model instance for {model_name} with params: {params}")
                logger.warning(f"Error: {str(e)}")
                # Return a very low score to deprioritize this parameter set
                return 0.0
        
        # Create storage for study persistence
        study_dir = config.OUTPUT_DIR / "optuna_studies"
        os.makedirs(study_dir, exist_ok=True)
        storage_name = f"sqlite:///{study_dir}/{model_name}_study.db"
        
        # Create Optuna study with persistence
        try:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=3, n_warmup_steps=3, interval_steps=1
            )
            
            study = optuna.create_study(
                study_name=f"{model_name}_study",
                direction="maximize", 
                pruner=pruner,
                storage=storage_name,
                load_if_exists=True
            )
            
            logger.info(f"Created/loaded study from {storage_name}")
            
            # Add early stopping callback - stop if no improvement in 10 trials
            def early_stopping_callback(study, trial):
                if trial.number > 10:
                    # Get the best value and the previous best value
                    values = [t.value for t in study.trials if t.value is not None]
                    if len(values) > 10:
                        best = max(values)
                        previous_best = max(values[:-10])
                        if best <= previous_best:
                            logger.info(f"Early stopping: No improvement in {model_name} after 10 trials.")
                            study.stop()
        except Exception as e:
            logger.warning(f"Could not create study with persistence: {e}. Using in-memory study.")
            # Fallback to in-memory study
            study = optuna.create_study(
                study_name=f"{model_name}_study",
                direction="maximize", 
                pruner=pruner
            )
            
            # Add early stopping callback (same as above)
            def early_stopping_callback(study, trial):
                if trial.number > 10:
                    values = [t.value for t in study.trials if t.value is not None]
                    if len(values) > 10:
                        best = max(values)
                        previous_best = max(values[:-10])
                        if best <= previous_best:
                            logger.info(f"Early stopping: No improvement in {model_name} after 10 trials.")
                            study.stop()
            
        # Run optimization
        try:
            # Use fewer trials and shorter timeout than default for quicker results
            n_trials = getattr(config, f'{model_name}_N_TRIALS', config.OPTUNA_N_TRIALS)
            timeout = getattr(config, f'{model_name}_TIMEOUT', config.OPTUNA_TIMEOUT)
            
            # Use a fraction of the configured values for a quicker first pass if needed
            if getattr(config, 'QUICK_TUNING', False):
                n_trials = min(n_trials, 20)  # Limit to 20 trials max
                timeout = min(timeout, 300)    # Limit to 5 minutes max
            
            logger.info(f"Running optimization for {model_name} with {n_trials} trials, timeout={timeout}s")
            
            study.optimize(
                objective, 
                n_trials=n_trials, 
                timeout=timeout,
                callbacks=[early_stopping_callback]
            )
            
            # Log results
            if study.best_trial:
                logger.info(f"Best trial for {model_name}: {study.best_trial.number}")
                logger.info(f"Best parameters: {study.best_trial.params}")
                logger.info(f"Best accuracy: {study.best_value:.5f}")
                
                # Store best parameters
                best_params[model_name] = study.best_trial.params
                
                # Save after each model (checkpoint)
                save_json(best_params, config.BEST_PARAMS_FILE)
                logger.info(f"Saved best parameters to {config.BEST_PARAMS_FILE}")
                
                # Create visualizations if possible
                create_study_visualization(study, model_name)
            else:
                logger.warning(f"No best trial found for {model_name}")
        except KeyboardInterrupt:
            logger.warning(f"Optimization for {model_name} interrupted by user.")
            # Try to save what we have so far
            if study.best_trial and study.best_trial.params:
                best_params[model_name] = study.best_trial.params
                # Save the partial results
                save_json(best_params, config.BEST_PARAMS_FILE)
                logger.info(f"Saved partial results for {model_name} to {config.BEST_PARAMS_FILE}")
            raise  # Re-raise to stop execution
        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {str(e)}")
            # Keep previous parameters if they exist
            if model_name in best_params:
                logger.info(f"Keeping previous best parameters for {model_name}")
            else:
                logger.warning(f"No parameters found for {model_name}. Using defaults.")
    
    # Save best parameters (final save in case there were changes)
    try:
        save_json(best_params, config.BEST_PARAMS_FILE)
        logger.info(f"Saved best parameters to {config.BEST_PARAMS_FILE}")
    except Exception as e:
        logger.error(f"Error saving best parameters: {str(e)}")
    
    return best_params

def suggest_params(trial, model_name):
    """Suggest hyperparameters for the given model."""
    if model_name == 'RandomForest':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 10, 30, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'random_state': config.RANDOM_SEED,
            'n_jobs': -1
        }
    
    elif model_name == 'GradientBoosting':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': config.RANDOM_SEED
        }
    
    elif model_name == 'XGBoost':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, log=True),
            'random_state': config.RANDOM_SEED,
            'n_jobs': -1,
            'tree_method': 'hist',  # For faster training
            'eval_metric': 'mlogloss'  # Required for multiclass
        }
    
    elif model_name == 'LightGBM':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0, log=True),
            'random_state': config.RANDOM_SEED,
            'n_jobs': -1,
            'force_col_wise': True,
            'verbose': -1,
            'feature_name': 'auto',
            'objective': 'multiclass'  # Make sure multiclass is set explicitly
        }
    
    elif model_name == 'CatBoost':
        return {
            'iterations': trial.suggest_int('iterations', 100, 1000, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'random_seed': config.RANDOM_SEED,
            'verbose': 0,
            'loss_function': 'MultiClass',
            'task_type': 'CPU'  # Explicitly set task type
        }
    
    elif model_name == 'ExtraTrees':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'random_state': config.RANDOM_SEED,
            'n_jobs': -1
        }
    
    elif model_name == 'MLP':
        return {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [
                (100,), (200,), (100, 50), (200, 100), (300, 150), (100, 50, 25)
            ]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': trial.suggest_categorical('solver', ['adam']),  # Reduced choices to more stable solver
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'max_iter': trial.suggest_int('max_iter', 200, 500),
            'early_stopping': True,
            'random_state': config.RANDOM_SEED
        }
        
    else:
        logger.warning(f"No hyperparameter suggestions defined for model: {model_name}")
        return {} 