import pandas as pd
import numpy as np
import logging
import argparse
import os
import time
from pathlib import Path

import config
from utils.logging_config import setup_logging
from utils.helpers import save_object, load_object, save_dataframe, load_dataframe, save_json, load_json
from preprocessing.pipeline import run_preprocessing
from models.train import train_models, train_stacking_model
from models.hyperparameter_tuning import tune_hyperparameters
from models.ensembling import build_stacking_classifier, build_voting_classifier
from models.calibration import calibrate_model
from utils.metrics import evaluate_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Salary Classification Pipeline')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training step')
    parser.add_argument('--model', choices=config.MODELS_TO_TRAIN, help='Train a specific model only')
    parser.add_argument('--ensemble-only', action='store_true', help='Skip base models, only run ensembling')
    parser.add_argument('--predict-only', action='store_true', help='Only run prediction, skip training')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed models from previous run')
    return parser.parse_args()

def load_raw_data():
    """Load raw data from CSV files."""
    logger.info("Loading raw data...")
    train_df = pd.read_csv(config.TRAIN_FILE)
    test_df = pd.read_csv(config.TEST_FILE)
    logger.info(f"Raw data loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

def predict_and_save(model, X_test, test_ids, output_file=config.SUBMISSION_FILE):
    """Make predictions and save them to a CSV file."""
    logger.info("Making predictions...")
    
    try:
        y_pred_proba = model.predict_proba(X_test)
        y_pred_numeric = np.argmax(y_pred_proba, axis=1)
        
        # Convert numeric predictions to original labels
        from preprocessing.encoding import apply_label_encoder
        y_pred = apply_label_encoder(y_pred_numeric, config.LABEL_ENCODER_FILE)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            config.TEST_ID_COLUMN: test_ids,
            config.TARGET_COLUMN: y_pred
        })
        
        # Save submission
        submission_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
        return submission_df
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return None

def main():
    """Main function to run the pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    start_time = time.time()
    logger.info("Starting Salary Classification Pipeline...")
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.PREPROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)
    
    # Step 1: Data Loading and Preprocessing
    if not args.skip_preprocessing and not args.predict_only:
        # Load raw data
        train_df, test_df = load_raw_data()
        
        # Run preprocessing
        logger.info("Running preprocessing...")
        X_train, y_train, X_test = run_preprocessing(train_df, test_df, config)
        
        # Save test IDs for prediction
        test_ids = test_df[config.TEST_ID_COLUMN].values
        save_dataframe(pd.DataFrame({config.TEST_ID_COLUMN: test_ids}), 
                       config.PREPROCESSED_DATA_DIR / "test_ids.parquet")
    else:
        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        try:
            X_train = load_dataframe(config.PREPROCESSED_TRAIN_FILE)
            y_train = load_dataframe(config.PREPROCESSED_DATA_DIR / "y_train.parquet")[config.TARGET_COLUMN].values
            X_test = load_dataframe(config.PREPROCESSED_TEST_FILE)
            test_ids = load_dataframe(config.PREPROCESSED_DATA_DIR / "test_ids.parquet")[config.TEST_ID_COLUMN].values
            logger.info("Preprocessed data loaded successfully.")
        except FileNotFoundError as e:
            logger.error(f"Error: {e}. Please run preprocessing first or check file paths.")
            return
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {str(e)}")
            return
    
    # Step 2: Model Training
    if not args.skip_training and not args.predict_only:
        # Identify failed models from previous runs if retry-failed flag is set
        failed_models = []
        if args.retry_failed:
            try:
                # Check which models are missing from the models directory
                existing_models = [f.stem for f in config.MODELS_DIR.glob("*.joblib")]
                failed_models = [model for model in config.MODELS_TO_TRAIN if model not in existing_models]
                if failed_models:
                    logger.info(f"Will retry training for previously failed models: {failed_models}")
                else:
                    logger.info("No failed models found to retry.")
            except Exception as e:
                logger.warning(f"Error identifying failed models: {str(e)}")
        
        # Hyperparameter tuning if requested
        if args.tune:
            logger.info("Running hyperparameter tuning...")
            # Tune a specific model or all models
            model_to_tune = args.model if args.model else None
            
            # If retry-failed is set, only tune those models
            if args.retry_failed and failed_models:
                model_to_tune = failed_models
                logger.info(f"Tuning only previously failed models: {model_to_tune}")
                
            try:
                best_params = tune_hyperparameters(X_train, y_train, model_name=model_to_tune)
                if best_params:
                    logger.info(f"Best parameters found for {len(best_params)} models")
                    # Save a detailed report of the best parameters
                    save_json(best_params, config.OUTPUT_DIR / "best_params_report.json")
                else:
                    logger.warning("No best parameters found during tuning.")
            except Exception as e:
                logger.error(f"Error during hyperparameter tuning: {str(e)}")
                # Continue with training using default parameters
                logger.info("Continuing with training using default or existing parameters.")
        
        # Train individual models
        if not args.ensemble_only:
            logger.info("Training individual models...")
            models_to_train = None
            
            # Determine which models to train
            if args.model:
                models_to_train = args.model
            elif args.retry_failed and failed_models:
                models_to_train = failed_models
                
            models_info = train_models(X_train, y_train, X_test, 
                                      model_name=models_to_train)
            
            if models_info:
                # Save models info
                models_accuracy = {name: {"accuracy": acc} for name, (_, acc) in models_info.items()}
                save_json(models_accuracy, config.OUTPUT_DIR / "models_accuracy.json")
                
                # Get best model for final prediction if not using ensemble
                if not config.USE_STACKING:
                    try:
                        best_model_name = max(models_info.items(), key=lambda x: x[1][1])[0]
                        best_model = models_info[best_model_name][0]
                        logger.info(f"Best individual model: {best_model_name} with accuracy {models_info[best_model_name][1]:.4f}")
                        
                        # Calibrate if needed
                        if config.USE_CALIBRATION:
                            from sklearn.model_selection import train_test_split
                            # Use a small validation set for calibration
                            X_cal, X_val, y_cal, y_val = train_test_split(X_train, y_train, test_size=0.2, 
                                                                        random_state=config.RANDOM_SEED)
                            calibrated_model = calibrate_model(best_model, X_val, y_val, 
                                                            method=config.CALIBRATION_METHOD, cv='prefit')
                            if calibrated_model:
                                best_model = calibrated_model
                        
                        # Save best model
                        save_object(best_model, config.FINAL_MODEL_FILE)
                    except ValueError as e:
                        logger.error(f"Could not determine best model: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error saving best individual model: {str(e)}")
            else:
                logger.warning("No models were successfully trained.")
                # Check if we can continue with existing models
                if any(config.MODELS_DIR.glob("*.joblib")):
                    logger.info("Found existing trained models in models directory. Will try to use them for ensembling.")
                else:
                    logger.error("No models available for prediction. Exiting.")
                    return
                
        # Train ensemble/stacking model if enabled
        if config.USE_STACKING:
            logger.info("Training stacking ensemble...")
            # Load trained models
            trained_models = {}
            
            # Try to load existing best parameters
            best_params = {}
            try:
                best_params = load_json(config.BEST_PARAMS_FILE)
            except (FileNotFoundError, Exception):
                logger.warning("Could not load best parameters file. Using default parameters.")
            
            for model_name in config.MODELS_TO_TRAIN:
                try:
                    model_path = config.MODELS_DIR / f"{model_name}.joblib"
                    model = load_object(model_path)
                    if model is not None:
                        trained_models[model_name] = model
                        logger.info(f"Successfully loaded {model_name} for stacking")
                    else:
                        logger.warning(f"Model {model_name} loaded as None. Skipping.")
                except (FileNotFoundError, Exception) as e:
                    logger.warning(f"Model {model_name} could not be loaded: {str(e)}. Skipping.")
            
            if len(trained_models) < 2:
                logger.error(f"Not enough trained models found for stacking ({len(trained_models)}). Need at least 2 models.")
                # Try to find any models as fallback
                try:
                    model_files = list(config.MODELS_DIR.glob("*.joblib"))
                    if model_files:
                        # Use most recently created model as fallback
                        best_model_path = sorted(model_files, key=lambda x: os.path.getmtime(x))[-1]
                        logger.info(f"Using most recently trained model as fallback: {best_model_path.name}")
                        final_model = load_object(best_model_path)
                        if final_model is not None:
                            save_object(final_model, config.FINAL_MODEL_FILE)
                            logger.info(f"Saved {best_model_path.name} as final model.")
                        else:
                            raise ValueError("Loaded model is None")
                    else:
                        raise FileNotFoundError("No model files found")
                except Exception as e:
                    logger.error(f"Could not find any usable models: {str(e)}")
                    logger.error("No models available for prediction. Exiting.")
                    return
            else:
                # Train stacking model
                stacking_model = train_stacking_model(trained_models, X_train, y_train, X_test)
                if stacking_model is None:
                    logger.error("Failed to create stacking model. Falling back to best individual model.")
                    # Find best individual model from models_accuracy.json
                    try:
                        models_accuracy = load_json(config.OUTPUT_DIR / "models_accuracy.json")
                        best_model_name = max(models_accuracy.items(), 
                                            key=lambda x: x[1]["accuracy"] if "accuracy" in x[1] else 0)[0]
                        best_model_path = config.MODELS_DIR / f"{best_model_name}.joblib"
                        best_model = load_object(best_model_path)
                        logger.info(f"Using best individual model as fallback: {best_model_name}")
                        save_object(best_model, config.FINAL_MODEL_FILE)
                    except Exception as e:
                        logger.error(f"Error falling back to best individual model: {str(e)}")
                        # Try to use any available model as a last resort
                        try:
                            model_files = list(config.MODELS_DIR.glob("*.joblib"))
                            if model_files:
                                best_model_path = sorted(model_files, key=lambda x: os.path.getmtime(x))[-1]
                                best_model = load_object(best_model_path)
                                logger.info(f"Using {best_model_path.name} as final model (last resort).")
                                save_object(best_model, config.FINAL_MODEL_FILE)
                            else:
                                raise FileNotFoundError("No model files found")
                        except Exception as e2:
                            logger.error(f"Failed to find any usable model: {str(e2)}")
                            return
                else:
                    save_object(stacking_model, config.FINAL_MODEL_FILE)
                    logger.info("Stacking model saved as final model.")
                    
                    # Add a voting ensemble model if configured
                    if config.USE_VOTING and len(trained_models) >= 3:
                        logger.info("Building voting ensemble model...")
                        
                        # Get the top performing models from trained_models
                        # For this we need the accuracy scores - loaded from json file
                        try:
                            model_scores = load_json(config.OUTPUT_DIR / "models_accuracy.json")
                            # Sort models by accuracy
                            sorted_models = []
                            for name in trained_models:
                                if name in model_scores and 'accuracy' in model_scores[name]:
                                    sorted_models.append((name, trained_models[name], model_scores[name]['accuracy']))
                            
                            # Sort by accuracy descending
                            sorted_models.sort(key=lambda x: x[2], reverse=True)
                            
                            # Take top 3-5 models for voting (name, model) tuples
                            top_models = [(name, model) for name, model, _ in sorted_models[:min(5, len(sorted_models))]]
                            
                            if len(top_models) >= 3:
                                # Build voting classifier
                                voting_model = build_voting_classifier(
                                    estimators=top_models,
                                    voting='soft'  # Use probability estimates for final decision
                                )
                                
                                if voting_model is not None:
                                    # Train the voting model
                                    try:
                                        voting_model.fit(X_train, y_train)
                                        
                                        # Save voting model
                                        save_object(voting_model, config.MODELS_DIR / "voting_model.joblib")
                                        
                                        # Save a copy as final model if specified in config
                                        if getattr(config, 'USE_VOTING_AS_FINAL', False):
                                            save_object(voting_model, config.FINAL_MODEL_FILE)
                                            logger.info("Voting ensemble model saved as final model.")
                                    except Exception as e:
                                        logger.error(f"Error training voting model: {str(e)}")
                                else:
                                    logger.error("Failed to build voting classifier")
                            else:
                                logger.warning(f"Not enough models with scores for voting ensemble ({len(top_models)}). Need at least 3.")
                        except Exception as e:
                            logger.error(f"Error building voting ensemble: {e}")
    
    # Step 3: Final Prediction
    # Load final model
    try:
        final_model = load_object(config.FINAL_MODEL_FILE)
        if final_model is None:
            raise ValueError("Loaded final model is None")
        logger.info("Final model loaded successfully.")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"{e}. Please train models first.")
        return
    except Exception as e:
        logger.error(f"Error loading final model: {str(e)}")
        return
    
    # Make predictions and save submission
    submission = predict_and_save(final_model, X_test, test_ids)
    if submission is None:
        logger.error("Failed to generate predictions.")
        return
    
    # Log completion time
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).")

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    logger = logging.getLogger("main")
    
    # Run main function
    main() 