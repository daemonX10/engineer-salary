import numpy as np
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
from utils.helpers import save_object, load_object
import config

logger = logging.getLogger(__name__)

def calibrate_model(model, X_cal, y_cal, method='isotonic', cv=5):
    """
    Calibrates a classifier to improve probability estimates.
    
    Args:
        model: Fitted classifier to calibrate
        X_cal: Features for calibration
        y_cal: Target for calibration
        method: 'sigmoid' (Platt scaling) or 'isotonic' (isotonic regression)
        cv: Number of folds for cross-validation or 'prefit' if model is already fitted
        
    Returns:
        Calibrated classifier or None if calibration failed
    """
    logger.info(f"Calibrating model using {method} method with cv={cv}")
    
    if not hasattr(model, 'predict_proba'):
        logger.error("Model does not have predict_proba method. Cannot calibrate.")
        return None
    
    try:
        # Check if we have enough samples for calibration
        min_samples_for_calibration = 200  # Minimum number of samples needed for reliable calibration
        if len(y_cal) < min_samples_for_calibration:
            logger.warning(f"Not enough samples for calibration ({len(y_cal)} < {min_samples_for_calibration}). "
                          "Using original model.")
            return model
        
        # Compute pre-calibration scores for comparison
        try:
            y_prob_pre = model.predict_proba(X_cal)
            pre_log_loss = log_loss(y_cal, y_prob_pre)
            pre_brier = brier_score_loss(y_cal, y_prob_pre[:, 1]) if y_prob_pre.shape[1] == 2 else np.nan
            logger.info(f"Pre-calibration Log Loss: {pre_log_loss:.6f}, Brier Score: {pre_brier:.6f}")
        except Exception as e:
            logger.warning(f"Could not compute pre-calibration scores: {str(e)}")
        
        # Create and fit calibrated classifier
        calibrated_model = CalibratedClassifierCV(
            base_estimator=model if cv != 'prefit' else None,
            method=method,
            cv=cv,
            n_jobs=-1 if cv != 'prefit' else None,
        )
        
        if cv == 'prefit':
            # If prefit, we need to provide the already trained model
            calibrated_model = CalibratedClassifierCV(
                base_estimator=None,  # None because we're using a prefit model
                method=method,
                cv='prefit'  # Using prefit model
            )
            calibrated_model.fit(X_cal, y_cal)
        else:
            # For cross-validation, we fit on the calibration data with the specified cv
            calibrated_model = CalibratedClassifierCV(
                base_estimator=model,
                method=method,
                cv=cv,
                n_jobs=-1
            )
            calibrated_model.fit(X_cal, y_cal)
        
        # Compute post-calibration scores
        try:
            y_prob_post = calibrated_model.predict_proba(X_cal)
            post_log_loss = log_loss(y_cal, y_prob_post)
            post_brier = brier_score_loss(y_cal, y_prob_post[:, 1]) if y_prob_post.shape[1] == 2 else np.nan
            logger.info(f"Post-calibration Log Loss: {post_log_loss:.6f}, Brier Score: {post_brier:.6f}")
            
            # Check if calibration improved the scores
            if post_log_loss >= pre_log_loss:
                logger.warning("Calibration did not improve log loss. Consider using the original model.")
                # We still return the calibrated model as it might perform better on test data
        except Exception as e:
            logger.warning(f"Could not compute post-calibration scores: {str(e)}")
        
        logger.info("Model calibration completed successfully")
        return calibrated_model
    
    except Exception as e:
        logger.error(f"Error during model calibration: {str(e)}")
        logger.warning("Using original uncalibrated model")
        return model

def calibrate_model_old(model, X_train, y_train, method='isotonic', cv='prefit'):
    """
    Calibrates a pre-fitted classifier using CalibratedClassifierCV.

    Args:
        model: The pre-fitted base estimator.
        X_train: Training features for calibration (can be a validation set).
        y_train: Training targets for calibration.
        method: 'isotonic' or 'sigmoid'.
        cv: Cross-validation strategy. 'prefit' assumes model is already trained.
            Alternatively, provide an integer for KFold splits or a CV object.

    Returns:
        A fitted CalibratedClassifierCV instance.
    """
    logger.info(f"Starting model calibration using method='{method}' and cv='{cv}'...")

    if cv != 'prefit':
        logger.warning(f"Using cv={cv} in CalibratedClassifierCV will refit the base model. Ensure this is intended.")
        # If refitting, ensure the base model has appropriate parameters set
        base_estimator = model # Pass the unfitted estimator configuration
    else:
         # If using 'prefit', the model passed should already be fitted.
         base_estimator = model


    calibrated_clf = CalibratedClassifierCV(
        base_estimator=base_estimator, # Pass the model instance
        method=method,
        cv=cv, # Use 'prefit' if model is already trained
        n_jobs=-1,
        ensemble=True # Recommended for prefit=True, averages calibration results
    )

    try:
        # If cv='prefit', fit CalibratedClassifierCV on a hold-out set
        logger.info("Fitting CalibratedClassifierCV...")
        calibrated_clf.fit(X_train, y_train)
        logger.info("Model calibration completed successfully.")

        # Save the calibrated model
        calibrated_model_path = config.MODELS_DIR / f"calibrated_{method}_model.joblib"
        save_object(calibrated_clf, calibrated_model_path)
        logger.info(f"Calibrated model saved to {calibrated_model_path}")

        return calibrated_clf

    except Exception as e:
        logger.error(f"Error during model calibration: {e}", exc_info=True)
        return None # Return None if calibration fails 