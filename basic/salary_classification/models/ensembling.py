import logging
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import StratifiedKFold
import config

# Import lightgbm conditionally
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)

def build_stacking_classifier(base_estimators, meta_learner=None, cv_splits=config.N_SPLITS_CV):
    """Builds a StackingClassifier."""
    logger.info("Building Stacking Classifier...")

    if not base_estimators:
        logger.error("No base estimators provided for StackingClassifier.")
        return None

    # Ensure base estimators are provided as (name, estimator) tuples
    if not all(isinstance(item, tuple) and len(item) == 2 for item in base_estimators):
         logger.error("base_estimators must be a list of (name, estimator) tuples.")
         # Attempt to create names if only estimators were passed
         if all(hasattr(est, 'get_params') for est in base_estimators):
              base_estimators = [(f'model_{i}', est) for i, est in enumerate(base_estimators)]
              logger.warning("Automatically generated names for base estimators.")
         else:
              return None
    
    # Validate each estimator to make sure it has fit/predict/predict_proba methods
    valid_estimators = []
    for name, estimator in base_estimators:
        if (hasattr(estimator, 'fit') and 
            hasattr(estimator, 'predict') and 
            hasattr(estimator, 'predict_proba')):
            valid_estimators.append((name, estimator))
        else:
            logger.warning(f"Estimator {name} missing required methods and will be excluded from stacking")
    
    if len(valid_estimators) < 2:
        logger.error(f"Not enough valid estimators for stacking. Need at least 2, but found {len(valid_estimators)}.")
        return None
    
    # Use the valid estimators
    base_estimators = valid_estimators

    if meta_learner is None:
        # Use a robust default meta-learner
        if LIGHTGBM_AVAILABLE:
            meta_learner = lgb.LGBMClassifier(
                random_state=config.RANDOM_SEED, 
                n_estimators=100, 
                learning_rate=0.05, 
                num_leaves=15,
                force_col_wise=True,
                verbose=-1,
                objective='multiclass'
            )
        else:
            # Fallback to LogisticRegression if LightGBM not available
            meta_learner = LogisticRegression(
                random_state=config.RANDOM_SEED,
                max_iter=1000,
                C=1.0,
                multi_class='multinomial',
                solver='lbfgs'
            )
        logger.info(f"Using default meta-learner: {meta_learner.__class__.__name__}")

    # Define the cross-validation strategy for generating level-one predictions
    cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=config.RANDOM_SEED)

    try:
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=cv_strategy,
            stack_method='predict_proba', # Use probabilities for meta-learner input
            n_jobs=-1, # Use multiple cores for fitting base estimators
            passthrough=False # Set to True to include original features
        )
        logger.info("StackingClassifier built successfully.")
        return stacking_clf
    except Exception as e:
        logger.error(f"Error building StackingClassifier: {str(e)}")
        return None

def build_voting_classifier(estimators, voting='soft', weights=None):
    """Builds a VotingClassifier."""
    logger.info(f"Building Voting Classifier (voting='{voting}')...")

    if not estimators:
        logger.error("No estimators provided for VotingClassifier.")
        return None

    # Ensure estimators are provided as (name, estimator) tuples
    if not all(isinstance(item, tuple) and len(item) == 2 for item in estimators):
         logger.error("Estimators must be a list of (name, estimator) tuples.")
         # Attempt to create names if only estimators were passed
         if all(hasattr(est, 'get_params') for est in estimators):
              estimators = [(f'model_{i}', est) for i, est in enumerate(estimators)]
              logger.warning("Automatically generated names for estimators.")
         else:
              return None
    
    # Validate each estimator
    valid_estimators = []
    for name, estimator in estimators:
        required_methods = ['fit', 'predict']
        if voting == 'soft':
            required_methods.append('predict_proba')
            
        if all(hasattr(estimator, method) for method in required_methods):
            valid_estimators.append((name, estimator))
        else:
            logger.warning(f"Estimator {name} missing required methods for {voting} voting and will be excluded")
    
    if len(valid_estimators) < 2:
        logger.error(f"Not enough valid estimators for voting. Need at least 2, but found {len(valid_estimators)}.")
        return None
    
    # Use the valid estimators
    estimators = valid_estimators
    
    # If weights provided, ensure they match the number of estimators
    if weights is not None and len(weights) != len(estimators):
        logger.warning(f"Number of weights ({len(weights)}) doesn't match number of estimators ({len(estimators)}). Ignoring weights.")
        weights = None

    try:
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting, # 'soft' (recommended) or 'hard'
            weights=weights, # Optional weights for each model
            n_jobs=-1 # Use multiple cores
        )
        logger.info("VotingClassifier built successfully.")
        return voting_clf
    except Exception as e:
        logger.error(f"Error building VotingClassifier: {str(e)}")
        return None 