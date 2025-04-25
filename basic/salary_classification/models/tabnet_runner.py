import argparse
import logging
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from salary_classification.models.tabnet_model import TabNetModel
from salary_classification.utils.logger import setup_logger
from config import DATA_DIR, MODELS_DIR, CAT_COLUMNS, NUM_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)

def train_tabnet_model(args):
    """
    Train and evaluate a TabNet model.
    
    Args:
        args: Command line arguments
    """
    # Setup logging
    log_dir = os.path.join(MODELS_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_file=os.path.join(log_dir, "tabnet_training.log"))
    
    logger.info("Loading and preparing data...")
    
    # Load the data
    data_path = os.path.join(DATA_DIR, args.data_file)
    data = pd.read_csv(data_path)
    
    # Split features and target
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    
    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    
    if args.use_validation:
        val_size = args.val_size / (1 - args.test_size)  # Adjust validation size relative to train set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=args.seed
        )
    else:
        X_val, y_val = None, None
    
    logger.info(f"Training data shape: {X_train.shape}")
    if X_val is not None:
        logger.info(f"Validation data shape: {X_val.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Set up TabNet model parameters
    tabnet_params = {
        "n_d": args.n_d,
        "n_a": args.n_a,
        "n_steps": args.n_steps,
        "gamma": args.gamma,
        "n_independent": args.n_independent,
        "n_shared": args.n_shared,
        "lambda_sparse": args.lambda_sparse,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "virtual_batch_size": args.virtual_batch_size,
    }
    
    # Initialize and train TabNet model
    logger.info("Initializing TabNet model...")
    tabnet_model = TabNetModel(params=tabnet_params)
    
    logger.info("Training TabNet model...")
    tabnet_model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    logger.info("Evaluating model on test data...")
    test_metrics = tabnet_model.evaluate(X_test, y_test)
    
    for metric_name, metric_value in test_metrics.items():
        logger.info(f"Test {metric_name}: {metric_value:.4f}")
    
    # Save model
    if args.save_model:
        model_path = os.path.join(MODELS_DIR, "tabnet_model")
        os.makedirs(model_path, exist_ok=True)
        tabnet_model.model.save_model(os.path.join(model_path, "tabnet_model.zip"))
        logger.info(f"Model saved to {model_path}")
    
    # Get feature importances
    feature_importances = tabnet_model.feature_importances()
    feature_names = X_train.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    logger.info("Top 10 important features:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"{row['Feature']}: {row['Importance']:.4f}")
    
    if args.save_importances:
        importance_path = os.path.join(MODELS_DIR, "tabnet_model", "feature_importances.csv")
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importances saved to {importance_path}")
    
    return tabnet_model, test_metrics

def main():
    """Main function to parse arguments and start training"""
    parser = argparse.ArgumentParser(description="Train a TabNet model for salary classification")
    
    # Data arguments
    parser.add_argument("--data_file", type=str, default="processed_data.csv", 
                        help="Processed data file name in the data directory")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="Test set proportion")
    parser.add_argument("--use_validation", action="store_true", 
                        help="Whether to use a validation set")
    parser.add_argument("--val_size", type=float, default=0.1, 
                        help="Validation set proportion of the entire dataset")
    
    # Model arguments
    parser.add_argument("--n_d", type=int, default=64, 
                        help="Width of the decision prediction layer")
    parser.add_argument("--n_a", type=int, default=64, 
                        help="Width of the attention embedding")
    parser.add_argument("--n_steps", type=int, default=5, 
                        help="Number of steps in the architecture")
    parser.add_argument("--gamma", type=float, default=1.5, 
                        help="Coefficient for feature reusage")
    parser.add_argument("--n_independent", type=int, default=2, 
                        help="Number of independent GLU layers")
    parser.add_argument("--n_shared", type=int, default=2, 
                        help="Number of shared GLU layers")
    parser.add_argument("--lambda_sparse", type=float, default=1e-3, 
                        help="Sparsity regularization")
    
    # Training arguments
    parser.add_argument("--max_epochs", type=int, default=100, 
                        help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Patience for early stopping")
    parser.add_argument("--batch_size", type=int, default=1024, 
                        help="Batch size")
    parser.add_argument("--virtual_batch_size", type=int, default=128, 
                        help="Virtual batch size")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    
    # Output arguments
    parser.add_argument("--save_model", action="store_true", 
                        help="Whether to save the trained model")
    parser.add_argument("--save_importances", action="store_true", 
                        help="Whether to save feature importances")
    
    args = parser.parse_args()
    
    # Train the model
    train_tabnet_model(args)

if __name__ == "__main__":
    main() 