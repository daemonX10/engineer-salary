import numpy as np
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from salary_classification.models.preprocessing import preprocess_for_tabnet
from salary_classification.models.base_model import BaseModel
from config import CAT_COLUMNS, NUM_COLUMNS

logger = logging.getLogger(__name__)

class TabNetModel(BaseModel):
    """TabNet model for salary classification."""
    
    def __init__(self, params=None):
        """
        Initialize TabNet model with parameters.
        
        Args:
            params: Dictionary of parameters for TabNet
        """
        super().__init__()
        self.name = "TabNet"
        
        # Default parameters
        default_params = {
            "n_d": 64,              # Width of the decision prediction layer
            "n_a": 64,              # Width of the attention embedding
            "n_steps": 5,           # Number of steps in the architecture
            "gamma": 1.5,           # Coefficient for feature reusage
            "n_independent": 2,     # Number of independent GLU layers
            "n_shared": 2,          # Number of shared GLU layers
            "lambda_sparse": 1e-3,  # Sparsity regularization
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=2e-2),
            "scheduler_params": dict(max_lr=0.05, 
                                    steps_per_epoch=100, 
                                    epochs=100),
            "scheduler_fn": torch.optim.lr_scheduler.OneCycleLR,
            "mask_type": "entmax",  # "sparsemax" or "entmax"
            "max_epochs": 100,
            "patience": 10,
            "batch_size": 1024,
            "virtual_batch_size": 128
        }
        
        # Update default parameters with provided parameters
        if params:
            default_params.update(params)
        
        self.params = default_params
        self.model = None
        self.cat_dims = None
        self.cat_idxs = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit TabNet model to training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            self: Trained model
        """
        logger.info(f"Training TabNet model with params: {self.params}")
        
        # Preprocess data for TabNet
        if X_val is not None:
            X_train_processed, X_val_processed, self.cat_dims, self.cat_idxs = preprocess_for_tabnet(
                X_train, X_val, CAT_COLUMNS, NUM_COLUMNS
            )
        else:
            X_train_processed, self.cat_dims, self.cat_idxs = preprocess_for_tabnet(
                X_train, None, CAT_COLUMNS, NUM_COLUMNS
            )
            X_val_processed = None
        
        # Initialize TabNet classifier
        self.model = TabNetClassifier(
            n_d=self.params["n_d"],
            n_a=self.params["n_a"],
            n_steps=self.params["n_steps"],
            gamma=self.params["gamma"],
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=1,  # Embedding dimensions for categorical features
            n_independent=self.params["n_independent"],
            n_shared=self.params["n_shared"],
            lambda_sparse=self.params["lambda_sparse"],
            optimizer_fn=self.params["optimizer_fn"],
            optimizer_params=self.params["optimizer_params"],
            scheduler_params=self.params["scheduler_params"],
            scheduler_fn=self.params["scheduler_fn"],
            mask_type=self.params["mask_type"],
            verbose=1
        )
        
        # Convert to numpy arrays
        X_train_np = X_train_processed.values
        y_train_np = y_train.values
        
        if X_val_processed is not None and y_val is not None:
            X_val_np = X_val_processed.values
            y_val_np = y_val.values
            
            # Fit model with validation data
            self.model.fit(
                X_train=X_train_np, 
                y_train=y_train_np,
                eval_set=[(X_val_np, y_val_np)],
                max_epochs=self.params["max_epochs"],
                patience=self.params["patience"],
                batch_size=self.params["batch_size"],
                virtual_batch_size=self.params["virtual_batch_size"]
            )
        else:
            # Fit model without validation data
            self.model.fit(
                X_train=X_train_np, 
                y_train=y_train_np,
                max_epochs=self.params["max_epochs"],
                patience=self.params["patience"],
                batch_size=self.params["batch_size"],
                virtual_batch_size=self.params["virtual_batch_size"]
            )
            
        logger.info("TabNet model training completed")
        return self
    
    def predict(self, X):
        """
        Make predictions using trained TabNet model.
        
        Args:
            X: Features for prediction
            
        Returns:
            predictions: Model predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Preprocess data
        X_processed, _, _ = preprocess_for_tabnet(X, None, CAT_COLUMNS, NUM_COLUMNS)
        
        # Convert to numpy array
        X_np = X_processed.values
        
        # Make predictions
        predictions = self.model.predict(X_np)
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            probabilities: Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Preprocess data
        X_processed, _, _ = preprocess_for_tabnet(X, None, CAT_COLUMNS, NUM_COLUMNS)
        
        # Convert to numpy array
        X_np = X_processed.values
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X_np)
        return probabilities
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Args:
            X: Features for evaluation
            y: True labels
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='weighted')
        precision = precision_score(y, predictions, average='weighted')
        recall = recall_score(y, predictions, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }
        
        return metrics
    
    def feature_importances(self):
        """
        Get feature importances from TabNet model.
        
        Returns:
            feature_importances: Array of feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        return self.model.feature_importances_ 