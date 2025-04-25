import logging
from sklearn.neural_network import MLPClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import config

logger = logging.getLogger(__name__)

def get_mlp_classifier(input_dim):
    """Defines the MLPClassifier model."""
    logger.info("Defining MLPClassifier model.")
    # Define a reasonable architecture, could be tuned
    hidden_layer_sizes = (
        max(int(input_dim / 2), 20), # First layer size
        max(int(input_dim / 4), 10), # Second layer size
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0001, # L2 penalty
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=300, # Increase iterations
        shuffle=True,
        random_state=config.RANDOM_SEED,
        tol=1e-4,
        early_stopping=True, # Enable early stopping
        n_iter_no_change=15, # Patience for early stopping
        validation_fraction=0.1,
        verbose=False # Set to True for training progress
    )
    return mlp

def get_tabnet_classifier(input_dim, output_dim, cat_idxs=None, cat_dims=None):
    """Defines the TabNetClassifier model."""
    if cat_idxs is None:
        cat_idxs = []
    if cat_dims is None:
        cat_dims = []

    logger.info("Defining TabNetClassifier model.")
    # TabNet parameters often require tuning
    tabnet_params = dict(
        n_d=16, # Width of the decision prediction layer
        n_a=16, # Width of the attention embedding for each mask
        n_steps=4, # Number of steps in the architecture
        gamma=1.3, # Coefficient for feature reusage penalty
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=2, # Embedding dimension for categorical features
        n_independent=2, # Number of independent Gated Linear Units layers at each step
        n_shared=2, # Number of shared Gated Linear Units layers at each step
        lambda_sparse=1e-3, # Importance of the sparsity loss
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 50, "gamma": 0.9}, # Learning rate scheduler
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax', # Type of mask function ('sparsemax' or 'entmax')
        input_dim=input_dim, # Must be set correctly if not inferred
        output_dim=output_dim, # Must be set correctly if not inferred
        verbose=0, # Verbosity level
        seed=config.RANDOM_SEED
    )

    # Note: device='cuda' if GPU is available and configured
    # Adjust batch_size and virtual_batch_size based on memory
    clf = TabNetClassifier(**tabnet_params)
    return clf

# Helper function to get categorical info for TabNet
def get_categorical_info(X_train, categorical_features):
    """Extracts categorical feature indices and dimensions for TabNet."""
    cat_idxs = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
    cat_dims = []
    if cat_idxs:
        # Calculate cardinality for each categorical feature
        for col in categorical_features:
            if col in X_train.columns:
                cat_dims.append(len(X_train[col].unique()))
        logger.info(f"Categorical information extracted: {len(cat_idxs)} features")
    return cat_idxs, cat_dims 