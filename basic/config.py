from pathlib import Path
import pandas as pd
import numpy as np
import os

# Paths
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
LOGS_DIR = PROJECT_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data files
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = REPORTS_DIR / "submission.csv"

# Feature engineering
TARGET_COLUMN = "salary_bucket"
ID_COLUMN = "id"
TEXT_COLUMNS = ["job_title", "company_name"]
CAT_COLUMNS = [
    "job_family", "job_title", "company_name", "level", "education", 
    "age_bucket", "years_experience_bucket", "country", "state", 
    "industry", "job_category"
]
NUM_COLUMNS = ["base_pay", "bonus_percentage", "stock_value", "total_pay"]
ORDINAL_COLUMNS = ["level", "education", "age_bucket", "years_experience_bucket"]
ORDINAL_MAPPINGS = {
    "level": {
        "Entry": 1, 
        "Junior": 2, 
        "Mid": 3, 
        "Senior": 4, 
        "Executive": 5
    },
    "education": {
        "High School": 1, 
        "Associates": 2, 
        "Bachelors": 3, 
        "Masters": 4, 
        "PhD": 5, 
        "Professional": 5
    },
    "age_bucket": {
        "20-29": 1, 
        "30-39": 2, 
        "40-49": 3, 
        "50-59": 4, 
        "60+": 5
    },
    "years_experience_bucket": {
        "0-2": 1, 
        "3-5": 2, 
        "6-10": 3, 
        "11-15": 4, 
        "16-20": 5, 
        "21+": 6
    }
}

# Feature importance threshold for feature selection
FEATURE_IMPORTANCE_THRESHOLD = 0.001

# Model configurations
RANDOM_SEED = 42
N_SPLITS_CV = 5
TEST_SIZE = 0.2
MODELS_TO_TRAIN = [
    "RandomForest",
    "GradientBoosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "ExtraTrees",
    "HistGradientBoosting",
    "MLP",
    "TabNet"
]

# Hyperparameters for different models
# These are just example defaults, should be tuned through hyperparameter optimization
MODEL_PARAMS = {
    "RandomForest": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
        "n_jobs": -1
    },
    "GradientBoosting": {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "random_state": RANDOM_SEED
    },
    "XGBoost": {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "n_jobs": -1
    },
    "LightGBM": {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "n_jobs": -1
    },
    "CatBoost": {
        "iterations": 200,
        "depth": 8,
        "learning_rate": 0.1,
        "random_seed": RANDOM_SEED,
        "verbose": 0
    },
    "ExtraTrees": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
        "n_jobs": -1
    },
    "HistGradientBoosting": {
        "max_iter": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "min_samples_leaf": 20,
        "random_state": RANDOM_SEED
    },
    "MLP": {
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": "auto",
        "learning_rate": "adaptive",
        "max_iter": 500,
        "random_state": RANDOM_SEED
    },
    "TabNet": {
        "n_d": 64,  # Width of the decision prediction layer
        "n_a": 64,  # Width of the attention embedding
        "n_steps": 5,  # Number of successive steps in the network
        "gamma": 1.5,  # Coefficient for feature reusage
        "n_independent": 2,  # Number of independent GLU layers
        "n_shared": 2,  # Number of shared GLU layers
        "lambda_sparse": 0.001,  # Sparsity coefficient
        "momentum": 0.3,  # Momentum for batch normalization
        "random_state": RANDOM_SEED  # Random seed
    }
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "salary_classification.log"
LOG_TO_CONSOLE = True

# Visualization settings
PLOT_FIGSIZE = (12, 8)
PLOT_STYLE = "ggplot"
PLOT_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
               '#1abc9c', '#d35400', '#34495e', '#c0392b', '#7f8c8d'] 