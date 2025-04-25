import os
from pathlib import Path

# --- Project Root ---
BASE_DIR = Path(__file__).resolve().parent

# --- Data Paths ---
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

# --- Output Paths ---
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = OUTPUT_DIR / "logs"
MODELS_DIR = OUTPUT_DIR / "models"
PREPROCESSED_DATA_DIR = OUTPUT_DIR / "preprocessed_data"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"

# --- Create Dirs ---
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

# --- Preprocessing ---
TARGET_COLUMN = "salary_category"
TEST_ID_COLUMN = "obs"
DROP_COLS_INITIAL = ['obs'] # Drop ID early
# Feature groups (will be refined during preprocessing)
BOOLEAN_FEATURES = ['feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_10', 'feature_11']
DATE_FEATURE = 'job_posted_date'
JOB_DESC_PREFIX = 'job_desc_'
N_JOB_DESC_FEATURES = 300
JOB_DESC_COLS = [f"{JOB_DESC_PREFIX}{str(i).zfill(3)}" for i in range(1, N_JOB_DESC_FEATURES + 1)]

# --- Model Training ---
RANDOM_SEED = 42
N_SPLITS_CV = 5 # Number of folds for StratifiedKFold
ACCURACY_THRESHOLD = 0.78 # Target accuracy to trigger advanced techniques

# --- Hyperparameter Tuning (Optuna) ---
OPTUNA_N_TRIALS = 40  # Reduced from 100 to speed up tuning
OPTUNA_TIMEOUT = 300  # Reduced from 1200 to 5 minutes per model
FORCE_RETUNE = True   # Set to True to force retuning even if parameters exist
QUICK_TUNING = False  # Set to True for even faster tuning with fewer trials

# Model-specific tuning settings (override defaults)
RandomForest_N_TRIALS = 35
RandomForest_TIMEOUT = 240
XGBoost_N_TRIALS = 55
XGBoost_TIMEOUT = 240
LightGBM_N_TRIALS = 15
LightGBM_TIMEOUT = 240
CatBoost_N_TRIALS = 50
CatBoost_TIMEOUT = 180

# --- Models to Train ---
MODELS_TO_TRAIN = [
    'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM',
    'CatBoost', 'ExtraTrees', 'MLP'
]

# --- Advanced Techniques ---
USE_STACKING = True
USE_VOTING = True # Enable voting ensemble
USE_PCA = True
USE_UMAP = True # Enable UMAP for feature reduction
N_PCA_COMPONENTS = 40 # Increased from 30
N_UMAP_COMPONENTS = 20 # Increased from 15
USE_POLYNOMIAL_FEATURES = True
POLYNOMIAL_DEGREE = 2
USE_CALIBRATION = True
CALIBRATION_METHOD = 'isotonic'

# --- Feature Selection ---
FEATURE_SELECTION_METHOD = 'SHAP' # Changed from 'SelectFromModel' to 'SHAP'
# Threshold for SelectFromModel (e.g., 'median', 'mean', float like 1e-5)
SFM_THRESHOLD = 'median'
# Number of features to select if using SHAP (top N)
SHAP_TOP_N_FEATURES = 500 # Increased from 100 to 500

# --- File Names ---
PREPROCESSED_TRAIN_FILE = PREPROCESSED_DATA_DIR / "train_processed.parquet"
PREPROCESSED_TEST_FILE = PREPROCESSED_DATA_DIR / "test_processed.parquet"
TARGET_ENCODER_FILE = MODELS_DIR / "target_encoder.joblib"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor_pipeline.joblib"
FINAL_MODEL_FILE = MODELS_DIR / "final_model.joblib"
SUBMISSION_FILE = SUBMISSIONS_DIR / "predictions.csv"
LOG_FILE = LOGS_DIR / "training.log"
BEST_PARAMS_FILE = OUTPUT_DIR / "best_params.json"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.csv" 