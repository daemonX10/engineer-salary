# Salary Classification Project

This project implements a comprehensive machine learning pipeline for salary classification.

## Project Structure

```
salary_classification/
├── preprocessing/         # Data preprocessing modules
├── models/                # Model training and evaluation modules
├── utils/                 # Utility functions
├── outputs/               # Model outputs and logs
├── notebooks/             # Analysis notebooks
├── data/                  # Data files
├── main.py                # Main entry point
├── config.py              # Configuration settings
└── requirements.txt       # Dependencies
```

## Setup

1. Place the training data (`train.csv`) and test data (`test.csv`) in the `data/` directory.

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Full Pipeline

To run the complete pipeline (preprocessing, training, and prediction):

```
python main.py
```

### Running Specific Steps

You can run specific steps of the pipeline:

- Skip preprocessing: `python main.py --skip-preprocessing`
- Skip training: `python main.py --skip-training`
- Only run prediction: `python main.py --predict-only`
- Train a specific model: `python main.py --model XGBoost`
- Only run ensembling: `python main.py --ensemble-only`

## Project Components

### Preprocessing

- Feature engineering (date features, job title features, job description aggregation)
- Imputation of missing values
- Encoding categorical variables
- Dimensionality reduction with PCA
- Feature scaling

### Models

The pipeline trains multiple models:
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- ExtraTrees
- HistGradient Boosting
- Multi-layer Perceptron (MLP)

### Advanced Techniques

- Hyperparameter optimization with Optuna
- Feature selection
- Stacking ensemble
- Probability calibration

## Output

- Trained models are saved to `outputs/models/`
- Predictions are saved to `outputs/submissions/predictions.csv`
- Logs are saved to `outputs/logs/training.log`
- Preprocessed data is saved to `outputs/preprocessed_data/`

## Configuration

You can adjust various settings in `config.py`:
- Data paths
- Model settings
- Feature engineering options
- Hyperparameter tuning settings 