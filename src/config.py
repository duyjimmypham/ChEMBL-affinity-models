"""
Configuration settings for the ChEMBL affinity models pipeline.
"""
import os
from pathlib import Path

# --- Logging ---
LOG_LEVEL = os.getenv("CHEMBL_PIPELINE_LOGLEVEL", "INFO").upper()

# --- Paths ---
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

# Ensure directories exist
for path in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    path.mkdir(exist_ok=True, parents=True)

# --- Data Processing ---
RANDOM_STATE = 42
THRESHOLD_ACTIVE_PIC50 = 6.0   # >= 1 µM considered active
THRESHOLD_INACTIVE_PIC50 = 4.5 # <= 30 µM considered inactive

ALLOWED_TYPES = {"IC50", "EC50", "Ki"}
ALLOWED_UNITS = {"nM"}
ALLOWED_REL = {"="}

# --- Fingerprints ---
FP_N_BITS = 2048
FP_RADIUS = 2

# --- Model Hyperparameters ---
# These configurations are used by the pipeline to instantiate models.
# You can adjust parameters here without modifying the code logic.

MODEL_CONFIGS = {
    "log_reg": {
        "type": "sklearn.linear_model.LogisticRegression",
        "params": {
            "max_iter": 1000,
            "solver": "lbfgs",
        },
        "param_grid": {
            "C": [0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        },
    },
    "random_forest": {
        "type": "sklearn.ensemble.RandomForestClassifier",
        "params": {
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        },
        "param_grid": {
            "n_estimators": [200, 400],
            "max_depth": [None, 20],
            "max_features": ["sqrt", 0.3],
            "class_weight": ["balanced"],
        },
    },
}

# XGBoost is optional; check availability in pipeline or here
XGBOOST_CONFIG = {
    "type": "xgboost.XGBClassifier",
    "params": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_estimators": 200,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "param_grid": {
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
}
