# src/pipeline.py

"""
ChEMBL Activity Prediction Pipeline

What this script does:
1. Fetches bioactivity data for a single ChEMBL target (IC50 in nM).
2. Cleans and labels the data (active vs inactive) using pIC50.
3. Featurizes SMILES into Morgan fingerprints (RDKit).
4. Does a scaffold-based train/test split.
5. Runs 5-fold CV + grid search for 3 model types:
      - Logistic Regression
      - Random Forest
      - XGBoost
6. Evaluates each best model on the held-out test set.
7. Saves best models to disk.

Run from project root:
    python -m src.pipeline
"""

import os
import numpy as np
import pandas as pd

from chembl_webresource_client.new_client import new_client

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from collections import defaultdict
import random

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    make_scorer,
)

import joblib


# -----------------------------
# CONFIG
# -----------------------------
TARGET_CHEMBL_ID = "CHEMBL1075091"  # example target (EGFR-ish). Change this to your target.
ACTIVITY_TYPE = "IC50"
UNIT = "nM"
THRESHOLD_PIC50 = 6.0           # pIC50 >= 6 → active
RANDOM_STATE = 42

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


# -----------------------------
# 1. DATA FETCH & CLEAN
# -----------------------------
def fetch_chembl_activity(target_chembl_id: str,
                          activity_type: str = "IC50",
                          unit: str = "nM") -> pd.DataFrame:
    """Download activity rows for a single target from ChEMBL."""
    activities = new_client.activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type=activity_type
    )
    df = pd.DataFrame(activities)

    # Keep only the columns we care about if they exist
    keep_cols = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Filter by units
    df = df[df["standard_units"] == unit]

    # Drop missing SMILES or values
    df = df.dropna(subset=["canonical_smiles", "standard_value"])

    # Ensure numeric
    df = df[pd.to_numeric(df["standard_value"], errors="coerce").notnull()].copy()
    df["standard_value"] = df["standard_value"].astype(float)

    # Basic sanity filter
    df = df[(df["standard_value"] > 0) & (df["standard_value"] <= 1e7)]

    return df.reset_index(drop=True)


def add_labels(df: pd.DataFrame, threshold_pIC50: float = 6.0) -> pd.DataFrame:
    """Add pIC50 column and binary activity label."""
    df = df.copy()
    # pIC50 = 9 - log10(IC50 in nM)
    df["pIC50"] = 9 - np.log10(df["standard_value"])
    df["activity"] = (df["pIC50"] >= threshold_pIC50).astype(int)
    return df


# -----------------------------
# 2. FEATURIZATION
# -----------------------------
def smiles_to_morgan(smiles: str, n_bits: int = 2048, radius: int = 2):
    """Convert SMILES to Morgan fingerprint (numpy array)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize(df: pd.DataFrame):
    X = []
    y = []
    smiles_list = []

    for _, row in df.iterrows():
        fp = smiles_to_morgan(row["canonical_smiles"])
        if fp is None:
            continue
        X.append(fp)
        y.append(row["activity"])
        smiles_list.append(row["canonical_smiles"])

    return np.array(X), np.array(y), smiles_list


# -----------------------------
# 3. SCAFFOLD SPLIT
# -----------------------------
def get_scaffold(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split_indices(smiles_list,
                           test_size: float = 0.2,
                           random_state: int = 42):
    """Group molecules by scaffold and assign whole scaffolds to train/test."""
    scaffold_to_indices = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaf = get_scaffold(smi)
        if scaf is None:
            scaf = f"NOSCAF_{idx}"
        scaffold_to_indices[scaf].append(idx)

    rng = random.Random(random_state)
    scaffolds = list(scaffold_to_indices.keys())
    rng.shuffle(scaffolds)

    n_total = len(smiles_list)
    n_test_target = int(n_total * test_size)

    test_indices = []
    train_indices = []

    for scaf in scaffolds:
        idxs = scaffold_to_indices[scaf]
        if len(test_indices) + len(idxs) <= n_test_target:
            test_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    return np.array(train_indices), np.array(test_indices)


# -----------------------------
# 4. MODEL CONFIGS
# -----------------------------
def get_model_configs():
    return {
        "log_reg": {
            "model": LogisticRegression(max_iter=1000, n_jobs=-1),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_jobs=-1,
                random_state=RANDOM_STATE
            ),
            "params": {
                "n_estimators": [200, 400],
                "max_depth": [None, 20],
                "max_features": ["sqrt", 0.3],
                "class_weight": ["balanced"],
            },
        },
        "xgboost": {
            "model": XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_estimators=200,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "max_depth": [3, 5],
                "learning_rate": [0.1, 0.05],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        },
    }


# -----------------------------
# 5. MAIN PIPELINE
# -----------------------------
def main():
    print(f"Fetching ChEMBL data for target {TARGET_CHEMBL_ID} ...")
    df_raw = fetch_chembl_activity(TARGET_CHEMBL_ID, ACTIVITY_TYPE, UNIT)
    print("Rows fetched:", len(df_raw))
    df_raw.to_csv("data/chembl_raw.csv", index=False)

    print("Labeling data ...")
    df_labeled = add_labels(df_raw, THRESHOLD_PIC50)
    df_labeled.to_csv("data/chembl_labeled.csv", index=False)
    print(df_labeled["activity"].value_counts())

    print("Featurizing ...")
    X, y, smiles = featurize(df_labeled)
    print("Feature matrix shape:", X.shape)

    print("Scaffold splitting ...")
    train_idx, test_idx = scaffold_split_indices(smiles, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

    model_configs = get_model_configs()
    scorer = make_scorer(roc_auc_score, needs_proba=True)
    best_models = {}

    for name, cfg in model_configs.items():
        print(f"\n===== Training {name} with 5-fold CV =====")
        grid = GridSearchCV(
            estimator=cfg["model"],
            param_grid=cfg["params"],
            scoring=scorer,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        print("Best CV ROC-AUC:", grid.best_score_)
        print("Best params:", grid.best_params_)
        best_models[name] = grid

    # evaluate
    for name, grid in best_models.items():
        best_est = grid.best_estimator_
        y_proba = best_est.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)

        print(f"\n=== {name} on scaffold test set ===")
        print("ROC-AUC:", round(auc, 3))
        print("PR-AUC:", round(ap, 3))
        print(classification_report(y_test, y_pred, digits=3))

        # save model
        joblib.dump(best_est, f"models/{name}_best.pkl")

    print("\nDone.")


if __name__ == "__main__":
    main()
# src/pipeline.py

"""
ChEMBL Activity Prediction Pipeline

What this script does:
1. Fetches bioactivity data for a single ChEMBL target (IC50 in nM).
2. Cleans and labels the data (active vs inactive) using pIC50.
3. Featurizes SMILES into Morgan fingerprints (RDKit).
4. Does a scaffold-based train/test split.
5. Runs 5-fold CV + grid search for 3 model types:
      - Logistic Regression
      - Random Forest
      - XGBoost
6. Evaluates each best model on the held-out test set.
7. Saves best models to disk.

Run from project root:
    python -m src.pipeline
"""

import os
import numpy as np
import pandas as pd

from chembl_webresource_client.new_client import new_client

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from collections import defaultdict
import random

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    make_scorer,
)

import joblib


# -----------------------------
# CONFIG
# -----------------------------
TARGET_CHEMBL_ID = "CHEMBL203"  # example target (EGFR-ish). Change this to your target.
ACTIVITY_TYPE = "IC50"
UNIT = "nM"
THRESHOLD_PIC50 = 6.0           # pIC50 >= 6 → active
RANDOM_STATE = 42

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


# -----------------------------
# 1. DATA FETCH & CLEAN
# -----------------------------
def fetch_chembl_activity(target_chembl_id: str,
                          activity_type: str = "IC50",
                          unit: str = "nM") -> pd.DataFrame:
    """Download activity rows for a single target from ChEMBL."""
    activities = new_client.activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type=activity_type
    )
    df = pd.DataFrame(activities)

    # Keep only the columns we care about if they exist
    keep_cols = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Filter by units
    df = df[df["standard_units"] == unit]

    # Drop missing SMILES or values
    df = df.dropna(subset=["canonical_smiles", "standard_value"])

    # Ensure numeric
    df = df[pd.to_numeric(df["standard_value"], errors="coerce").notnull()].copy()
    df["standard_value"] = df["standard_value"].astype(float)

    # Basic sanity filter
    df = df[(df["standard_value"] > 0) & (df["standard_value"] <= 1e7)]

    return df.reset_index(drop=True)


def add_labels(df: pd.DataFrame, threshold_pIC50: float = 6.0) -> pd.DataFrame:
    """Add pIC50 column and binary activity label."""
    df = df.copy()
    # pIC50 = 9 - log10(IC50 in nM)
    df["pIC50"] = 9 - np.log10(df["standard_value"])
    df["activity"] = (df["pIC50"] >= threshold_pIC50).astype(int)
    return df


# -----------------------------
# 2. FEATURIZATION
# -----------------------------
def smiles_to_morgan(smiles: str, n_bits: int = 2048, radius: int = 2):
    """Convert SMILES to Morgan fingerprint (numpy array)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize(df: pd.DataFrame):
    X = []
    y = []
    smiles_list = []

    for _, row in df.iterrows():
        fp = smiles_to_morgan(row["canonical_smiles"])
        if fp is None:
            continue
        X.append(fp)
        y.append(row["activity"])
        smiles_list.append(row["canonical_smiles"])

    return np.array(X), np.array(y), smiles_list


# -----------------------------
# 3. SCAFFOLD SPLIT
# -----------------------------
def get_scaffold(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split_indices(smiles_list,
                           test_size: float = 0.2,
                           random_state: int = 42):
    """Group molecules by scaffold and assign whole scaffolds to train/test."""
    scaffold_to_indices = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaf = get_scaffold(smi)
        if scaf is None:
            scaf = f"NOSCAF_{idx}"
        scaffold_to_indices[scaf].append(idx)

    rng = random.Random(random_state)
    scaffolds = list(scaffold_to_indices.keys())
    rng.shuffle(scaffolds)

    n_total = len(smiles_list)
    n_test_target = int(n_total * test_size)

    test_indices = []
    train_indices = []

    for scaf in scaffolds:
        idxs = scaffold_to_indices[scaf]
        if len(test_indices) + len(idxs) <= n_test_target:
            test_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    return np.array(train_indices), np.array(test_indices)


# -----------------------------
# 4. MODEL CONFIGS
# -----------------------------
def get_model_configs():
    return {
        "log_reg": {
            "model": LogisticRegression(max_iter=1000, n_jobs=-1),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_jobs=-1,
                random_state=RANDOM_STATE
            ),
            "params": {
                "n_estimators": [200, 400],
                "max_depth": [None, 20],
                "max_features": ["sqrt", 0.3],
                "class_weight": ["balanced"],
            },
        },
        "xgboost": {
            "model": XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_estimators=200,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "max_depth": [3, 5],
                "learning_rate": [0.1, 0.05],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        },
    }


# -----------------------------
# 5. MAIN PIPELINE
# -----------------------------
def main():
    print(f"Fetching ChEMBL data for target {TARGET_CHEMBL_ID} ...")
    df_raw = fetch_chembl_activity(TARGET_CHEMBL_ID, ACTIVITY_TYPE, UNIT)
    print("Rows fetched:", len(df_raw))
    df_raw.to_csv("data/chembl_raw.csv", index=False)

    print("Labeling data ...")
    df_labeled = add_labels(df_raw, THRESHOLD_PIC50)
    df_labeled.to_csv("data/chembl_labeled.csv", index=False)
    print(df_labeled["activity"].value_counts())

    print("Featurizing ...")
    X, y, smiles = featurize(df_labeled)
    print("Feature matrix shape:", X.shape)

    print("Scaffold splitting ...")
    train_idx, test_idx = scaffold_split_indices(smiles, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

    model_configs = get_model_configs()
    scorer = make_scorer(roc_auc_score, needs_proba=True)
    best_models = {}

    for name, cfg in model_configs.items():
        print(f"\n===== Training {name} with 5-fold CV =====")
        grid = GridSearchCV(
            estimator=cfg["model"],
            param_grid=cfg["params"],
            scoring=scorer,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        print("Best CV ROC-AUC:", grid.best_score_)
        print("Best params:", grid.best_params_)
        best_models[name] = grid

    # evaluate
    for name, grid in best_models.items():
        best_est = grid.best_estimator_
        y_proba = best_est.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)

        print(f"\n=== {name} on scaffold test set ===")
        print("ROC-AUC:", round(auc, 3))
        print("PR-AUC:", round(ap, 3))
        print(classification_report(y_test, y_pred, digits=3))

        # save model
        joblib.dump(best_est, f"models/{name}_best.pkl")

    print("\nDone.")


if __name__ == "__main__":
    main()
