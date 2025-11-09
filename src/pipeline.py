#!/usr/bin/env python3
"""
Generic ChEMBL activity prediction pipeline.

Usage:
    python pipeline.py CHEMBL1075091
    python pipeline.py CHEMBL203

Behavior:
- If data/{ID}_activities.db exists, load from there (table: activities)
- Else, fetch activities from ChEMBL API and save to that DB
- Apply generic filtering (nM, "=", IC50/EC50/Ki)
- Label to active/inactive via pIC50 threshold
- Featurize SMILES → Morgan fingerprints
- Scaffold split (train/test)
- Tanimoto overlap check (train vs test)
- 5-fold CV grid search for 3 model families
- Evaluate on test
- Save models to models/, metrics to results/

This is target-agnostic. The JSON from inspect.py was just to *inspect*,
not to hardcode target-specific rules.
"""

import sys
import os
import json
import sqlite3
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from chembl_webresource_client.new_client import new_client

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

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


# global-ish config (not target-specific)
RANDOM_STATE = 42
THRESHOLD_PIC50 = 6.0  # pIC50 >= 6 → active
ALLOWED_TYPES = {"IC50", "EC50", "Ki"}  # generic, common potency types
ALLOWED_UNITS = {"nM"}                  # stay consistent
ALLOWED_REL = {"="}                     # avoid ">" and "<"

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# -------------- data loading --------------
def load_from_sqlite(db_path: str, table: str = "activities") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def fetch_from_chembl(target_id: str) -> pd.DataFrame:
    acts = new_client.activity.filter(target_chembl_id=target_id)
    df = pd.DataFrame(acts)
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: str, table: str = "activities"):
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()

import json  # you probably already have this at the top

def detect_dominant_activity_type(target_id: str):
    """
    Look for data/{target_id}_summary.json (made by inspect_chembl.py),
    and if it exists, return the most common activity type among IC50/EC50/Ki.
    If the file doesn't exist or nothing matches, return None.
    """
    summary_path = f"data/{target_id}_summary.json"
    dominant_type = None

    if not os.path.exists(summary_path):
        # no summary = nothing to auto-select
        return None

    with open(summary_path, "r") as f:
        summary = json.load(f)

    counts = summary.get("activity_types", {})
    allowed_types = ["IC50", "EC50", "Ki"]

    best_type = None
    best_count = 0
    for t in allowed_types:
        if t in counts and counts[t] > best_count:
            best_type = t
            best_count = counts[t]

    if best_type:
        print(f"Auto-selected dominant activity type from JSON: {best_type}")
    return best_type


# -------------- cleaning / filtering --------------
def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    # keep only columns we care about
    keep_cols = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "assay_description",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # drop missing values
    df = df.dropna(subset=["standard_value", "canonical_smiles"])

    # numeric
    df = df[pd.to_numeric(df["standard_value"], errors="coerce").notnull()].copy()
    df["standard_value"] = df["standard_value"].astype(float)

    # generic filters (NOT target-specific)
    if "standard_units" in df.columns:
        df = df[df["standard_units"].isin(ALLOWED_UNITS)]
    if "standard_relation" in df.columns:
        df = df[df["standard_relation"].isin(ALLOWED_REL)]
    if "standard_type" in df.columns:
        df = df[df["standard_type"].isin(ALLOWED_TYPES)]

    # drop absurd values
    df = df[(df["standard_value"] > 0) & (df["standard_value"] <= 1e7)]

    return df.reset_index(drop=True)


def add_labels(df: pd.DataFrame, threshold_pIC50: float = THRESHOLD_PIC50) -> pd.DataFrame:
    df = df.copy()
    # IC50/EC50/Ki are in nM → pIC50 = 9 - log10(nM)
    df["pIC50"] = 9 - np.log10(df["standard_value"])
    df["activity"] = (df["pIC50"] >= threshold_pIC50).astype(int)
    return df


def sanity_check_labels(df: pd.DataFrame):
    vc = df["activity"].value_counts()
    print("Class counts:\n", vc)
    if len(vc) < 2:
        raise SystemExit("Only one class present after filtering; try different filters/threshold.")
    if vc.min() < 20:
        print("WARNING: one class has <20 samples; model may be unstable.")
    return vc


# -------------- featurization --------------
def smiles_to_morgan(smiles: str, n_bits: int = 2048, radius: int = 2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize(df: pd.DataFrame):
    X, y, smiles = [], [], []
    for _, row in df.iterrows():
        fp = smiles_to_morgan(row["canonical_smiles"])
        if fp is None:
            continue
        X.append(fp)
        y.append(row["activity"])
        smiles.append(row["canonical_smiles"])
    return np.array(X), np.array(y), smiles


# -------------- scaffold split --------------
def get_scaffold(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf)


def scaffold_split_indices(smiles_list, test_size=0.2, random_state=42):
    from collections import defaultdict
    scaf_to_idx = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaf = get_scaffold(smi)
        if scaf is None:
            scaf = f"NOSCAF_{i}"
        scaf_to_idx[scaf].append(i)

    rng = random.Random(random_state)
    scaffolds = list(scaf_to_idx.keys())
    rng.shuffle(scaffolds)

    n_total = len(smiles_list)
    n_test_target = int(n_total * test_size)

    test_idx, train_idx = [], []
    for scaf in scaffolds:
        idxs = scaf_to_idx[scaf]
        if len(test_idx) + len(idxs) <= n_test_target:
            test_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    return np.array(train_idx), np.array(test_idx)


# -------------- tanimoto check --------------
def smiles_list_to_fps(smiles_list, radius=2, n_bits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        fps.append(fp)
    return fps


def summarize_train_test_tanimoto(train_fps, test_fps, thresholds=(0.6, 0.7, 0.8)):
    max_sims = []
    for tfp in test_fps:
        if tfp is None:
            max_sims.append(0.0)
            continue
        best = 0.0
        for trfp in train_fps:
            if trfp is None:
                continue
            sim = DataStructs.TanimotoSimilarity(tfp, trfp)
            if sim > best:
                best = sim
        max_sims.append(best)

    max_sims = np.array(max_sims)
    summary = {
        "mean_max_sim": float(max_sims.mean()),
        "median_max_sim": float(np.median(max_sims)),
    }
    for th in thresholds:
        summary[f"pct_test_ge_{th}"] = float((max_sims >= th).mean()) * 100.0
    return summary, max_sims


# -------------- model configs --------------
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
            "model": RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
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


def majority_baseline(y_train, y_test):
    vals, counts = np.unique(y_train, return_counts=True)
    maj = vals[np.argmax(counts)]
    acc = float(np.mean(y_test == maj))
    pos_rate = float(np.mean(y_test == 1))
    return {
        "majority_label": int(maj),
        "test_accuracy": acc,
        "test_positive_rate": pos_rate,
    }


# -------------- main --------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py CHEMBL_ID")
        sys.exit(1)

    target_id = sys.argv[1].strip().upper()
    db_path = f"data/{target_id}_activities.db"

    # 1) get data (from local DB if present, else from ChEMBL)
    if os.path.exists(db_path):
        print(f"Loading data from {db_path} ...")
        df_raw = load_from_sqlite(db_path)
    else:
        print(f"Fetching data from ChEMBL for {target_id} ...")
        df_raw = fetch_from_chembl(target_id)
        save_to_sqlite(df_raw, db_path)
        print(f"Saved raw activities to {db_path}")

    print("Rows fetched:", len(df_raw))

    # 2) see if we have a summary JSON → auto-pick a dominant assay type
    dominant_type = detect_dominant_activity_type(target_id)

    # 3) generic filtering
    df = clean_and_filter(df_raw)

    # 4) if we found a dominant type (e.g. EC50), keep only that
    if dominant_type and "standard_type" in df.columns:
        df = df[df["standard_type"] == dominant_type]

    print("Rows after filtering:", len(df))

    if df.empty:
        raise SystemExit("No rows left after filtering. Try relaxing filters or checking the target.")

    # 5) label
    df = add_labels(df)
    sanity_check_labels(df)

    # 6) featurize
    X, y, smiles = featurize(df)
    print("Feature matrix:", X.shape)

    # 7) scaffold split
    train_idx, test_idx = scaffold_split_indices(
        smiles, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_smiles = [smiles[i] for i in train_idx]
    test_smiles = [smiles[i] for i in test_idx]

    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

    # 8) tanimoto check
    train_fps = smiles_list_to_fps(train_smiles)
    test_fps = smiles_list_to_fps(test_smiles)
    tanimoto_summary, _ = summarize_train_test_tanimoto(train_fps, test_fps)
    print("\n=== Train–Test Tanimoto Summary ===")
    for k, v in tanimoto_summary.items():
        print(f"{k}: {v:.2f}")

    # 9) baseline
    baseline = majority_baseline(y_train, y_test)
    print("\nMajority baseline:", baseline)

    # 10) model training
    model_configs = get_model_configs()
    scorer = make_scorer(roc_auc_score, needs_proba=True)
    metrics = {
        "target_id": target_id,
        "tanimoto_summary": tanimoto_summary,
        "baseline": baseline,
        "models": {},
    }
    best_name = None
    best_auc = -1.0

    for name, cfg in model_configs.items():
        print(f"\n===== Training {name} (5-fold CV) =====")
        grid = GridSearchCV(
            estimator=cfg["model"],
            param_grid=cfg["params"],
            scoring=scorer,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        cv_auc = float(grid.best_score_)
        print("Best CV ROC-AUC:", cv_auc)
        print("Best params:", grid.best_params_)

        best_est = grid.best_estimator_
        y_proba = best_est.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        test_auc = float(roc_auc_score(y_test, y_proba))
        test_pr = float(average_precision_score(y_test, y_proba))

        print(f"=== {name} on test ===")
        print("ROC-AUC:", round(test_auc, 3))
        print("PR-AUC:", round(test_pr, 3))
        print(classification_report(y_test, y_pred, digits=3))

        model_path = f"models/{target_id}_{name}_best.pkl"
        joblib.dump(best_est, model_path)

        metrics["models"][name] = {
            "cv_roc_auc": cv_auc,
            "test_roc_auc": test_auc,
            "test_pr_auc": test_pr,
            "best_params": grid.best_params_,
            "model_path": model_path,
        }

        if test_auc > best_auc:
            best_auc = test_auc
            best_name = name

    # 11) save best model path + metrics
    if best_name is not None:
        best_model_path = metrics["models"][best_name]["model_path"]
        metrics["best_model"] = {
            "name": best_name,
            "test_roc_auc": best_auc,
            "model_path": best_model_path,
        }
        with open("results/best_model.txt", "w") as f:
            f.write(best_model_path)

    metrics_path = f"results/{target_id}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nBest model:", metrics.get("best_model"))
    print(f"Metrics saved to {metrics_path}")
    print("Done.")
