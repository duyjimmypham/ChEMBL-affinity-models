#!/usr/bin/env python3
"""
Train per-target ChEMBL affinity models end-to-end.

Usage:
    python src/pipeline.py CHEMBL203 [--chembl-sqlite path]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV

from chembl_cache import has_remote_updates, load_target_meta, save_target_meta
from chembl_client_utils import fetch_paginated
from local_chembl import fetch_local_activities, find_default_local_db

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


LOG_LEVEL = os.getenv("CHEMBL_PIPELINE_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
LOGGER = logging.getLogger("chembl_pipeline")


RANDOM_STATE = 42
THRESHOLD_ACTIVE_PIC50 = 6.0   # >= 1 µM considered active
THRESHOLD_INACTIVE_PIC50 = 4.5 # <= 30 µM considered inactive
ALLOWED_TYPES = {"IC50", "EC50", "Ki"}
ALLOWED_UNITS = {"nM"}
ALLOWED_REL = {"="}

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

for path in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    path.mkdir(exist_ok=True)


# -------------- data io --------------
def load_from_sqlite(db_path: Path, table: str = "activities") -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def fetch_from_chembl(target_id: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    log = logger or LOGGER
    checkpoint_path = DATA_DIR / f"{target_id}_activities_checkpoint.jsonl"
    rows = fetch_paginated(
        new_client.activity,
        {"target_chembl_id": target_id},
        checkpoint_path=checkpoint_path,
        logger=log,
    )
    return pd.DataFrame(rows)


def save_to_sqlite(df: pd.DataFrame, db_path: Path, table: str = "activities") -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA synchronous = OFF;")
        conn.execute("PRAGMA journal_mode = MEMORY;")
        df.to_sql(table, conn, if_exists="replace", index=False)
    finally:
        conn.close()


def detect_dominant_activity_type(
    target_id: str, logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    Auto-select the dominant activity type using inspect_chembl summary JSON if present.
    """
    log = logger or LOGGER
    summary_path = DATA_DIR / f"{target_id}_summary.json"
    if not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = summary.get("activity_types", {})
    best_type = None
    best_count = 0
    for activity_type in ("IC50", "EC50", "Ki"):
        count = counts.get(activity_type, 0)
        if count > best_count:
            best_type = activity_type
            best_count = count

    if best_type:
        log.info("Auto-selected dominant activity type from JSON: %s", best_type)
    return best_type


# -------------- preprocessing --------------
def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "assay_description",
        "updated_on",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df = df.dropna(subset=["standard_value", "canonical_smiles"])

    df = df[pd.to_numeric(df["standard_value"], errors="coerce").notnull()].copy()
    df["standard_value"] = df["standard_value"].astype(float)

    if "standard_units" in df.columns:
        df = df[df["standard_units"].isin(ALLOWED_UNITS)]
    if "standard_relation" in df.columns:
        df = df[df["standard_relation"].isin(ALLOWED_REL)]
    if "standard_type" in df.columns:
        df = df[df["standard_type"].isin(ALLOWED_TYPES)]

    df = df[(df["standard_value"] > 0) & (df["standard_value"] <= 1e7)]
    return df.reset_index(drop=True)


def compute_last_updated(df: pd.DataFrame) -> Optional[str]:
    if "updated_on" not in df.columns:
        return None
    series = pd.to_datetime(df["updated_on"], errors="coerce", utc=True)
    series = series.dropna()
    if series.empty:
        return None
    return series.max().isoformat()


def compute_p_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate row-level dataframe with p_activity (currently identical to pIC50 for nM data).
    """
    df = df.copy()
    df["p_activity"] = 9 - np.log10(df["standard_value"])
    return df


def aggregate_per_molecule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple activity measurements per molecule into a single row using median p_activity.
    """
    if "molecule_chembl_id" not in df.columns or "canonical_smiles" not in df.columns:
        raise RuntimeError("Cannot aggregate without molecule identifiers and SMILES.")
    grouped = (
        df.groupby(["molecule_chembl_id", "canonical_smiles"], dropna=False)
        .agg(
            p_activity_median=("p_activity", "median"),
            n_measurements=("p_activity", "count"),
            p_activity_std=("p_activity", "std"),
        )
        .reset_index()
    )
    if grouped.empty:
        raise RuntimeError("No molecules available after aggregation.")
    grouped.rename(columns={"p_activity_median": "p_activity"}, inplace=True)
    LOGGER.info("Aggregated to %d unique molecules.", len(grouped))
    return grouped


def add_molecule_level_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign activity labels at molecule level using aggregated p_activity values.
    """
    df = df.copy()

    def classify(p_activity: float) -> Optional[int]:
        if p_activity >= THRESHOLD_ACTIVE_PIC50:
            return 1
        if p_activity <= THRESHOLD_INACTIVE_PIC50:
            return 0
        return None

    df["activity"] = df["p_activity"].apply(classify)
    before = len(df)
    df = df[df["activity"].notna()].copy()
    dropped = before - len(df)
    if dropped > 0:
        LOGGER.info(
            "Dropped %d ambiguous molecules with %.1f < p_activity < %.1f",
            dropped,
            THRESHOLD_INACTIVE_PIC50,
            THRESHOLD_ACTIVE_PIC50,
        )
    if df.empty:
        raise RuntimeError("No molecules remain after applying activity thresholds.")
    df["activity"] = df["activity"].astype(int)
    return df


def add_quantile_level_labels(
    df: pd.DataFrame,
    low_quantile: float = 0.30,
    high_quantile: float = 0.70,
    min_per_class: int = 20,
) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    """
    Fallback labeling using within-target quantiles.
    Returns (labeled_df, {"low": q_low, "high": q_high}).
    """
    if df.empty or "p_activity" not in df.columns:
        return None, {"low": None, "high": None}

    df = df.copy()
    q_low = float(df["p_activity"].quantile(low_quantile))
    q_high = float(df["p_activity"].quantile(high_quantile))
    thresholds = {"low": q_low, "high": q_high}

    if np.isnan(q_low) or np.isnan(q_high) or q_high <= q_low:
        return None, thresholds

    df["activity"] = np.where(
        df["p_activity"] >= q_high,
        1,
        np.where(df["p_activity"] <= q_low, 0, np.nan),
    )
    df = df[df["activity"].notna()].copy()
    if df.empty:
        return None, thresholds

    df["activity"] = df["activity"].astype(int)
    class_counts = df["activity"].value_counts()
    if len(class_counts) < 2 or (class_counts < min_per_class).any():
        return None, thresholds

    return df, thresholds


def sanity_check_labels(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> None:
    log = logger or LOGGER
    vc = df["activity"].value_counts()
    log.info("Class counts:\n%s", vc)
    if vc.min() < 20:
        log.warning("One class has <20 samples; model may be unstable.")


def smiles_to_morgan(smiles: str, n_bits: int = 2048, radius: int = 2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_bitvect(smiles: str, n_bits: int = 2048, radius: int = 2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def featurize(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    fps = []
    smiles = df["canonical_smiles"].tolist()
    for smi in smiles:
        fp = smiles_to_morgan(smi)
        if fp is None:
            raise ValueError(f"Could not generate fingerprint for {smi}")
        fps.append(fp)
    X = np.vstack(fps)
    y = df["activity"].values.astype(int)
    return X, y, smiles


def get_scaffold(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold else None


def scaffold_split_indices(
    smiles_list: Sequence[str], test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray]:
    scaffolds = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaff = get_scaffold(smi) or f"NO_SCAFFOLD_{idx}"
        scaffolds[scaff].append(idx)

    scaffold_items = list(scaffolds.items())
    random.Random(random_state).shuffle(scaffold_items)

    train_idx: List[int] = []
    test_idx: List[int] = []
    total = len(smiles_list)
    test_target = int(total * test_size)

    for scaffold, indices in scaffold_items:
        if len(test_idx) < test_target:
            test_idx.extend(indices)
        else:
            train_idx.extend(indices)

    if not train_idx or not test_idx:
        raise RuntimeError("Scaffold split failed; adjust test_size or data quality.")

    return np.array(train_idx), np.array(test_idx)


def smiles_list_to_fps(smiles_list: Sequence[str], radius: int = 2, n_bits: int = 2048):
    fps = []
    for smi in smiles_list:
        fp = smiles_to_bitvect(smi, n_bits=n_bits, radius=radius)
        fps.append(fp)
    return fps


from rdkit.DataStructs.cDataStructs import ExplicitBitVect
def summarize_train_test_tanimoto(
    train_fps: Sequence[Optional[ExplicitBitVect]],
    test_fps: Sequence[Optional[ExplicitBitVect]],
    thresholds: Sequence[float] = (0.6, 0.7, 0.8),
) -> Tuple[Dict[str, float], np.ndarray]:
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
            best = max(best, sim)
        max_sims.append(best)

    max_sims = np.array(max_sims)
    summary: Dict[str, float] = {
        "mean_max_sim": float(max_sims.mean()),
        "median_max_sim": float(np.median(max_sims)),
    }
    for th in thresholds:
        summary[f"pct_test_ge_{th}"] = float((max_sims >= th).mean()) * 100.0
    return summary, max_sims


def get_model_configs(logger: Optional[logging.Logger] = None) -> Dict[str, Dict[str, object]]:
    configs: Dict[str, Dict[str, object]] = {
        "log_reg": {
            "model": LogisticRegression(max_iter=1000, solver="lbfgs"),
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
    }

    if XGBOOST_AVAILABLE:
        configs["xgboost"] = {
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
        }
    else:
        (logger or LOGGER).warning("XGBoost not available; skipping that model family.")

    return configs


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    vals, counts = np.unique(y_train, return_counts=True)
    maj = vals[np.argmax(counts)]
    acc = float(np.mean(y_test == maj))
    pos_rate = float(np.mean(y_test == 1))
    return {
        "majority_label": int(maj),
        "test_accuracy": acc,
        "test_positive_rate": pos_rate,
    }


# -------------- pipeline orchestration --------------
def run_pipeline(
    target_id: str,
    *,
    force_refresh: bool = False,
    chembl_sqlite: Optional[str] = None,
    skip_update_check: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    log = logger or LOGGER
    target_id = target_id.strip().upper()
    db_path = DATA_DIR / f"{target_id}_activities.db"

    dataset_warnings: List[str] = []
    df_raw: Optional[pd.DataFrame] = None
    if not chembl_sqlite:
        auto_db = find_default_local_db()
        if auto_db:
            log.info("Detected local ChEMBL SQLite at %s", auto_db)
            chembl_sqlite = str(auto_db)

    local_db_path = Path(chembl_sqlite).expanduser() if chembl_sqlite else None

    if local_db_path:
        log.info("Loading activities from local ChEMBL SQLite: %s", local_db_path)
        df_raw = fetch_local_activities(target_id, local_db_path)
        save_to_sqlite(df_raw, db_path)
        save_target_meta(target_id, compute_last_updated(df_raw), len(df_raw))
    else:
        meta = load_target_meta(target_id)
        cache_exists = db_path.exists()
        use_cache = cache_exists and not force_refresh
        needs_refresh = True

        if use_cache and (skip_update_check or meta is None):
            needs_refresh = False
        elif use_cache and meta is not None:
            needs_refresh = has_remote_updates(target_id, meta.get("last_updated"), logger=log)

        if not needs_refresh and cache_exists:
            log.info("Using cached activities from %s", db_path)
            df_raw = load_from_sqlite(db_path)
        else:
            log.info("Fetching activities from ChEMBL for %s", target_id)
            df_raw = fetch_from_chembl(target_id, logger=log)
            save_to_sqlite(df_raw, db_path)
            save_target_meta(target_id, compute_last_updated(df_raw), len(df_raw))
            log.info("Saved raw activities to %s", db_path)

    n_raw_rows = len(df_raw)
    log.info("Rows fetched: %d", n_raw_rows)

    dominant_type = detect_dominant_activity_type(target_id, logger=log)

    df = clean_and_filter(df_raw)
    if dominant_type and "standard_type" in df.columns:
        df = df[df["standard_type"] == dominant_type]

    n_filtered_rows = len(df)
    log.info("Rows after filtering: %d", n_filtered_rows)
    if df.empty:
        raise RuntimeError("No rows left after filtering. Check target or adjust filters.")

    df = compute_p_activity(df)
    df = aggregate_per_molecule(df)
    df_mols = df.copy()
    n_molecules_total = len(df_mols)
    df = add_molecule_level_labels(df)
    sanity_check_labels(df, logger=log)
    labeling_strategy = "absolute"
    quantile_thresholds = None
    class_counts_abs = df["activity"].value_counts().to_dict()
    unique_classes_abs = sorted(class_counts_abs.keys())

    if len(unique_classes_abs) < 2:
        log.warning(
            "Only one activity class after absolute thresholds for %s; attempting quantile fallback.",
            target_id,
        )
        df_fallback, quantile_thresholds = add_quantile_level_labels(df_mols)
        if df_fallback is None:
            log.error(
                "Quantile fallback labeling also failed for %s (class_counts=%s).",
                target_id,
                class_counts_abs,
            )
            metrics = {
                "target_id": target_id,
                "trainable": False,
                "reason": "quantile_failed",
                "class_counts": class_counts_abs,
                "labeling_strategy": "quantile_fallback",
                "quantile_thresholds": quantile_thresholds,
                "n_raw_rows": n_raw_rows,
                "n_filtered_rows": n_filtered_rows,
                "n_molecules_total": n_molecules_total,
                "n_molecules_labeled": len(df),
                "pos_rate": None,
                "dataset_warnings": dataset_warnings,
            }
            metrics_path = RESULTS_DIR / f"{target_id}_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            log.info("Metrics saved to %s", metrics_path)
            return {
                "target_id": target_id,
                "metrics_path": str(metrics_path),
                "metrics": metrics,
                "best_model_path": None,
            }

        df = df_fallback
        labeling_strategy = "quantile_fallback"
        sanity_check_labels(df, logger=log)
        class_counts = df["activity"].value_counts().to_dict()
        unique_classes = sorted(class_counts.keys())
        log.info(
            "Applied quantile fallback (low=%.2f, high=%.2f). Class counts: %s",
            quantile_thresholds["low"],
            quantile_thresholds["high"],
            class_counts,
        )
        if len(unique_classes) < 2:
            log.error(
                "Quantile fallback still resulted in a single class for %s.",
                target_id,
            )
            metrics = {
                "target_id": target_id,
                "trainable": False,
                "reason": "quantile_failed",
                "class_counts": class_counts,
                "labeling_strategy": labeling_strategy,
                "quantile_thresholds": quantile_thresholds,
                "n_raw_rows": n_raw_rows,
                "n_filtered_rows": n_filtered_rows,
                "n_molecules_total": n_molecules_total,
                "n_molecules_labeled": len(df),
                "pos_rate": None,
                "dataset_warnings": dataset_warnings,
            }
            metrics_path = RESULTS_DIR / f"{target_id}_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            log.info("Metrics saved to %s", metrics_path)
            return {
                "target_id": target_id,
                "metrics_path": str(metrics_path),
                "metrics": metrics,
                "best_model_path": None,
            }
    else:
        class_counts = class_counts_abs

    n_molecules_labeled = len(df)
    n_pos = class_counts.get(1, 0)
    n_neg = class_counts.get(0, 0)
    total_labeled = n_pos + n_neg
    pos_rate = (n_pos / total_labeled) if total_labeled > 0 else None

    MIN_MOLECULES = 60
    MIN_PER_CLASS = 25

    def record_warning(flag: str, message: str):
        log.warning(message)
        dataset_warnings.append(flag)

    if n_molecules_labeled < MIN_MOLECULES:
        record_warning(
            "too_few_molecules",
            f"Dataset is very small for target {target_id}: "
            f"n_molecules_labeled={n_molecules_labeled}. Model may not generalize well.",
        )
    if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
        record_warning(
            "too_few_per_class",
            f"One or both classes are very small for target {target_id}: "
            f"n_pos={n_pos}, n_neg={n_neg} (MIN_PER_CLASS={MIN_PER_CLASS}).",
        )
    if pos_rate is not None and (pos_rate < 0.1 or pos_rate > 0.9):
        record_warning(
            "extreme_imbalance",
            f"Extreme class imbalance for target {target_id}: pos_rate={pos_rate:.3f} "
            f"(n_pos={n_pos}, n_neg={n_neg}).",
        )

    log.info(
        "Data summary for %s: molecules_labeled=%d, n_pos=%d, n_neg=%d, pos_rate=%s",
        target_id,
        n_molecules_labeled,
        n_pos,
        n_neg,
        f"{pos_rate:.3f}" if pos_rate is not None else "NA",
    )

    X, y, smiles = featurize(df)
    log.info("Feature matrix shape: %s", X.shape)

    train_idx, test_idx = scaffold_split_indices(smiles, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_smiles = [smiles[i] for i in train_idx]
    test_smiles = [smiles[i] for i in test_idx]
    log.info("Train size: %d | Test size: %d", X_train.shape[0], X_test.shape[0])

    train_fps = smiles_list_to_fps(train_smiles)
    test_fps = smiles_list_to_fps(test_smiles)
    tanimoto_summary, _ = summarize_train_test_tanimoto(train_fps, test_fps)
    log.info("=== Train/Test Tanimoto Summary ===")
    for k, v in tanimoto_summary.items():
        log.info("%s: %.2f", k, v)

    baseline = majority_baseline(y_train, y_test)
    log.info("Majority baseline: %s", baseline)

    model_configs = get_model_configs(logger=log)
    scorer = make_scorer(roc_auc_score, needs_proba=True)
    metrics: Dict[str, object] = {
        "target_id": target_id,
        "tanimoto_summary": tanimoto_summary,
        "baseline": baseline,
        "models": {},
        "labeling_strategy": labeling_strategy,
        "class_counts": class_counts,
        "n_raw_rows": n_raw_rows,
        "n_filtered_rows": n_filtered_rows,
        "n_molecules_total": n_molecules_total,
        "n_molecules_labeled": n_molecules_labeled,
        "pos_rate": pos_rate,
        "dataset_warnings": dataset_warnings,
        "quantile_thresholds": quantile_thresholds,
    }

    best_name = None
    best_auc = -1.0

    for name, cfg in model_configs.items():
        log.info("===== Training %s (5-fold CV) =====", name)
        grid = GridSearchCV(
            estimator=cfg["model"],
            param_grid=cfg["params"],
            scoring=scorer,
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        cv_auc = float(grid.best_score_)
        log.info("%s best CV ROC-AUC: %.3f", name, cv_auc)
        log.info("%s best params: %s", name, grid.best_params_)

        best_est = grid.best_estimator_
        y_proba = best_est.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        test_auc = float(roc_auc_score(y_test, y_proba))
        test_pr = float(average_precision_score(y_test, y_proba))
        report_text = classification_report(y_test, y_pred, digits=3)
        log.info("%s test ROC-AUC: %.3f | PR-AUC: %.3f", name, test_auc, test_pr)
        log.info("%s classification report:\n%s", name, report_text)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)

        model_path = MODELS_DIR / f"{target_id}_{name}_best.pkl"
        joblib.dump(best_est, model_path)

        metrics["models"][name] = {
            "cv_roc_auc": cv_auc,
            "test_roc_auc": test_auc,
            "test_pr_auc": test_pr,
            "best_params": grid.best_params_,
            "model_path": str(model_path),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
            "confusion_matrix": cm,
            "classification_report": report_dict,
        }

        if test_auc > best_auc:
            best_auc = test_auc
            best_name = name

    best_model_path = None
    if best_name:
        best_model_path = metrics["models"][best_name]["model_path"]
        metrics["best_model"] = {
            "name": best_name,
            "test_roc_auc": best_auc,
            "model_path": best_model_path,
        }
        (RESULTS_DIR / "best_model.txt").write_text(best_model_path, encoding="utf-8")

    metrics_path = RESULTS_DIR / f"{target_id}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Metrics saved to %s", metrics_path)

    return {
        "target_id": target_id,
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "best_model_path": best_model_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train target-specific ChEMBL models.")
    parser.add_argument("target_id", help="ChEMBL target ID (e.g., CHEMBL1075091)")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached data and refetch activities from ChEMBL.",
    )
    parser.add_argument(
        "--chembl-sqlite",
        help="Override path to local chembl_<release>.db (auto-detects data/chembl_releases/*).",
    )
    parser.add_argument(
        "--skip-update-check",
        action="store_true",
        help="Skip remote timestamp checks before refreshing cached data.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = run_pipeline(
            args.target_id,
            force_refresh=args.force_refresh,
            chembl_sqlite=args.chembl_sqlite,
            skip_update_check=args.skip_update_check,
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Pipeline failed: %s", exc)
        sys.exit(1)

    best_model = result["metrics"].get("best_model")
    if best_model:
        LOGGER.info("Best model: %s (ROC-AUC=%.3f)", best_model["name"], best_model["test_roc_auc"])
        LOGGER.info("Best model path: %s", best_model["model_path"])
    LOGGER.info("Pipeline complete.")


if __name__ == "__main__":
    main()
