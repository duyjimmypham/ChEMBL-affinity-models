"""
Utility helpers shared by the scoring entry points (score_single/batch/smiles).
Import and call score_smiles() to reuse the core scoring logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np

# Import from our new modules
from config import RESULTS_DIR, FP_N_BITS, FP_RADIUS
from features import smiles_to_morgan


def load_best_model_path(target_id: Optional[str] = None) -> Optional[str]:
    """Finds the path to the best model for a given target.

    Args:
        target_id (Optional[str]): The ChEMBL target ID. If provided, checks
            for a target-specific metrics file.

    Returns:
        Optional[str]: The absolute path to the model file, or None if not found.
    """
    if target_id:
        metrics_path = RESULTS_DIR / f"{target_id}_metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as fh:
                metrics = json.load(fh)
            best = metrics.get("best_model") or {}
            if "model_path" in best:
                return best["model_path"]

    global_path = RESULTS_DIR / "best_model.txt"
    if global_path.exists():
        return global_path.read_text(encoding="utf-8").strip()
    return None


def load_model(model_path: str, cache: Dict[str, object]) -> object:
    """Loads a model from disk, using a cache to avoid reloading.

    Args:
        model_path (str): Path to the pickled model file.
        cache (Dict[str, object]): A dictionary to store loaded models.

    Returns:
        object: The loaded scikit-learn (or compatible) model.
    """
    if model_path in cache:
        return cache[model_path]
    model = joblib.load(model_path)
    cache[model_path] = model
    return model


def score_smiles(
    smiles: str,
    *,
    target_id: Optional[str] = None,
    model_cache: Optional[Dict[str, object]] = None,
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Score a single SMILES string.

    Args:
        smiles (str): The SMILES string to score.
        target_id (Optional[str]): The target ID to score against.
        model_cache (Optional[Dict[str, object]]): Cache for loaded models.

    Returns:
        Tuple[Optional[float], Optional[str], Optional[str]]:
            - probability (float or None): The predicted probability of activity.
            - model_path_used (str or None): The path of the model used.
            - error_message (str or None): Error description if scoring failed.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None, "missing_smiles"

    model_path = load_best_model_path(target_id)
    if not model_path:
        return None, None, "no_model_for_target" if target_id else "no_model_available"

    mp = Path(model_path)
    if not mp.exists():
        return None, None, f"model_missing:{mp}"

    # Use the centralized feature generation
    fp = smiles_to_morgan(smiles, n_bits=FP_N_BITS, radius=FP_RADIUS)
    if fp is None:
        return None, None, "invalid_smiles"

    cache = model_cache if model_cache is not None else {}
    model = load_model(str(mp), cache)
    
    # Reshape for single sample prediction
    proba = model.predict_proba([fp])[0, 1]
    return float(proba), str(mp), None
