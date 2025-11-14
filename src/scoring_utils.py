#!/usr/bin/env python3
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
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


N_BITS = 2048
RADIUS = 2


def smiles_to_morgan(smi: str, n_bits: int = N_BITS, radius: int = RADIUS):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_best_model_path(target_id: Optional[str] = None) -> Optional[str]:
    results_dir = Path("results")

    if target_id:
        metrics_path = results_dir / f"{target_id}_metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as fh:
                metrics = json.load(fh)
            best = metrics.get("best_model") or {}
            if "model_path" in best:
                return best["model_path"]

    global_path = results_dir / "best_model.txt"
    if global_path.exists():
        return global_path.read_text(encoding="utf-8").strip()
    return None


def load_model(model_path: str, cache: Dict[str, object]) -> object:
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

    Returns:
        (probability, model_path_used, error_message)
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None, "missing_smiles"

    model_path = load_best_model_path(target_id)
    if not model_path:
        return None, None, "no_model_for_target" if target_id else "no_model_available"

    mp = Path(model_path)
    if not mp.exists():
        return None, None, f"model_missing:{mp}"

    fp = smiles_to_morgan(smiles)
    if fp is None:
        return None, None, "invalid_smiles"

    cache = model_cache if model_cache is not None else {}
    model = load_model(str(mp), cache)
    proba = model.predict_proba([fp])[0, 1]
    return float(proba), str(mp), None
