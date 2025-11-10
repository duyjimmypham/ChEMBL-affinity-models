#!/usr/bin/env python3
"""
Score a single SMILES string with the best model saved by the pipeline.

Usage:
    python score_single.py "CCOc1ccccc1" --target CHEMBL1075091
If --target is omitted, it falls back to results/best_model.txt.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


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


def load_best_model_path(target_id: str | None = None) -> str | None:
    results_dir = Path("results")

    if target_id:
        metrics_path = results_dir / f"{target_id}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            best = metrics.get("best_model") or {}
            if "model_path" in best:
                return best["model_path"]

    global_path = results_dir / "best_model.txt"
    if global_path.exists():
        return global_path.read_text().strip()

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles", help="SMILES string to score")
    parser.add_argument("--target", help="ChEMBL ID used for training (e.g. CHEMBL1075091)")
    args = parser.parse_args()

    model_path = load_best_model_path(args.target)
    if not model_path:
        print("❌ No trained model found. Run the pipeline first.")
        sys.exit(1)

    mp = Path(model_path)
    if not mp.exists():
        print(f"❌ Model file not found: {mp}")
        sys.exit(1)

    model = joblib.load(mp)

    fp = smiles_to_morgan(args.smiles)
    if fp is None:
        print("❌ Invalid SMILES string.")
        sys.exit(1)

    proba = model.predict_proba([fp])[0, 1]
    print(f"✅ Model: {mp.name}")
    print(f"SMILES: {args.smiles}")
    print(f"Predicted activity probability: {proba:.3f}")


if __name__ == "__main__":
    main()
