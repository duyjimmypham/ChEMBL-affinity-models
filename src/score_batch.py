#!/usr/bin/env python3
"""
Batch-score SMILES using the best models saved by the pipeline.

Usage:
    # CSV
    python score_batch.py --input smiles.csv --output scored.csv

    # CSV with per-row target_id
    python score_batch.py --input smiles.csv --output scored.csv --target-col target_id

    # SQLite
    python score_batch.py --input compounds.db --table molecules --output scored.csv
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser.add_argument("--input", required=True, help="Input CSV or SQLite DB")
    parser.add_argument("--output", required=True, help="Output CSV for predictions")
    parser.add_argument("--table", help="Table name (for SQLite input)")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name")
    parser.add_argument("--target-col", default="target_id", help="Target column name (optional)")
    args = parser.parse_args()

    inp_path = Path(args.input)
    if inp_path.suffix.lower() == ".csv":
        df = pd.read_csv(inp_path)
    elif inp_path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        if not args.table:
            raise ValueError("For SQLite input you must supply --table")
        con = sqlite3.connect(inp_path)
        df = pd.read_sql_query(f"SELECT * FROM {args.table}", con)
        con.close()
    else:
        raise ValueError(f"Unsupported input format: {inp_path.suffix}")

    smiles_col = args.smiles_col
    has_target = args.target_col in df.columns

    df["pred_proba"] = np.nan
    df["model_path_used"] = None
    df["scoring_error"] = None

    model_cache: dict[str, object] = {}

    for idx, row in df.iterrows():
        smi = row.get(smiles_col, None)
        if not isinstance(smi, str) or not smi.strip():
            df.at[idx, "scoring_error"] = "missing_smiles"
            continue

        target_id = row[args.target_col] if has_target else None
        model_path = load_best_model_path(str(target_id) if target_id else None)
        if not model_path:
            df.at[idx, "scoring_error"] = "no_model_for_target"
            continue

        mp = Path(model_path)
        if not mp.exists():
            df.at[idx, "scoring_error"] = f"model_missing:{mp}"
            continue

        # featurize
        fp = smiles_to_morgan(smi)
        if fp is None:
            df.at[idx, "scoring_error"] = "invalid_smiles"
            continue

        # load model (cache by path)
        if model_path in model_cache:
            model = model_cache[model_path]
        else:
            model = joblib.load(mp)
            model_cache[model_path] = model

        proba = model.predict_proba([fp])[0, 1]
        df.at[idx, "pred_proba"] = float(proba)
        df.at[idx, "model_path_used"] = model_path

    df.to_csv(args.output, index=False)
    print(f"✅ Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
