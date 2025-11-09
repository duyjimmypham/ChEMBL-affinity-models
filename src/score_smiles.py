#!/usr/bin/env python3
"""
Batch-score SMILES with the best model(s) saved by pipeline.py.

Examples
--------
# simplest: use the last trained model (results/best_model.txt)
python score_smiles.py --input smiles.csv --output scored.csv

# CSV has per-row target_id, use each row's model if available
python score_smiles.py --input smiles.csv --output scored.csv --target-col target_id

# SQLite table
python score_smiles.py --input toscore.db --table molecules --output scored.csv

Columns
-------
- input must have a SMILES column (default: "smiles")
- if it also has a target column (default: "target_id"), we try to load that target's model
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


def load_best_model_path_for_target(target_id: str) -> str | None:
    """
    Look for results/{target_id}_metrics.json and read ["best_model"]["model_path"].
    Return None if not found.
    """
    metrics_path = Path("results") / f"{target_id}_metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    best = metrics.get("best_model")
    if not best:
        return None
    return best.get("model_path")


def load_global_best_model_path() -> str | None:
    global_path = Path("results") / "best_model.txt"
    if global_path.exists():
        return global_path.read_text().strip()
    return None


def load_model_cache(model_path: str, cache: dict) -> object:
    """
    Load a model once and keep it in memory.
    """
    if model_path in cache:
        return cache[model_path]
    model = joblib.load(model_path)
    cache[model_path] = model
    return model


def read_input(args: argparse.Namespace) -> pd.DataFrame:
    inp = Path(args.input)
    if inp.suffix.lower() == ".csv":
        df = pd.read_csv(inp)
    elif inp.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        if not args.table:
            raise ValueError("For SQLite input you must provide --table")
        con = sqlite3.connect(inp)
        df = pd.read_sql_query(f"SELECT * FROM {args.table}", con)
        con.close()
    else:
        raise ValueError(f"Unsupported input format: {inp.suffix}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV or SQLite DB file")
    parser.add_argument("--output", required=True, help="Output CSV with predictions")
    parser.add_argument("--table", help="Table name (only for SQLite input)")
    parser.add_argument("--smiles-col", default="smiles", help="Name of the SMILES column")
    parser.add_argument("--target-col", default="target_id", help="Name of the target column (optional)")
    args = parser.parse_args()

    df = read_input(args)

    if args.smiles_col not in df.columns:
        raise ValueError(f"Input missing SMILES column '{args.smiles_col}'")

    has_target = args.target_col in df.columns

    # We'll add these columns
    df["pred_proba"] = np.nan
    df["model_path_used"] = None
    df["smiles_valid"] = True
    df["scoring_error"] = None

    # global fallback model
    global_model_path = load_global_best_model_path()

    # small cache so we don't reload same model 1000 times
    model_cache: dict[str, object] = {}

    for idx, row in df.iterrows():
        smi = row[args.smiles_col]
        tgt = row[args.target_col] if has_target else None

        # pick model path
        model_path = None
        if tgt:
            model_path = load_best_model_path_for_target(str(tgt))
        if model_path is None:
            model_path = global_model_path

        if model_path is None:
            df.at[idx, "scoring_error"] = "no model found"
            df.at[idx, "smiles_valid"] = True
            continue

        model_path = str(model_path)
        if not Path(model_path).exists():
            df.at[idx, "scoring_error"] = f"model missing: {model_path}"
            continue

        # featurize
        fp = smiles_to_morgan(smi)
        if fp is None:
            df.at[idx, "smiles_valid"] = False
            df.at[idx, "scoring_error"] = "invalid SMILES"
            continue

        # load model
        model = load_model_cache(model_path, model_cache)

        # predict
        proba = model.predict_proba([fp])[0, 1]
        df.at[idx, "pred_proba"] = float(proba)
        df.at[idx, "model_path_used"] = model_path

    # write
    out_path = Path(args.output)
    df.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
