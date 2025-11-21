#!/usr/bin/env python3
"""
Batch-score SMILES using the best models saved by the pipeline.

Usage:
    python src/score_batch.py --input smiles.csv --output scored.csv
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from scoring_utils import load_best_model_path, load_model
from features import batch_smiles_to_morgan
from config import FP_N_BITS, FP_RADIUS


def read_input(path: Path, table: str | None) -> pd.DataFrame:
    """Reads input data from CSV or SQLite."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        if not table:
            raise ValueError("For SQLite input you must supply --table")
        with sqlite3.connect(str(path)) as con:
            return pd.read_sql_query(f"SELECT * FROM {table}", con)
    raise ValueError(f"Unsupported input format: {suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV or SQLite DB")
    parser.add_argument("--output", required=True, help="Output CSV for predictions")
    parser.add_argument("--table", help="Table name (for SQLite input)")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name")
    parser.add_argument(
        "--target-col",
        default="target_id",
        help="Target column name (optional)",
    )
    parser.add_argument(
        "--target-id",
        help="Global target ID to use if not specified in the file (overrides file column if present)",
    )
    args = parser.parse_args()

    inp_path = Path(args.input)
    df = read_input(inp_path, args.table)

    smiles_col = args.smiles_col
    if smiles_col not in df.columns:
        raise ValueError(f"Input missing SMILES column '{smiles_col}'")

    # Determine target ID(s)
    # If --target-id is provided, use it for all rows.
    # Otherwise, look for --target-col in df.
    # If neither, fail (unless we want to support a default model, but that's risky).
    
    global_target = args.target_id
    
    if global_target:
        print(f"Using global target ID: {global_target}")
        # Process all as one batch
        process_batch(df, smiles_col, global_target)
    elif args.target_col in df.columns:
        print(f"Using target IDs from column: {args.target_col}")
        # Group by target_id and process batches
        results = []
        grouped = df.groupby(args.target_col)
        for target_id, group in grouped:
            # We process the group copy, then append to results
            # Note: This might reorder rows compared to input.
            # If order matters, we should assign back to original index.
            sub_df = group.copy()
            process_batch(sub_df, smiles_col, str(target_id))
            results.append(sub_df)
        
        # Reassemble
        df = pd.concat(results)
    else:
        # Fallback: Try to load a default model? Or error?
        # Let's try to load the "best_model.txt" global default if it exists
        print("No target ID specified. Attempting to use global default model...")
        process_batch(df, smiles_col, None)

    df.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")


def process_batch(df: pd.DataFrame, smiles_col: str, target_id: str | None) -> None:
    """Scores a DataFrame in-place using vectorized operations."""
    
    # 1. Load Model
    model_path = load_best_model_path(target_id)
    if not model_path:
        df["scoring_error"] = "no_model_available"
        df["pred_proba"] = np.nan
        return

    mp = Path(model_path)
    if not mp.exists():
        df["scoring_error"] = f"model_missing:{mp}"
        df["pred_proba"] = np.nan
        return

    # We don't need a complex cache here since we process by group/batch
    model = load_model(str(mp), {})
    
    # 2. Vectorize Fingerprints
    smiles_list = df[smiles_col].astype(str).tolist()
    X, valid_mask = batch_smiles_to_morgan(smiles_list, n_bits=FP_N_BITS, radius=FP_RADIUS)
    
    # 3. Predict
    # Initialize columns
    if "pred_proba" not in df.columns:
        df["pred_proba"] = np.nan
    if "model_path_used" not in df.columns:
        df["model_path_used"] = None
    if "scoring_error" not in df.columns:
        df["scoring_error"] = None

    # Only predict for valid SMILES
    if np.any(valid_mask):
        X_valid = X[valid_mask]
        try:
            # predict_proba returns [n_samples, 2] usually
            probs = model.predict_proba(X_valid)[:, 1]
            
            # Assign back. We need to be careful with indexing if df is a slice or filtered.
            # Since we are modifying the passed df (which is a copy or slice in the main loop),
            # we can use boolean indexing on the df itself assuming the order matches smiles_list.
            
            # Create a temporary array for the full batch
            full_probs = np.full(len(df), np.nan)
            full_probs[valid_mask] = probs
            
            df["pred_proba"] = full_probs
            df["model_path_used"] = str(mp)
            
        except Exception as e:
            df["scoring_error"] = f"prediction_failed:{e}"
    
    # Mark invalid SMILES
    # We can use the inverse of valid_mask to set error
    invalid_indices = ~valid_mask
    if np.any(invalid_indices):
        df.loc[invalid_indices, "scoring_error"] = "invalid_smiles"

if __name__ == "__main__":
    main()
