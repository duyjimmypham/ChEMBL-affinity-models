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

from scoring_utils import score_smiles as score_smiles_entry


def read_input(path: Path, table: str | None) -> pd.DataFrame:
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
    args = parser.parse_args()

    inp_path = Path(args.input)
    df = read_input(inp_path, args.table)

    smiles_col = args.smiles_col
    has_target = args.target_col in df.columns

    if smiles_col not in df.columns:
        raise ValueError(f"Input missing SMILES column '{smiles_col}'")

    df["pred_proba"] = np.nan
    df["model_path_used"] = None
    df["scoring_error"] = None

    model_cache: dict[str, object] = {}

    for idx, row in df.iterrows():
        smi = row.get(smiles_col)
        target_id = None
        if has_target:
            tgt_val = row.get(args.target_col)
            if pd.notna(tgt_val):
                target_id = str(tgt_val)

        proba, model_path, error = score_smiles_entry(
            smi,
            target_id=target_id,
            model_cache=model_cache,
        )

        if error:
            df.at[idx, "scoring_error"] = error
            continue

        df.at[idx, "pred_proba"] = proba
        df.at[idx, "model_path_used"] = model_path

    df.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
