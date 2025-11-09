#!/usr/bin/env python3
"""
Inspect a ChEMBL target by ID, fetch all activity data,
summarize it, and save to SQLite or CSV.

Usage:
    python inspect_chembl.py CHEMBL1075091

Creates:
    data/{target_id}_activities.db
    data/{target_id}_summary.json
"""

import sys
import os
import json
import sqlite3
import pandas as pd
from chembl_webresource_client.new_client import new_client


def fetch_target_data(target_id: str) -> pd.DataFrame:
    """Fetch all activity data for a given ChEMBL target."""
    acts = new_client.activity.filter(target_chembl_id=target_id)
    df = pd.DataFrame(acts)
    if df.empty:
        print(f"No activities found for {target_id}")
        sys.exit(0)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Select useful columns and clean numeric values."""
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

    df = df.dropna(subset=["standard_value"])
    df = df[pd.to_numeric(df["standard_value"], errors="coerce").notnull()]
    df["standard_value"] = df["standard_value"].astype(float)
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Generate quick summary stats."""
    summary = {
        "rows_total": int(len(df)),
        "unique_molecules": int(df["molecule_chembl_id"].nunique()),
        "activity_types": df["standard_type"].value_counts().to_dict(),
        "units": df["standard_units"].value_counts().to_dict(),
        "relations": df["standard_relation"].value_counts().to_dict(),
    }
    return summary


def save_to_sqlite(df: pd.DataFrame, target_id: str):
    """Save the dataframe to a SQLite database."""
    os.makedirs("data", exist_ok=True)
    db_path = f"data/{target_id}_activities.db"
    conn = sqlite3.connect(db_path)
    df.to_sql("activities", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Saved {len(df)} rows to {db_path}")
    return db_path


def save_summary_json(summary: dict, target_id: str):
    """Save the summary dictionary to JSON."""
    os.makedirs("data", exist_ok=True)
    json_path = f"data/{target_id}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Summary saved to {json_path}")
    return json_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_chembl.py CHEMBL_ID")
        sys.exit(1)

    target_id = sys.argv[1].strip().upper()
    print(f"Fetching ChEMBL activities for {target_id} ...")

    df_raw = fetch_target_data(target_id)
    print(f"Total rows fetched: {len(df_raw)}")

    df = clean_dataframe(df_raw)
    print(f"Rows with numeric standard_value: {len(df)}")

    summary = summarize(df)
    print(json.dumps(summary, indent=2))

    save_to_sqlite(df, target_id)
    save_summary_json(summary, target_id)


if __name__ == "__main__":
    main()
