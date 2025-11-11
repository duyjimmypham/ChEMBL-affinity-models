#!/usr/bin/env python3
"""
Fetch and summarize ChEMBL data for a target.

Usage:
    python src/inspect_chembl.py CHEMBL1075091
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from typing import List

import pandas as pd
from chembl_webresource_client.new_client import new_client

DATA_DIR = Path("data")


def fetch_activities(target_id: str) -> pd.DataFrame:
    """Fetch all activity rows for a given ChEMBL target."""
    acts = new_client.activity.filter(target_chembl_id=target_id)
    # 'acts' is an iterable of dicts; turn into DataFrame
    df = pd.DataFrame(list(acts))
    return df


def clean_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the columns we care about and make numeric values numeric."""
    keep_cols = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "assay_chembl_id",
        "assay_description",
        "document_chembl_id",
    ]
    cols_present = [c for c in keep_cols if c in df.columns]
    df = df[cols_present].copy()

    if "standard_value" in df.columns:
        df = df[df["standard_value"].notna()]
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
        df = df[df["standard_value"].notna()]

    return df


def save_activities_sqlite(df: pd.DataFrame, target_id: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    db_path = DATA_DIR / f"{target_id}_activities.db"
    con = sqlite3.connect(str(db_path))
    df.to_sql("activities", con, if_exists="replace", index=False)
    con.close()


def fetch_molecule_metadata(molecule_ids: List[str]) -> pd.DataFrame:
    """
    Fetch molecule-level info for a list of molecule_chembl_id.
    We keep only a small set of fields that are useful for us.
    """
    rows = []
    mol_client = new_client.molecule

    for mid in molecule_ids:
        try:
            rec = mol_client.get(mid)
        except Exception:
            # some clients prefer filter(...) instead of get(...)
            try:
                recs = mol_client.filter(molecule_chembl_id=mid)
                rec = list(recs)[0] if recs else None
            except Exception:
                rec = None

        if not rec:
            continue

        rows.append(
            {
                "molecule_chembl_id": mid,
                "pref_name": rec.get("pref_name"),
                "max_phase": rec.get("max_phase"),
                "molecule_type": rec.get("molecule_type"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["molecule_chembl_id", "pref_name", "max_phase", "molecule_type"]
        )
    return pd.DataFrame(rows)


def fetch_mechanisms_for_target(target_id: str) -> pd.DataFrame:
    """
    Fetch mechanism rows for the target.
    This is the “this molecule acts on this target” curated link.
    """
    try:
        mech = new_client.mechanism.filter(target_chembl_id=target_id)
        df = pd.DataFrame(list(mech))
    except Exception:
        df = pd.DataFrame()
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(str(path), index=False)


def build_summary(
    target_id: str,
    activities: pd.DataFrame,
    molecules: pd.DataFrame,
    mechanisms: pd.DataFrame,
) -> dict:
    summary = {
        "target_id": target_id,
        "n_activity_rows": int(len(activities)),
        "n_unique_molecules_in_activities": int(
            activities["molecule_chembl_id"].nunique()
        )
        if "molecule_chembl_id" in activities.columns
        else 0,
        "activity_types": activities["standard_type"].value_counts().to_dict()
        if "standard_type" in activities.columns
        else {},
        "activity_units": activities["standard_units"].value_counts().to_dict()
        if "standard_units" in activities.columns
        else {},
        "activity_relations": activities["standard_relation"].value_counts().to_dict()
        if "standard_relation" in activities.columns
        else {},
        "n_molecules_with_metadata": int(len(molecules)),
        "max_phase_counts": molecules["max_phase"].value_counts(dropna=False).to_dict()
        if "max_phase" in molecules.columns and len(molecules) > 0
        else {},
        "n_mechanism_rows": int(len(mechanisms)),
        "mechanism_molecules": mechanisms["molecule_chembl_id"]
        .dropna()
        .unique()
        .tolist()
        if "molecule_chembl_id" in mechanisms.columns
        else [],
    }
    return summary


def save_summary_json(summary: dict, target_id: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    json_path = DATA_DIR / f"{target_id}_summary.json"
    with open(str(json_path), "w") as f:
        json.dump(summary, f, indent=2)


def print_report(summary: dict) -> None:
    print("\n=== ChEMBL target inspection report ===")
    print(f"Target: {summary['target_id']}")
    print(f"Activity rows: {summary['n_activity_rows']}")
    print(
        f"Unique molecules in activities: {summary['n_unique_molecules_in_activities']}"
    )

    print("\nActivity types (from activities):")
    for k, v in summary.get("activity_types", {}).items():
        print(f"  {k}: {v}")

    print("\nUnits (from activities):")
    for k, v in summary.get("activity_units", {}).items():
        print(f"  {k}: {v}")

    print("\nRelations (from activities):")
    for k, v in summary.get("activity_relations", {}).items():
        print(f"  {k}: {v}")

    print(f"\nMolecules with metadata fetched: {summary['n_molecules_with_metadata']}")
    if summary.get("max_phase_counts"):
        print("max_phase counts:")
        for k, v in summary["max_phase_counts"].items():
            print(f"  {k}: {v}")

    print(f"\nMechanism rows for this target: {summary['n_mechanism_rows']}")
    if summary["n_mechanism_rows"] > 0:
        print("Example mechanism-linked molecules:")
        for mid in summary["mechanism_molecules"][:10]:
            print(f"  {mid}")
    print("=== End of report ===\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/inspect_chembl.py CHEMBL_ID")
        sys.exit(1)

    target_id = sys.argv[1].strip().upper()
    DATA_DIR.mkdir(exist_ok=True)

    print(f"[1/5] Fetching activities for {target_id} ...")
    acts_df = fetch_activities(target_id)
    if acts_df.empty:
        print("No activities found for this target.")
        sys.exit(0)

    acts_df = clean_activities(acts_df)
    save_activities_sqlite(acts_df, target_id)
    print(f"[2/5] Saved activities to data/{target_id}_activities.db")

    # fetch molecule metadata
    if "molecule_chembl_id" in acts_df.columns:
        unique_mols = (
            acts_df["molecule_chembl_id"].dropna().astype(str).unique().tolist()
        )
    else:
        unique_mols = []
    print(f"[3/5] Fetching molecule metadata for {len(unique_mols)} molecules ...")
    mols_df = fetch_molecule_metadata(unique_mols)
    mols_path = DATA_DIR / f"{target_id}_molecules.csv"
    write_csv(mols_df, mols_path)
    print(f"Saved molecule metadata to {mols_path}")

    # fetch mechanisms
    print(f"[4/5] Fetching mechanism entries for {target_id} ...")
    mech_df = fetch_mechanisms_for_target(target_id)
    mech_path = DATA_DIR / f"{target_id}_mechanisms.csv"
    write_csv(mech_df, mech_path)
    print(f"Saved mechanisms to {mech_path}")

    # build and save summary
    summary = build_summary(target_id, acts_df, mols_df, mech_df)
    save_summary_json(summary, target_id)

    # print report for cross-checking
    print_report(summary)


if __name__ == "__main__":
    main()
