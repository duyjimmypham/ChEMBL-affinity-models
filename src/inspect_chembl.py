#!/usr/bin/env python3
"""
Fetch and summarize ChEMBL data for a target.

Usage:
    python src/inspect_chembl.py CHEMBL203 [--fast] [--chembl-sqlite path]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

from chembl_cache import has_remote_updates, load_target_meta, save_target_meta
from chembl_client_utils import fetch_paginated
from local_chembl import (
    fetch_local_activities,
    fetch_local_mechanisms,
    fetch_local_molecule_metadata,
    find_default_local_db,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

LOG_LEVEL = os.getenv("CHEMBL_PIPELINE_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
LOGGER = logging.getLogger("chembl_inspect")

ACTIVITY_PAGE_SIZE = int(os.getenv("CHEMBL_ACTIVITY_PAGE_SIZE", "5000"))
MOLECULE_BATCH_SIZE = int(os.getenv("CHEMBL_MOL_BATCH_SIZE", "1000"))


def fetch_activities(target_id: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    log = logger or LOGGER
    checkpoint = DATA_DIR / f"{target_id}_inspect_activities_checkpoint.jsonl"
    rows = fetch_paginated(
        new_client.activity,
        {"target_chembl_id": target_id},
        checkpoint_path=checkpoint,
        page_size=ACTIVITY_PAGE_SIZE,
        logger=log,
    )
    return pd.DataFrame(rows)


def clean_activities(df: pd.DataFrame) -> pd.DataFrame:
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
        "updated_on",
    ]
    cols_present = [c for c in keep_cols if c in df.columns]
    df = df[cols_present].copy()

    if "standard_value" in df.columns:
        df = df[df["standard_value"].notna()]
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
        df = df[df["standard_value"].notna()]

    return df


def compute_last_updated(df: pd.DataFrame) -> Optional[str]:
    if "updated_on" not in df.columns:
        return None
    series = pd.to_datetime(df["updated_on"], errors="coerce", utc=True)
    series = series.dropna()
    if series.empty:
        return None
    return series.max().isoformat()


def save_activities_sqlite(df: pd.DataFrame, target_id: str) -> Path:
    db_path = DATA_DIR / f"{target_id}_activities.db"
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA synchronous = OFF;")
        con.execute("PRAGMA journal_mode = MEMORY;")
        df.to_sql("activities", con, if_exists="replace", index=False)
    finally:
        con.close()
    return db_path


def _chunked(items: List[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_molecule_metadata(
    molecule_ids: List[str],
    *,
    max_retries: int = 3,
    logger: Optional[logging.Logger] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    rows = []
    log = logger or LOGGER
    mol_client = new_client.molecule
    seen_ids = set()

    total_batches = max(1, (len(molecule_ids) + MOLECULE_BATCH_SIZE - 1) // MOLECULE_BATCH_SIZE)
    iterator = _chunked(molecule_ids, MOLECULE_BATCH_SIZE)
    progress = tqdm(
        iterator,
        total=total_batches,
        desc="Molecule metadata",
        disable=not show_progress,
    )

    for batch in progress:
        batch = [mid for mid in batch if mid and mid not in seen_ids]
        if not batch:
            continue
        seen_ids.update(batch)

        recs = []
        for attempt in range(max_retries):
            try:
                recs = list(mol_client.filter(molecule_chembl_id__in=batch))
                break
            except Exception as exc:  # pylint: disable=broad-except
                wait_time = 2 ** attempt
                log.warning(
                    "Molecule batch fetch failed (size=%d, attempt %d/%d): %s. Retrying in %ss",
                    len(batch),
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_time,
                )
                time.sleep(wait_time)
        if not recs:
            continue

        for rec in recs:
            rows.append(
                {
                    "molecule_chembl_id": rec.get("molecule_chembl_id"),
                    "pref_name": rec.get("pref_name"),
                    "max_phase": rec.get("max_phase"),
                    "molecule_type": rec.get("molecule_type"),
                }
            )

    progress.close()

    if not rows:
        return pd.DataFrame(
            columns=["molecule_chembl_id", "pref_name", "max_phase", "molecule_type"]
        )
    return pd.DataFrame(rows)


def fetch_mechanisms_for_target(target_id: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    log = logger or LOGGER
    rows = fetch_paginated(
        new_client.mechanism,
        {"target_chembl_id": target_id},
        checkpoint_path=None,
        logger=log,
    )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def build_summary(
    target_id: str,
    activities: pd.DataFrame,
    molecules: pd.DataFrame,
    mechanisms: pd.DataFrame,
) -> Dict[str, object]:
    return {
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


def save_summary_json(summary: Dict[str, object], target_id: str) -> Path:
    json_path = DATA_DIR / f"{target_id}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return json_path


def log_summary(summary: Dict[str, object], logger: Optional[logging.Logger] = None) -> None:
    log = logger or LOGGER
    log.info("=== ChEMBL target inspection report ===")
    log.info("Target: %s", summary["target_id"])
    log.info("Activity rows: %d", summary["n_activity_rows"])
    log.info("Unique molecules: %d", summary["n_unique_molecules_in_activities"])

    log.info("Activity types: %s", summary.get("activity_types", {}))
    log.info("Units: %s", summary.get("activity_units", {}))
    log.info("Relations: %s", summary.get("activity_relations", {}))

    log.info("Molecules with metadata: %d", summary["n_molecules_with_metadata"])
    log.info("Max phase counts: %s", summary.get("max_phase_counts", {}))
    log.info("Mechanism rows: %d", summary["n_mechanism_rows"])
    if summary["mechanism_molecules"]:
        log.info(
            "Example mechanism-linked molecules: %s",
            summary["mechanism_molecules"][:10],
        )
    log.info("=== End of report ===")


def run_inspection(
    target_id: str,
    *,
    force_refresh: bool = False,
    fast: bool = False,
    chembl_sqlite: Optional[str] = None,
    skip_update_check: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    log = logger or LOGGER
    target_id = target_id.strip().upper()

    db_path = DATA_DIR / f"{target_id}_activities.db"

    if not chembl_sqlite:
        auto_db = find_default_local_db()
        if auto_db:
            log.info("Detected local ChEMBL SQLite at %s", auto_db)
            chembl_sqlite = str(auto_db)

    local_db_path = Path(chembl_sqlite).expanduser() if chembl_sqlite else None

    acts_df: Optional[pd.DataFrame] = None

    if local_db_path:
        log.info("Loading activities from local ChEMBL SQLite: %s", local_db_path)
        acts_df = fetch_local_activities(target_id, local_db_path)
        save_activities_sqlite(acts_df, target_id)
        save_target_meta(target_id, compute_last_updated(acts_df), len(acts_df))
    else:
        meta = load_target_meta(target_id)
        if (
            db_path.exists()
            and not force_refresh
            and not skip_update_check
            and meta
            and not has_remote_updates(target_id, meta.get("last_updated"), logger=log)
        ):
            log.info(
                "No new activities for %s since %s; using cached DB.",
                target_id,
                meta.get("last_updated"),
            )
            with sqlite3.connect(str(db_path)) as conn:
                acts_df = pd.read_sql("SELECT * FROM activities", conn)
        if acts_df is None:
            log.info("Fetching activities for %s ...", target_id)
            acts_df = fetch_activities(target_id, logger=log)
            if acts_df.empty:
                raise RuntimeError("No activities found for this target.")
            acts_df = clean_activities(acts_df)
            save_activities_sqlite(acts_df, target_id)
            save_target_meta(target_id, compute_last_updated(acts_df), len(acts_df))
            log.info("Saved activities to %s", db_path)

    if fast:
        mols_df = pd.DataFrame(columns=["molecule_chembl_id", "pref_name", "max_phase", "molecule_type"])
        mech_df = pd.DataFrame(columns=["molecule_chembl_id", "mechanism_of_action"])
        log.info("Fast mode enabled: skipped molecule/mechanism fetches.")
    else:
        molecule_ids = (
            acts_df["molecule_chembl_id"].dropna().astype(str).unique().tolist()
            if "molecule_chembl_id" in acts_df.columns
            else []
        )
        if local_db_path:
            log.info("Fetching molecule metadata from local SQLite ...")
            mols_df = fetch_local_molecule_metadata(molecule_ids, local_db_path)
            log.info("Fetching mechanisms from local SQLite ...")
            mech_df = fetch_local_mechanisms(target_id, local_db_path)
        else:
            log.info("Fetching molecule metadata for %d molecules ...", len(molecule_ids))
            mols_df = fetch_molecule_metadata(molecule_ids, logger=log, show_progress=True)
            log.info("Fetching mechanism entries for %s ...", target_id)
            mech_df = fetch_mechanisms_for_target(target_id, logger=log)

    mols_path = DATA_DIR / f"{target_id}_molecules.csv"
    mech_path = DATA_DIR / f"{target_id}_mechanisms.csv"
    write_csv(mols_df, mols_path)
    write_csv(mech_df, mech_path)

    summary = build_summary(target_id, acts_df, mols_df, mech_df)
    summary_path = save_summary_json(summary, target_id)
    log_summary(summary, logger=log)

    return {
        "target_id": target_id,
        "activities_path": str(db_path),
        "molecules_path": str(mols_path),
        "mechanisms_path": str(mech_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a ChEMBL target.")
    parser.add_argument("target_id", help="ChEMBL target ID (e.g., CHEMBL1075091)")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refetch activities even if cached data exists.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip molecule/mechanism fetches (activities only).",
    )
    parser.add_argument(
        "--chembl-sqlite",
        help="Override path to a local chembl_<release>.db; autodetects data/chembl_releases/* if omitted.",
    )
    parser.add_argument(
        "--skip-update-check",
        action="store_true",
        help="Do not query remote timestamps before refreshing cached data.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        run_inspection(
            args.target_id,
            force_refresh=args.force_refresh,
            fast=args.fast,
            chembl_sqlite=args.chembl_sqlite,
            skip_update_check=args.skip_update_check,
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Inspection failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
