#!/usr/bin/env python3
"""
Utilities for reading data directly from a local ChEMBL SQLite dump.
Imported by the inspection and pipeline scripts when a local database is available.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


DEFAULT_DUMP_DIR = Path("data") / "chembl_releases"
LEGACY_DUMP_DIR = Path("data") / "chembl_dumps"


ACTIVITY_QUERY_BASE = """
SELECT
    md.chembl_id AS molecule_chembl_id,
    cs.canonical_smiles,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    ass.chembl_id AS assay_chembl_id,
    ass.description AS assay_description,
    doc.chembl_id AS document_chembl_id,
    {updated_on_expr}
FROM activities act
JOIN assays ass ON act.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN molecule_dictionary md ON act.molregno = md.molregno
LEFT JOIN compound_structures cs ON act.molregno = cs.molregno
LEFT JOIN docs doc ON act.doc_id = doc.doc_id
WHERE td.chembl_id = ?
"""

MECHANISM_QUERY = """
SELECT
    md.chembl_id AS molecule_chembl_id,
    mech.mechanism_of_action,
    mech.action_type,
    mech.tid
FROM mechanism mech
JOIN target_dictionary td ON mech.tid = td.tid
JOIN molecule_dictionary md ON mech.molregno = md.molregno
WHERE td.chembl_id = ?
"""

DRUG_MECHANISM_QUERY = """
SELECT
    md.chembl_id AS molecule_chembl_id,
    dm.mechanism_of_action,
    dm.action_type,
    dm.tid
FROM drug_mechanism dm
JOIN target_dictionary td ON dm.tid = td.tid
JOIN molecule_dictionary md ON dm.molregno = md.molregno
WHERE td.chembl_id = ?
"""

MOLECULE_METADATA_QUERY = """
SELECT
    chembl_id AS molecule_chembl_id,
    pref_name,
    max_phase,
    molecule_type
FROM molecule_dictionary
WHERE chembl_id IN ({placeholders})
"""


def _connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"ChEMBL SQLite database not found: {path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cursor.fetchone() is not None


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def fetch_local_activities(target_id: str, db_path: Path) -> pd.DataFrame:
    with _connect(db_path) as conn:
        include_updated = _table_has_column(conn, "activities", "updated_on")
        updated_expr = "act.updated_on" if include_updated else "NULL AS updated_on"
        query = ACTIVITY_QUERY_BASE.format(updated_on_expr=updated_expr)
        return pd.read_sql_query(query, conn, params=(target_id,))


def fetch_local_mechanisms(target_id: str, db_path: Path) -> pd.DataFrame:
    with _connect(db_path) as conn:
        if _table_exists(conn, "drug_mechanism"):
            query = DRUG_MECHANISM_QUERY
        elif _table_exists(conn, "mechanism"):
            query = MECHANISM_QUERY
        else:
            return pd.DataFrame(
                columns=["molecule_chembl_id", "mechanism_of_action", "action_type", "tid"]
            )
        return pd.read_sql_query(query, conn, params=(target_id,))


def fetch_local_molecule_metadata(molecule_ids: Sequence[str], db_path: Path, chunk_size: int = 500) -> pd.DataFrame:
    ids = [mid for mid in molecule_ids if mid]
    if not ids:
        return pd.DataFrame(columns=["molecule_chembl_id", "pref_name", "max_phase", "molecule_type"])

    uniq_ids = list(dict.fromkeys(ids))
    rows = []
    with _connect(db_path) as conn:
        for i in range(0, len(uniq_ids), chunk_size):
            batch = uniq_ids[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(batch))
            query = MOLECULE_METADATA_QUERY.format(placeholders=placeholders)
            result = conn.execute(query, batch)
            rows.extend(result.fetchall())

    if not rows:
        return pd.DataFrame(columns=["molecule_chembl_id", "pref_name", "max_phase", "molecule_type"])

    return pd.DataFrame([dict(r) for r in rows])


def find_default_local_db(env_var: str = "CHEMBL_SQLITE_PATH") -> Optional[Path]:
    env_value = os.getenv(env_var)
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.exists():
            return candidate

    dump_dir = DEFAULT_DUMP_DIR if DEFAULT_DUMP_DIR.exists() else LEGACY_DUMP_DIR
    if not dump_dir.exists():
        return None

    candidates = sorted(dump_dir.glob("chembl_*.db"), reverse=True)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
