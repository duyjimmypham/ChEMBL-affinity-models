#!/usr/bin/env python3
"""
Helpers for tracking local ChEMBL cache metadata and detecting when remote data changes.
Imported by inspect_chembl.py and pipeline.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from chembl_webresource_client.new_client import new_client

META_DIR = Path("data") / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger("chembl_cache")


def _meta_path(target_id: str) -> Path:
    return META_DIR / f"{target_id}_activities_meta.json"


def load_target_meta(target_id: str) -> Optional[Dict[str, Any]]:
    path = _meta_path(target_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_target_meta(target_id: str, last_updated: Optional[str], row_count: int) -> None:
    payload = {
        "target_id": target_id,
        "last_updated": last_updated,
        "row_count": row_count,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    path = _meta_path(target_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _to_datetime(timestamp: Optional[str]) -> Optional[datetime]:
    if not timestamp:
        return None
    ts = timestamp.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def fetch_latest_remote_timestamp(target_id: str, logger: Optional[logging.Logger] = None) -> Optional[str]:
    log = logger or LOGGER
    client = new_client.activity
    try:
        rows = client.filter(
            target_chembl_id=target_id,
            limit=1,
            order_by="-updated_on",
        )
        for row in rows:
            return row.get("updated_on")
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("Failed to fetch latest timestamp for %s: %s", target_id, exc)
    return None


def has_remote_updates(
    target_id: str,
    last_known_timestamp: Optional[str],
    logger: Optional[logging.Logger] = None,
) -> bool:
    if not last_known_timestamp:
        return True

    latest = fetch_latest_remote_timestamp(target_id, logger=logger)
    if latest is None:
        # conservatively fetch
        return True

    known_dt = _to_datetime(last_known_timestamp)
    latest_dt = _to_datetime(latest)
    if not known_dt or not latest_dt:
        return True
    return latest_dt > known_dt
