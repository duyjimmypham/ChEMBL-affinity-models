#!/usr/bin/env python3
"""
Shared helpers for resilient ChEMBL API access (pagination, retries, checkpoints).
Imported by inspect_chembl.py and pipeline.py.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_PAGE_SIZE = 1000
DEFAULT_MAX_RETRIES = 5
BACKOFF_SECONDS = 2.0


def _load_checkpoint(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _append_checkpoint(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


def fetch_paginated(
    resource,
    filters: Dict[str, Any],
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    checkpoint_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch all rows for a ChEMBL resource using pagination.

    Args:
        resource: chembl_webresource_client handle, e.g., new_client.activity
        filters: keyword arguments forwarded to .filter(...)
        page_size: number of rows to request per page.
        max_retries: retries per page before raising.
        checkpoint_path: if provided, progress is written as JSONL so we can resume.
        logger: optional logger for progress updates.
    """

    log = logger or logging.getLogger(__name__)
    rows: List[Dict[str, Any]] = []
    offset = 0

    if checkpoint_path:
        rows = _load_checkpoint(checkpoint_path)
        existing_target = None
        if rows:
            first = rows[0]
            existing_target = first.get("target_chembl_id")

        current_target = filters.get("target_chembl_id")
        if rows and existing_target and existing_target == current_target:
            offset = len(rows)
            log.info(
                "Resuming ChEMBL fetch for %s from checkpoint (%d rows)",
                existing_target,
                offset,
            )
        else:
            if rows:
                log.info("Discarding checkpoint because target changed (%s -> %s)", existing_target, current_target)
            rows = []
            if checkpoint_path.exists():
                checkpoint_path.unlink()

    while True:
        attempt = 0
        batch: Optional[List[Dict[str, Any]]] = None
        while attempt < max_retries:
            try:
                batch_iter = resource.filter(limit=page_size, offset=offset, **filters)
                batch = list(batch_iter)
                break
            except Exception as exc:
                wait_time = BACKOFF_SECONDS * (2 ** attempt)
                log.warning(
                    "ChEMBL fetch error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_time,
                )
                time.sleep(wait_time)
                attempt += 1
        if batch is None:
            raise RuntimeError("Exceeded retry budget while fetching from ChEMBL.")
        if not batch:
            break

        rows.extend(batch)
        offset += len(batch)
        log.info("Fetched %d rows (total=%d)", len(batch), offset)

        if checkpoint_path:
            _append_checkpoint(checkpoint_path, batch)

    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()

    return rows
