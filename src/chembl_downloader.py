#!/usr/bin/env python3
"""
Download official ChEMBL SQLite dumps for offline use.

Usage:
    python src/chembl_downloader.py --release 36 [--output data/chembl_releases]
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("chembl_downloader")

BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases"
DEFAULT_OUTPUT_DIR = Path("data") / "chembl_releases"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {dest.name}",
        ) as progress, dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))


def _extract_sqlite_from_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        sqlite_member = next((m for m in tar.getmembers() if m.name.endswith(".db")), None)
        if sqlite_member is None:
            raise RuntimeError("Could not find .db file inside archive.")

        with tempfile.TemporaryDirectory() as tmpdir:
            tar.extract(sqlite_member, path=tmpdir)
            extracted = Path(tmpdir) / sqlite_member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted), dest)
    LOGGER.info("Extracted SQLite DB to %s", dest)


def normalize_release(release: str) -> str:
    return release.lower().replace("chembl_", "").strip()


def fetch_latest_release() -> str:
    index_url = f"{BASE_URL}/"
    LOGGER.info("Checking latest release from %s", index_url)
    response = requests.get(index_url, timeout=30)
    response.raise_for_status()
    matches = re.findall(r"chembl_(\d+)/", response.text, flags=re.IGNORECASE)
    if not matches:
        raise RuntimeError("Could not determine latest ChEMBL release.")
    latest = max(matches, key=lambda x: int(x))
    LOGGER.info("Latest available release: chembl_%s", latest)
    return latest


def detect_local_release(output_dir: Path) -> Tuple[Optional[str], Optional[Path]]:
    if not output_dir.exists():
        return None, None
    candidates = sorted(output_dir.glob("chembl_*.db"))
    latest_version: Optional[str] = None
    latest_path: Optional[Path] = None
    for candidate in candidates:
        match = re.search(r"chembl_(\d+)\.db", candidate.name, re.IGNORECASE)
        if not match:
            continue
        version = match.group(1)
        if latest_version is None or int(version) > int(latest_version):
            latest_version = version
            latest_path = candidate
    return latest_version, latest_path


def download_release(
    release: str,
    output_dir: Path,
    keep_archive: bool,
    skip_existing: bool,
) -> Path:
    release = normalize_release(release)
    archive_name = f"chembl_{release}_sqlite.tar.gz"
    db_name = f"chembl_{release}.db"
    url = f"{BASE_URL}/chembl_{release}/{archive_name}"

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / archive_name
    db_path = output_dir / db_name

    if skip_existing and db_path.exists():
        LOGGER.info("Database already exists at %s; skipping download.", db_path)
        return db_path

    LOGGER.info("Fetching %s", url)
    _download_file(url, archive_path)

    LOGGER.info("Extracting SQLite DB to %s", db_path)
    _extract_sqlite_from_tar(archive_path, db_path)

    if not keep_archive:
        archive_path.unlink(missing_ok=True)

    return db_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ChEMBL SQLite dumps.")
    parser.add_argument(
        "--release",
        help="ChEMBL release number (e.g., 34). Omit to fetch the newest available.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store the dump (default: data/chembl_releases)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the tar.gz archive after extraction",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if the decompressed DB already exists.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Download without asking when a newer release is detected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        target_release = (
            normalize_release(args.release) if args.release else fetch_latest_release()
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Unable to determine release: %s", exc)
        raise SystemExit(1) from exc

    local_release, local_path = detect_local_release(output_dir)
    if local_release:
        if int(local_release) == int(target_release):
            LOGGER.info("chembl_%s.db already present at %s; nothing to do.", local_release, local_path)
            return
        if int(local_release) > int(target_release):
            LOGGER.info(
                "Local chembl_%s.db is newer than requested chembl_%s; skipping download.",
                local_release,
                target_release,
            )
            return
        if not args.yes:
            prompt = input(
                f"A newer release (chembl_{target_release}) is available. "
                f"Download and replace local chembl_{local_release}.db? [y/N]: "
            ).strip().lower()
            if prompt not in {"y", "yes"}:
                LOGGER.info("Aborted by user.")
                return

    try:
        db_path = download_release(
            target_release,
            output_dir=output_dir,
            keep_archive=args.keep_archive,
            skip_existing=args.skip_existing,
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Download failed: %s", exc)
        raise SystemExit(1) from exc

    LOGGER.info("ChEMBL SQLite ready at %s", db_path)


if __name__ == "__main__":
    main()
