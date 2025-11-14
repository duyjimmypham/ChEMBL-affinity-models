## Quick Verification Checklist

1. **Environment**
   - `python -m venv .venv && .\.venv\Scripts\activate`
   - `pip install -r requirements.txt`

2. **Download a local ChEMBL snapshot (optional but recommended)**
   ```bash
   python src/chembl_downloader.py --release 34 --skip-existing
   ```
   Place the resulting `chembl_34.db` under `data/chembl_releases/` (the tools auto-detect the newest file here, or honor `CHEMBL_SQLITE_PATH`).

3. **Fast inspection (activities only)**
   ```bash
   python src/inspect_chembl.py CHEMBL1075091 --fast --skip-update-check
   ```
   Confirms the resumable fetch, progress bars, and cache metadata files under `data/meta/`.

4. **Inspection against the local SQLite dump**
   ```bash
   python src/inspect_chembl.py CHEMBL1075091 --chembl-sqlite data/chembl_releases/chembl_34.db
   ```
   Ensures API-free mode produces the same artifacts in `data/`.

5. **Pipeline training using the cached dataset**
   ```bash
   python src/pipeline.py CHEMBL1075091
   ```
   Verify that `results/CHEMBL1075091_metrics.json` and `models/*.pkl` are refreshed only when new assays exist (check logs for “Using cached activities…”).
