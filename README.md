# ChEMBL Affinity Models

End-to-end tooling for building target-specific bioactivity classifiers from the [ChEMBL](https://www.ebi.ac.uk/chembl/) knowledge base. The project covers data ingestion, molecule-level curation, scaffold-aware model training, scoring utilities, and simple diagnostics/plots.

---

## Features

- **Centralized configuration** – `src/config.py` defines all constants, paths, and model hyperparameters in one place for easy experimentation.
- **Modular featurization** – `src/features.py` provides optimized fingerprint generation with both single-molecule and vectorized batch processing.
- **Data acquisition** – `src/inspect_chembl.py` fetches assays/molecules/mechanisms per target, either from the public API (with batching/checkpoints) or from a local SQLite dump.
- **Molecule-level aggregation** – the training pipeline collapses replicate measurements, applies absolute activity thresholds (with a quantile fallback when necessary), and emits warnings for tiny or imbalanced datasets.
- **Model training** – `src/pipeline.py` featurizes molecules (RDKit Morgan fingerprints), performs Bemis–Murcko scaffold splits, and tunes Logistic Regression, Random Forest, and optional XGBoost models via cross‑validated grid search.
- **Optimized batch scoring** – `src/score_batch.py` uses vectorized operations for 10x-100x speedup on large datasets.
- **Visual demo** – `src/demo.py` generates assets for the React-based portfolio animation in `demo/`.
- **Diagnostics** – metrics JSON files include ROC/PR curves, confusion matrices, labeling strategy, dataset warnings, and are easily visualized via `scripts/plot_metrics.py`.

---

## Requirements

- Python **3.11**
- The Python packages listed in [`requirements.txt`](requirements.txt). Install into a virtual environment:

```bash
python -m venv .venv
. .venv/Scripts/activate        # PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> **RDKit compatibility:** keep NumPy `< 2` and SciPy `< 1.12` (as reflected in `requirements.txt`) so the prebuilt `rdkit-pypi` wheels stay importable.

---

## Quick Start

### 1. (Optional) Download a local ChEMBL dump

Having the SQLite release locally makes repeated runs much faster and keeps you offline.

```bash
python src/chembl_downloader.py --release 36 --output data/chembl_releases --skip-existing
```

`inspect_chembl.py`/`pipeline.py` will auto-detect `data/chembl_releases/chembl_*.db` (or you can pass `--chembl-sqlite path/to/db`).

### 2. Inspect a target

```bash
python src/inspect_chembl.py CHEMBL1075091 --fast
```

This fetches assays, molecules, and mechanisms; writes `data/{target}_*.db/csv/json`; and caches metadata in `data/meta/`.

### 3. Train models

```bash
python src/pipeline.py CHEMBL1075091
```

Highlights:

- Molecule-level aggregation with absolute thresholds (≥6.0 active / ≤4.5 inactive) plus a quantile fallback when only one class remains.
- Dataset suitability warnings (`too_few_molecules`, `too_few_per_class`, `extreme_imbalance`) logged and recorded in the metrics JSON.
- Bemis–Murcko scaffold split, 5-fold grid search, ROC/PR curve storage.
  Outputs land in `models/` (pickles) and `results/` (metrics JSON + `best_model.txt` pointer).

### 4. Score molecules

Single SMILES:

```bash
python src/score_single.py "CCOC(=O)N" --target CHEMBL1075091
```

Batch CSV (optimized with vectorization):

```bash
python src/score_batch.py --input data/my_smiles.csv --output scored.csv --target-id CHEMBL1075091
```

For advanced batching (CSV or SQLite with per-row target IDs) use `src/score_smiles.py`.

### 5. Visual Demo (React App)

The project includes a high-quality React-based animation in the `demo/` folder. To use it with your model's predictions:

1.  **Update Assets**: Run the Python script to generate the molecule image and prediction data.
    ```bash
    python src/demo.py --target CHEMBL203 --smiles "CC(=O)Oc1ccccc1C(=O)O"
    ```
2.  **Run Demo**: Navigate to the `demo` folder and start the web app.
    ```bash
    cd demo
    npm install
    npm run dev
    ```
    The animation will now display your specific molecule and prediction.

### 6. Plot metrics

```bash
python scripts/plot_metrics.py CHEMBL1075091 --model log_reg
```

PNG files are saved in `results/{target}_plots/` for quick sharing.

> For a more detailed, copy/pasteable workflow (including sanity tips) see [`docs/TESTING.md`](docs/TESTING.md).

---

## Repository Layout

```
.
├── archive/                 # Retired helper notebooks (kept for reference)
├── data/
│   ├── chembl_releases/     # Local ChEMBL SQLite dumps (auto-detected)
│   ├── meta/                # Per-target metadata (last_updated, row counts)
│   └── CHEMBLxxxx_*         # Cached activities/molecules/mechanisms
├── docs/TESTING.md          # Hands-on testing checklist
├── models/                  # Trained model artifacts (.pkl)
├── results/                 # Metrics JSON + ROC/PR plots + best_model.txt
├── scripts/plot_metrics.py  # ROC/PR visualizer
├── src/
│   ├── config.py            # Centralized configuration (NEW)
│   ├── features.py          # Fingerprint generation utilities (NEW)
│   ├── chembl_cache.py      # Cache metadata helpers
│   ├── chembl_client_utils.py
│   ├── chembl_downloader.py
│   ├── demo.py              # React demo asset generator (NEW)
│   ├── inspect_chembl.py
│   ├── local_chembl.py
│   ├── pipeline.py          # Main training pipeline
│   ├── score_batch.py       # Optimized batch scoring
│   ├── score_single.py
│   ├── score_smiles.py
│   └── scoring_utils.py
└── examples/                # Sample metrics/plots for two demo targets
```

---

## Recent Improvements (Phase 1)

### Code Organization

- **`src/config.py`**: All configuration constants, thresholds, and model hyperparameters now centralized for easy tuning
- **`src/features.py`**: Single source of truth for molecular fingerprint generation, eliminating code duplication

### Performance Optimization

- **Vectorized batch scoring**: `score_batch.py` now processes entire datasets at once instead of row-by-row iteration
- **Expected speedup**: 10x-100x faster for large molecule libraries

### Developer Experience

- **Google-style docstrings**: All major functions now documented
- **Type hints**: Improved IDE support and code clarity
- **sklearn compatibility**: Updated to latest API standards

### Portfolio Features

- **React Integration**: `src/demo.py` bridges the Python backend with a high-quality React animation for portfolio showcasing

---

## Dataset Warnings & Labeling

- **Absolute thresholds:** actives are `p_activity ≥ 6.0` (≤1 μM) and inactives are `≤ 4.5` (≥30 μM). Gray-zone molecules are dropped.
- **Quantile fallback:** if only one class remains, the pipeline automatically labels molecules via within-target quantiles (default 30% vs 70%) and records the chosen thresholds in the metrics file.
- **Warnings:** Suitability checks append flags to `dataset_warnings` inside each metrics JSON and emit log messages. Inspect these before trusting a model trained on extremely small or imbalanced datasets.

---

## Contributing / Next Steps

- File issues or PRs for new model architectures, visualization ideas, or UI integrations.
- See `docs/TESTING.md` for regression testing ideas before submitting changes.

---

## License

MIT License – feel free to use this codebase in academic or commercial settings. See [`LICENSE`](LICENSE) for details.
