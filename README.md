# 🧬 ChEMBL Activity Prediction Pipeline

A fully automated cheminformatics machine learning pipeline to predict **compound bioactivity** for any **ChEMBL target**.  
It integrates data fetching, curation, feature generation, scaffold-aware evaluation, and model deployment — fully reproducible from the command line.

---

## 🌐 Project Goals

This repository provides a modular and transparent workflow for:

- Building ML-ready datasets directly from the ChEMBL API  
- Cleaning and standardizing biological assay data  
- Detecting the dominant activity type (IC50 / EC50 / Ki / Kd)  
- Generating molecular fingerprints for QSAR-style modeling  
- Performing scaffold-based splits for realistic evaluation  
- Training and comparing classical ML models (LogReg, RF, XGBoost)  
- Saving the best model and associated metrics automatically  
- Scoring new molecules or databases without retraining  

---

## 📦 Repository Structure

```
project_root/
│
├── pipeline.py                  # Main training entry point
├── inspect_chembl.py            # Summarizes assay info for a ChEMBL target
├── score_smiles.py              # Batch or single-molecule scoring
│
├── fingerprint_visualization.py # Optional: fingerprint exploration
├── tanimoto_demo.py             # Optional: Tanimoto similarity demo
│
├── data/                        # Cached raw data & summaries
│   ├── CHEMBLxxxx_activities.db
│   └── CHEMBLxxxx_summary.json
│
├── models/                      # Serialized models (.joblib)
│   └── CHEMBLxxxx_random_forest.joblib
│
└── results/                     # Evaluation reports & logs
    ├── CHEMBLxxxx_metrics.json
    └── best_model.txt
```

---

## 🚀 Quick Start

### 1️⃣ Install dependencies

```bash
pip install rdkit-pypi scikit-learn xgboost joblib pandas numpy requests tqdm
```

> 🧩 **Note:** XGBoost is optional — the pipeline automatically skips it if not installed.

---

### 2️⃣ Inspect a target (optional but recommended)

Fetch activity data and summarize assay types:

```bash
python inspect_chembl.py CHEMBL1075091
```

Creates:
- `data/CHEMBL1075091_activities.db`
- `data/CHEMBL1075091_summary.json` (e.g. dominant_type: "IC50")

---

### 3️⃣ Train models

```bash
python pipeline.py CHEMBL1075091
```

This automatically:
- Loads cached or freshly fetched assay data  
- Filters by allowed activity types and numeric `nM` values  
- Labels actives (pActivity ≥ 6.0)  
- Converts SMILES → Morgan fingerprints (2048 bits, radius 2)  
- Splits data by Bemis–Murcko scaffold (80/20)  
- Computes train–test Tanimoto overlap (chemical leakage summary)  
- Trains Logistic Regression, Random Forest, and XGBoost (5-fold CV)  
- Evaluates on held-out scaffolds  
- Saves metrics and best model path to `results/`  

---

### 4️⃣ Score new molecules

#### (a) Single SMILES

```bash
python score_smiles.py "CCOc1ccccc1" --target CHEMBL1075091
```

Automatically locates and uses the best model for that target (or falls back to `results/best_model.txt`).

#### (b) CSV file of SMILES

```bash
python score_smiles.py --input smiles.csv --output scored.csv
```

Expected columns:

| smiles | target_id (optional) |
|---------|---------------------|

Output columns:

| smiles | target_id | pred_proba | model_path_used | scoring_error |

#### (c) SQLite database

```bash
python score_smiles.py --input compounds.db --table molecules --output scored.csv
```

The table must include a `smiles` column (and optionally `target_id`).

---

## ⚙️ Pipeline Design

### Data Curation
- Filters to `standard_relation == "="` and `standard_units == "nM"`  
- Keeps allowed activity types (`IC50`, `EC50`, `Ki`, `Kd`)  
- Uses dominant type from `*_summary.json` if available  
- Computes `pActivity = 9 − log10(nM)` and labels active if ≥ 6.0  

### Feature Generation
- Morgan fingerprints (`radius=2`, `nBits=2048`) via RDKit  
- Invalid SMILES automatically skipped  

### Scaffold Split
- Bemis–Murcko scaffold grouping → 80/20 train/test  
- Prevents scaffold leakage and inflated performance  

### Chemical Leakage Summary
- Computes test-to-train Tanimoto similarities  
- Reports mean, median, and 90th percentile of maximum overlaps  

### Model Families

| Model | Library | Description |
|--------|----------|-------------|
| Logistic Regression | scikit-learn | Simple linear baseline |
| Random Forest | scikit-learn | Strong baseline ensemble |
| XGBoost | xgboost | Gradient boosting (optional) |

Each trained with 5-fold cross-validation and evaluated by test ROC-AUC.

### Output Metrics
`results/{ID}_metrics.json` example:

```json
{
  "target_id": "CHEMBL1075091",
  "tanimoto": { "median_max_test_to_train": 0.42 },
  "models": {
    "random_forest": { "cv_roc_auc_mean": 0.91, "test_roc_auc": 0.88 },
    "logreg": { "cv_roc_auc_mean": 0.79 },
    "xgboost": { "cv_roc_auc_mean": 0.93, "test_roc_auc": 0.89 }
  },
  "best_model": {
    "name": "random_forest",
    "model_path": "models/CHEMBL1075091_random_forest.joblib"
  }
}
```

---

## 🧮 Example Workflow

```bash
# Inspect new target
python inspect_chembl.py CHEMBL203

# Train models
python pipeline.py CHEMBL203

# Score new library
python score_smiles.py --input screening_hits.csv --output predictions.csv --target CHEMBL203
```

---

## 📊 Visualization Utilities

- **`fingerprint_visualization.py`** — visualize bit patterns and overlaps between molecules  
- **`tanimoto_demo.py`** — show pairwise Tanimoto similarities for toy molecules  

These are optional and do not affect pipeline training.

---

## 🧱 Reproducibility

- Deterministic runs under fixed random seeds  
- Cached datasets allow re-runs without repeated API calls  
- All models and metrics are versioned by target ID  

---

## 🧩 Extensibility

- Swap fingerprints (ECFP4, MACCS) in `featurization.py`  
- Add new ML models in `modeling.py`  
- Modify thresholds or split ratios via a config file (planned)  
- Integrate deep-learning models (e.g., GNNs) using the same scaffold split  

---

## 🧠 Troubleshooting

| Issue | Likely Cause | Solution |
|-------|---------------|-----------|
| Empty dataset | No valid `canonical_smiles` or numeric activities | Use `inspect_chembl.py` to check; try another target |
| “XGBoost not found” | Missing library | `pip install xgboost` or ignore |
| Poor performance | Too few actives/inactives | Expand dataset or relax pActivity threshold |
| Slow leakage check | Large dataset | Subsample in `tanimoto_leakage_summary()` |

---

## 📚 References

- Bento et al., *Nucleic Acids Research* (2014). “The ChEMBL bioactivity database: an update.”  
- Rogers & Hahn, *J. Chem. Inf. Model.* (2010). “Extended-Connectivity Fingerprints.”  
- Bemis & Murcko, *J. Med. Chem.* (1996). “Molecular Frameworks.”  

---

## 🧾 License

MIT License — free for academic and commercial use.

---

## 🤝 Contributions

Pull requests are welcome!  
If you add a new model or feature, include:
- Example or test script  
- Documentation updates  

---

## ✨ Acknowledgments

Built using **ChEMBL**, **RDKit**, and **scikit-learn**.  
Designed for open, reproducible cheminformatics research and education.
