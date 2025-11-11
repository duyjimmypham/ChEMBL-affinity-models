# 🧬 ChEMBL-Affinity-Models

A modular cheminformatics + machine learning pipeline for **target-specific bioactivity prediction** using **ChEMBL** data.  
It automates data retrieval, curation, feature generation, scaffold-aware training, and scoring — fully reproducible from the command line.

---

## 🌐 Project Overview

This repository enables:

- Building ML-ready datasets directly from the ChEMBL API  
- Cleaning, filtering, and labeling biological assays (IC50 / EC50 / Ki / Kd)  
- Detecting the dominant activity type per target  
- Featurizing SMILES into Morgan fingerprints  
- Performing Bemis–Murcko scaffold splits for realistic evaluation  
- Training Logistic Regression, Random Forest, and optional XGBoost models  
- Excluding reference ligands (clinical ≥ Phase 2 or mechanism-linked) from training  
- Scoring single or batch molecules against trained models  

---

## 📦 Repository Structure

```
project_root/
│
├── src/
│   ├── pipeline.py          # Main training entry point
│   ├── inspect_chembl.py    # Fetches ChEMBL data; builds target datasets
│   ├── score_single.py      # Score a single SMILES
│   ├── score_batch.py       # Score multiple SMILES (CSV or DB)
│
├── data/
│   ├── CHEMBLxxxx_activities.db
│   ├── CHEMBLxxxx_molecules.csv
│   ├── CHEMBLxxxx_mechanisms.csv
│   └── CHEMBLxxxx_summary.json
│
├── models/                  # Trained models (.joblib)
│   └── CHEMBLxxxx_random_forest.joblib
│
└── results/                 # Metrics & logs
    ├── CHEMBLxxxx_metrics.json
    └── best_model.txt
```

---

## 🚀 Quick Start

### 1️⃣ Install dependencies

```bash
pip install rdkit-pypi scikit-learn xgboost joblib pandas numpy requests tqdm
```
> XGBoost is optional — the pipeline skips it if missing.

---

### 2️⃣ Inspect a target

Fetch assays, molecules, and mechanisms for any ChEMBL target:

```bash
python src/inspect_chembl.py CHEMBL1075091
```

Creates:
- `data/CHEMBL1075091_activities.db`  
- `data/CHEMBL1075091_molecules.csv`  
- `data/CHEMBL1075091_mechanisms.csv`  
- `data/CHEMBL1075091_summary.json`

---

### 3️⃣ Train models

```bash
python src/pipeline.py CHEMBL1075091
```

Performs:
- Load / fetch ChEMBL assays  
- Filter (nM, “=”, IC50/EC50/Ki/Kd)  
- Label actives (`pActivity ≥ 6.0`)  
- Exclude phase ≥ 2 or mechanism-linked ligands  
- Morgan fingerprints (radius 2, 2048 bits)  
- Scaffold split (80/20) + leakage check  
- 5-fold CV training (LogReg, RF, XGBoost)  
- Evaluation on held-out scaffolds  
- Write metrics + best model path  

---

### 4️⃣ Score new molecules

#### Single SMILES
```bash
python src/score_single.py "CCOc1ccccc1" --target CHEMBL1075091
```

#### Batch (CSV or DB)
```bash
python src/score_batch.py --input smiles.csv --output scored.csv --target CHEMBL1075091
```

Expected input column: `smiles` (and optional `target_id`).

Outputs prediction probabilities and model path used.

---

## ⚙️ Pipeline Design

- **Curation:** keeps numeric nM values with `=` relation  
- **Featurization:** RDKit Morgan FP (2048 bits, radius 2)  
- **Scaffold Split:** Bemis–Murcko (80/20)  
- **Metrics:** CV ROC-AUC + test ROC-AUC by model  
- **Caching:** reuses activities.db and summary JSON to avoid repeated API calls  
- **Reference Exclusion:** removes ligands with max_phase ≥ 2 or known mechanisms  

---

## 🧱 Reproducibility

- Deterministic seeds and scaffold splits  
- Cached datasets for repeatable runs  
- Models & metrics versioned by target ID  

---

## 📚 References

- Bento et al., *Nucleic Acids Res.* 2014 — ChEMBL database  
- Rogers & Hahn, *J. Chem. Inf. Model.* 2010 — ECFP fingerprints  
- Bemis & Murcko, *J. Med. Chem.* 1996 — Scaffold frameworks  

---

## 🧾 License
MIT License — free for academic and commercial use.
