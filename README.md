# ChEMBL Activity Prediction Pipeline

Predict compound bioactivity for a given ChEMBL target using open data and classical machine learning.

---

## Overview

This project demonstrates an end-to-end cheminformatics ML workflow:

1. **Fetch** assay data for one ChEMBL target (e.g., EGFR = CHEMBL203)  
2. **Clean & label** IC50 → pIC50 → active/inactive  
3. **Featurize** molecules as Morgan fingerprints (RDKit)  
4. **Split** data by molecular scaffold for realistic evaluation  
5. **Train & tune** Logistic Regression, Random Forest, and XGBoost with 5-fold CV  
6. **Evaluate** ROC-AUC / PR-AUC on held-out scaffolds  
7. **Save** best models to `models/`

---

## Quick start

```bash
git clone https://github.com/yourusername/chembl-activity-ml.git
cd chembl-activity-ml
pip install -r requirements.txt
python -m src.pipeline
