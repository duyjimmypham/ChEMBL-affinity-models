# src/score_smiles.py
"""
Score a single SMILES string with the best trained model.

Usage:
    python -m src.score_smiles "CCOc1ccccc1"

Notes:
- Make sure you've already run src.pipeline to train and save a model.
- Update MODEL_PATH below to point to the best model you saved
  (e.g. models/xgboost_best.pkl or models/random_forest_best.pkl).
"""

import sys
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# <<< CHANGE THIS if your best model is different >>>
MODEL_PATH = "models/random_forest_best.pkl"

N_BITS = 2048
RADIUS = 2


def smiles_to_morgan(smiles: str, n_bits: int = N_BITS, radius: int = RADIUS):
    """Convert a SMILES string to a Morgan fingerprint compatible with training."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main(smiles_str: str):
    # load model
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Model not found at {MODEL_PATH}. Run src.pipeline first.")
        return

    # featurize
    fp = smiles_to_morgan(smiles_str)
    if fp is None:
        print("Invalid SMILES string. Could not parse with RDKit.")
        return

    # predict
    proba = model.predict_proba([fp])[0, 1]
    print(f"SMILES: {smiles_str}")
    print(f"Predicted activity probability: {proba:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.score_smiles \"CCOc1ccccc1\"")
    else:
        main(sys.argv[1])
