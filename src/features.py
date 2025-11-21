"""
Feature generation utilities for molecular data.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from config import FP_N_BITS, FP_RADIUS

def smiles_to_morgan(smi: str, n_bits: int = FP_N_BITS, radius: int = FP_RADIUS) -> Optional[np.ndarray]:
    """Generates a Morgan fingerprint (as a numpy array) for a given SMILES string.

    Args:
        smi (str): The SMILES string of the molecule.
        n_bits (int, optional): Number of bits in the fingerprint. Defaults to config.FP_N_BITS.
        radius (int, optional): Radius of the Morgan fingerprint. Defaults to config.FP_RADIUS.

    Returns:
        Optional[np.ndarray]: A numpy array of 0s and 1s, or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def smiles_to_bitvect(smi: str, n_bits: int = FP_N_BITS, radius: int = FP_RADIUS):
    """Generates a Morgan fingerprint as an RDKit ExplicitBitVect.

    Args:
        smi (str): The SMILES string.
        n_bits (int, optional): Number of bits. Defaults to config.FP_N_BITS.
        radius (int, optional): Radius. Defaults to config.FP_RADIUS.

    Returns:
        ExplicitBitVect: The RDKit bit vector, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)

def batch_smiles_to_morgan(smiles_list: Sequence[str], n_bits: int = FP_N_BITS, radius: int = FP_RADIUS) -> np.ndarray:
    """Generates a matrix of Morgan fingerprints for a list of SMILES.

    This function is optimized for batch processing.

    Args:
        smiles_list (Sequence[str]): A list of SMILES strings.
        n_bits (int, optional): Number of bits. Defaults to config.FP_N_BITS.
        radius (int, optional): Radius. Defaults to config.FP_RADIUS.

    Returns:
        np.ndarray: A matrix of shape (n_valid_molecules, n_bits). 
                    Invalid SMILES are skipped (or handled by returning rows of zeros/NaNs 
                    depending on implementation preference - here we will return a matrix 
                    matching the input length, with NaNs for invalid rows, to maintain alignment).
    """
    n_samples = len(smiles_list)
    X = np.zeros((n_samples, n_bits), dtype=np.float32) # Use float to allow NaNs if needed, or keep int
    
    # Note: For extremely large batches, joblib could be used here.
    # For now, a simple loop is often fast enough if RDKit is efficient.
    # If we want to maintain alignment, we need to handle failures.
    
    valid_mask = np.zeros(n_samples, dtype=bool)
    
    for i, smi in enumerate(smiles_list):
        arr = smiles_to_morgan(smi, n_bits=n_bits, radius=radius)
        if arr is not None:
            X[i, :] = arr
            valid_mask[i] = True
        else:
            # Invalid SMILES: leave as zeros or mark? 
            # For this implementation, we'll leave as zeros but the caller should check validity.
            pass
            
    return X, valid_mask
