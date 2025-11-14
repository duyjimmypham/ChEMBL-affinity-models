#!/usr/bin/env python3
"""
Score a single SMILES string with the latest trained model.

Usage:
    python src/score_single.py "CCO" --target CHEMBL203
"""

from __future__ import annotations

import argparse
import sys

from scoring_utils import score_smiles as score_smiles_entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles", help="SMILES string to score")
    parser.add_argument(
        "--target",
        help="ChEMBL ID used for training (e.g., CHEMBL1075091). "
        "Falls back to global best model if omitted.",
    )
    args = parser.parse_args()

    proba, model_path, error = score_smiles_entry(args.smiles, target_id=args.target)
    if error:
        print(f"Scoring failed: {error}")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"SMILES: {args.smiles}")
    print(f"Predicted activity probability: {proba:.3f}")


if __name__ == "__main__":
    main()
