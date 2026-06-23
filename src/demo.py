"""Generate the molecule image and prediction data used by the React demo."""

import argparse
import json
import sys
from pathlib import Path
from typing import Union

from rdkit import Chem
from rdkit.Chem import Draw

from scoring_utils import score_smiles

# Configuration
DEMO_PUBLIC_DIR = Path("demo/public")
MOLECULE_IMG_PATH = DEMO_PUBLIC_DIR / "molecule.png"
DATA_JSON_PATH = DEMO_PUBLIC_DIR / "data.json"


def generate_molecule_image(smiles: str, output_path: Path) -> None:
    """Generate a transparent PNG representation of a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    img = Draw.MolToImage(mol, size=(300, 300))
    img = img.convert("RGBA")
    pixels = []
    for pixel in img.getdata():
        if pixel[0] > 240 and pixel[1] > 240 and pixel[2] > 240:
            pixels.append((255, 255, 255, 0))
        else:
            pixels.append(pixel)
    img.putdata(pixels)
    img.save(output_path, "PNG")
    print(f"Saved molecule image to {output_path}")


def update_data_json(prediction: Union[float, str], output_path: Path) -> None:
    """Write a display-ready activity probability to the demo data file."""
    if isinstance(prediction, float):
        value = f"{prediction:.3f}"
    else:
        value = str(prediction)

    data = {"prediction": value}
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2)
    print(f"Saved prediction data to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update React demo assets.")
    parser.add_argument("--target", required=True, help="ChEMBL Target ID")
    parser.add_argument("--smiles", required=True, help="SMILES string")
    args = parser.parse_args()

    DEMO_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing {args.target} with SMILES: {args.smiles}")

    try:
        proba, _, error = score_smiles(args.smiles, target_id=args.target)
        if error:
            print(f"Error scoring: {error}")
            sys.exit(1)
        if proba is None:
            raise RuntimeError("Model returned no probability.")
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)

    try:
        generate_molecule_image(args.smiles, MOLECULE_IMG_PATH)
    except Exception as e:
        print(f"Image generation failed: {e}")
        sys.exit(1)

    try:
        update_data_json(proba, DATA_JSON_PATH)
    except Exception as e:
        print(f"JSON update failed: {e}")
        sys.exit(1)

    print("Assets updated successfully.")


if __name__ == "__main__":
    main()
