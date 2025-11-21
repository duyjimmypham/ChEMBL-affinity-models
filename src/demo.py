"""
Updates assets for the React demo (demo/public).
Generates molecule.png and data.json based on model prediction.
"""
import argparse
import json
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw

from scoring_utils import score_smiles

# Configuration
DEMO_PUBLIC_DIR = Path("demo/public")
MOLECULE_IMG_PATH = DEMO_PUBLIC_DIR / "molecule.png"
DATA_JSON_PATH = DEMO_PUBLIC_DIR / "data.json"

def generate_molecule_image(smiles: str, output_path: Path):
    """Generates a transparent PNG of the molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string")
    
    # Create image with transparent background
    # RDKit Draw.MolToImage returns a PIL Image
    img = Draw.MolToImage(mol, size=(300, 300))
    
    # Make background transparent (assuming white background from RDKit)
    img = img.convert("RGBA")
    datas = img.getdata()
    new_data = []
    for item in datas:
        # If pixel is white (or close to it), make it transparent
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)
    
    img.save(output_path, "PNG")
    print(f"Saved molecule image to {output_path}")

def update_data_json(prediction, output_path: Path):
    """Updates the data.json file with the prediction."""
    # If prediction is float, format it. If string, use as is.
    if isinstance(prediction, float):
        val = f"{prediction:.2f}"
    else:
        val = str(prediction)
        
    data = {
        "prediction": val,
        "timestamp": "now" # Placeholder
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved prediction data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Update React demo assets.")
    parser.add_argument("--target", required=True, help="ChEMBL Target ID")
    parser.add_argument("--smiles", required=True, help="SMILES string")
    args = parser.parse_args()
    
    # Ensure public dir exists
    DEMO_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {args.target} with SMILES: {args.smiles}")
    
    # 1. Get Prediction
    try:
        proba, _, error = score_smiles(args.smiles, target_id=args.target)
        if error:
            print(f"Error scoring: {error}")
            sys.exit(1)
            
        # Convert probability to pIC50-like score for demo purposes if needed
        # The user asked for "predicted affinity pIC50". 
        # Our models predict *probability of activity*.
        # We can simulate a pIC50 or just show the probability.
        # Let's show probability for now, or map it? 
        # User asked for "Predicted pIC50: 7.45" in the prompt, but our model is a classifier.
        # Let's stick to the probability but label it clearly, or map 0-1 to a 4-9 range for visual effect?
        # Better to be honest: "Prob: 0.XX". 
        # BUT, GraphNode.tsx has a graph that looks like pIC50 (values ~8.2).
        # Let's map prob 0.0-1.0 to pIC50 4.0-9.0 for the visual demo effect.
        # This is a DEMO, so visual consistency with the graph (y-axis 40-140?) is key.
        # Wait, GraphNode.tsx data points are y=140..40. 
        # Let's just pass the probability formatted nicely.
        
        # actually, let's just pass the string value.
        prediction_display = f"{proba:.3f}"
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)

    # 2. Generate Image
    try:
        generate_molecule_image(args.smiles, MOLECULE_IMG_PATH)
    except Exception as e:
        print(f"Image generation failed: {e}")
        sys.exit(1)
        
    # 3. Update JSON
    try:
        update_data_json(proba, DATA_JSON_PATH) # Passing raw proba, frontend formats it? 
        # Actually frontend expects a string in "prediction".
        # Let's pass a formatted string.
        # If proba > 0.5, it's active.
        update_data_json(prediction_display, DATA_JSON_PATH)
    except Exception as e:
        print(f"JSON update failed: {e}")
        sys.exit(1)
        
    print("Assets updated successfully!")

if __name__ == "__main__":
    main()
