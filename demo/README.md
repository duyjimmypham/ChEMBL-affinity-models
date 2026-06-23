
# ChEMBL Prediction Demo

Animated visualization of the molecule-scoring workflow used by this project.

## Running the demo

```bash
npm install
npm run dev
```

Create model-specific demo assets from the repository root:

```bash
python src/demo.py --target CHEMBL203 --smiles "CC(=O)Oc1ccccc1C(=O)O"
```
