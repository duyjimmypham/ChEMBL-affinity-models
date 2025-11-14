import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, DataStructs

# Helper: convert SMILES to fingerprint bits
def smiles_to_fp(smi, radius, n_bits=256):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Molecules to compare
molecules = {
    "Ethanol": "CCO",
    "Phenol": "c1ccccc1O"
}

radii = [1, 2, 3]
n_bits = 256  # smaller for plotting clarity

fig, axes = plt.subplots(len(molecules), len(radii), figsize=(10, 4))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

for i, (name, smi) in enumerate(molecules.items()):
    mol = Chem.MolFromSmiles(smi)
    Draw.MolToFile(mol, f"{name}.png", size=(150, 150))
    for j, r in enumerate(radii):
        fp = smiles_to_fp(smi, r, n_bits)
        axes[i, j].imshow(fp.reshape(16, 16), cmap="Greens", interpolation="nearest")
        axes[i, j].set_title(f"{name}\nRadius={r}")
        axes[i, j].axis("off")

plt.suptitle("Morgan Fingerprint Visualization (Ethanol vs Phenol)")
plt.show()
