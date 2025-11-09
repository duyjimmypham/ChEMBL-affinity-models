from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def morgan_fp(smiles: str, radius: int, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)

# two example molecules
smi_a = "CCO"           # ethanol
smi_b = "c1ccccc1O"     # phenol

radii = [1, 2, 3]
n_bits = 2048

print(f"Comparing:\n  A: {smi_a}\n  B: {smi_b}\n")

for r in radii:
    fp_a = morgan_fp(smi_a, r, n_bits)
    fp_b = morgan_fp(smi_b, r, n_bits)
    # guard in case of bad SMILES
    if fp_a is None or fp_b is None:
        print(f"Radius {r}: could not generate fingerprints.")
        continue
    sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
    print(f"Radius {r}: Tanimoto = {sim:.3f}")
