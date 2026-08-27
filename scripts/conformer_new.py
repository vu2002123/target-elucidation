from pathlib import Path
from dimorphite_dl import protonate_smiles
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy

smiles_str = "CN1CCN(CC1)CCCN2C3=CC=CC=C3SC4=C2C=C(C=C4)C(F)(F)F"
mol_name = "Trifluoperazine"
WORK_DIR = Path.home() / "target-elucidation/data"
RAW_DIR = WORK_DIR / "raw"
INTERIM_DIR = WORK_DIR / "interim"

protonated_mol: list[str] = protonate_smiles(smiles_str, ph_min=6.8, ph_max=7.4)
print(f"Protonated: {protonated_mol}")

params = AllChem.ETKDGv3()

pdbqt_dir = INTERIM_DIR / mol_name
pdbqt_dir.mkdir(exist_ok=True, parents=True)
output_filename = pdbqt_dir / f"{mol_name}_protonated.sdf"

# 2. Open the SDWriter using a context manager
with Chem.SDWriter(output_filename) as writer:
    for index, smi in enumerate(protonated_mol):
        # Step A: Convert SMILES to Mol object
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Warning: Could not parse SMILES '{smi}'. Skipping.")
            continue

        # Step B: Add explicit hydrogens (CRITICAL for accurate 3D geometry)
        mol_with_H = Chem.AddHs(mol)

        # Step C: Generate 3D coordinates
        # EmbedMolecule generates a single conformation. Returns -1 if it fails.
        embed_status = AllChem.EmbedMolecule(mol_with_H)
        if embed_status == -1:
            print(f"Warning: Could not generate 3D coords for '{smi}'. Skipping.")
            continue
        # Step E: Assign a name to the molecule (appears as the title in the SDF)
        mol_with_H.SetProp("_Name", f"{mol_name}_{index + 1}")

        # Step F: Write the molecule to the file
        writer.write(mol_with_H)


for mol in Chem.SDMolSupplier(output_filename, removeHs=False):
    mk_prep = MoleculePreparation()
    state_name = mol.GetProp("_Name")
    molsetup_list = mk_prep(mol)
    molsetup = molsetup_list[0]
    pdbqt_string = PDBQTWriterLegacy.write_string(molsetup)
    print(pdbqt_string[0])
    pdbqt_file = pdbqt_dir / f"{state_name}.pdbqt"
    with open(pdbqt_file, "w") as file:
        file.write(pdbqt_string[0])
