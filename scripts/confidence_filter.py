from pathlib import Path
from typing import NamedTuple, List
import gemmi
import pandas as pd
import numpy as np


def get_plddt_gemmi(file_path):
    # Read the MMCIF file
    doc = gemmi.cif.read_file(str(file_path))
    block = doc.sole_block()
    st = gemmi.make_structure_from_block(block)
    plddts = []
    for model in st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    plddts.append(atom.b_iso)

    return np.mean(plddts)


def calculate_pocket_reliability(file_path, centroid, radius=8.0):
    st = gemmi.read_structure(str(file_path))
    ns = gemmi.NeighborSearch(st[0], st.cell, radius)
    ns.populate(include_h=False)
    pos = gemmi.Position(*centroid)
    marks = ns.find_atoms(pos, "\0", radius=radius)
    plddts = []
    for mark in marks:
        cra = mark.to_cra(st[0])
        if cra.atom:
            plddts.append(cra.atom.b_iso)
    if not plddts:
        return 0.0
    return np.min(plddts)


DATA_DIR = Path("/home/vu2002123/target-elucidation/data/HPC_input")
pocket_df = pd.read_csv(DATA_DIR / "Dataset1/Dataset1_protein_list.csv")
pocket_df["CIF_file"] = pocket_df.apply(
    lambda row: str(DATA_DIR / f"AF_v4/{row['Name']}.cif.gz"), axis=1
)

pocket_df["min_plddt"] = [
    calculate_pocket_reliability(r.CIF_file, [r.Center_x, r.Center_y, r.Center_z])
    for r in pocket_df.itertuples()
]
filter_df = pocket_df.query("min_plddt >= 70")
filter_df.shape

pocket_df.to_csv(DATA_DIR / "Dataset1/protein_list_plddt.csv", index=False)
filter_df.to_csv(DATA_DIR / "Dataset1/protein_list_plddt_filtered.csv", index=False)
