#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
import argparse
import pandas as pd
from aqme.csearch import csearch
from aqme.cmin import cmin
from rdkit import Chem


def protonate_smiles(smiles, pH=7.4):
    cmd = ["obabel", f"-:{smiles}", "-osmi", "-p", str(pH)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.split()[0]


def get_best_conformer(input_sdf, output_sdf):
    supplier = Chem.SDMolSupplier(input_sdf, removeHs=False, sanitize=False)
    lowest_energy = float("inf")
    best_conformer = None
    for i, mol in enumerate(supplier):
        if mol is None:
            print(f"Warning: Could not read conformer {i}")
            continue
        if mol.HasProp("Energy"):
            energy = float(mol.GetProp("Energy"))
            if energy < lowest_energy:
                lowest_energy = energy
                best_conformer = mol
        else:
            print(f"Warning: Conformer {i} has no <Energy> tag.")
    if best_conformer is not None:
        writer = Chem.SDWriter(output_sdf)
        writer.write(best_conformer)
        writer.close()
    else:
        print("Failed to find any conformers with an <Energy> property.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sdf file containing conformers from input csv"
    )
    # Required arguments
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        metavar=" ",
        type=str,
        help="CSV file of compound name (header code_name) and SMILES (header SMILES)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar=" ",
        type=str,
        help="Output folder to store the sdf files",
    )
    args = parser.parse_args()

    work_dir = Path.home() / "target-elucidation"
    input_dir = work_dir / "data/raw"

    csv_file = input_dir / str(args.input)
    csv_file_protonated = input_dir / f"{Path(args.input).stem}_protonated.csv"
    out_dir = work_dir / "data/processed" / str(args.output)
    raw_dir = out_dir / "raw"

    df = pd.read_csv(csv_file)
    df["pSMILES"] = df["SMILES"].apply(protonate_smiles)
    df_new = df[["code_name", "pSMILES"]].rename(columns={"pSMILES": "SMILES"})
    df_new.to_csv(csv_file_protonated, index=False)

    csearch(destination=str(raw_dir), program="rdkit", input=str(csv_file_protonated))

    # os.chdir(raw_dir)
    # min_dir = out_dir / "min"
    # cmin(files="*.sdf", destination=str(min_dir), program="xtb")

    for file in raw_dir.glob("*rdkit.sdf"):
        name = "_".join(file.stem.split("_")[:-1])
        get_best_conformer(file, raw_dir / f"{name}.sdf")
