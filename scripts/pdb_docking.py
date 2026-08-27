#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Dock all drugs against all receptors using GNINA."
    )

    parser.add_argument(
        "--drug_file",
        help="Text file containing one drug name per line",
    )
    parser.add_argument(
        "--protein_csv",
        help="CSV containing receptor and cognate_ligand columns",
    )
    parser.add_argument(
        "--output_dir",
        help="Folder for docking results",
    )

    args = parser.parse_args()

    ligand_dir = Path.home() / "target-elucidation/data/HPC_compound/protonated_minimized"
    receptor_dir = Path.home() / "target-elucidation/data/raw/PDB"
    cognate_dir = Path.home() / "target-elucidation/data/raw/PDB"
    output_dir = Path.home() / "target-elucidation/data/interim" / args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.drug_file, "r") as file:
        drug_names = [line.strip() for line in file]

    proteins = pd.read_csv(args.protein_csv)

    for _, row in proteins.iterrows():
        receptor_name = str(row["receptor"]).strip()
        cognate_name = str(row["cognate_ligand"]).strip()

        receptor_file = receptor_dir / receptor_name
        cognate_file = cognate_dir / cognate_name

        receptor_stem = receptor_file.stem
        receptor_output_dir = output_dir / receptor_stem
        receptor_output_dir.mkdir(parents=True, exist_ok=True)

        for drug_name in drug_names:
            drug_file = ligand_dir / f"{drug_name}_rdkit.sdf"

            output_file = receptor_output_dir / f"{drug_name}_docked.sdf"

            log_file = receptor_output_dir / f"{drug_name}_gnina.log"

            command = [
                "gnina",
                "--receptor",
                str(receptor_file),
                "--ligand",
                str(drug_file),
                "--autobox_ligand",
                str(cognate_file),
                "--autobox_add",
                "4",
                "--exhaustiveness",
                "16",
                "--cnn_scoring",
                "rescore",
                "--out",
                str(output_file),
                "--log",
                str(log_file),
            ]

            print(f"Docking {drug_name} against {receptor_name}")

            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
