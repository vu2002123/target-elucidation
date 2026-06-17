#!/usr/bin/env python3

import sys
import subprocess
import os
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate csv file for parallelization"
    )
    parser.add_argument(
        "-n",
        "--batch_name",
        required=True,
        type=str,
        help="Name of result folder",
    )
    parser.add_argument(
        "-p",
        "--protein_list",
        required=True,
        type=str,
        help="Input file with UniProt ID",
    )
    parser.add_argument(
        "-l",
        "--ligand_list",
        required=True,
        type=str,
        help="Input file with compound names",
    )
    return parser.parse_args()


def get_docking_list(protein_list: Path | str):
    protein_file = (
        Path.home() / "docking_refined" / "input" / "AF_v6" / "all_pockets.csv"
    )
    protein_df = pd.read_csv(protein_file)

    directory = Path.home() / "docking_refined" / "input" / "AF_v6" / "prepared_pdbqt"
    protein_df["PDB_path"] = protein_df.apply(
        lambda row: str(directory / f"{row['Name']}_prepared.pdbqt"), axis=1
    )

    with open(protein_list, "r") as file:
        protein_id_list = [line.strip() for line in file]

    filtered_protein_df = protein_df.query("ID in @protein_id_list")

    return filtered_protein_df


def save_config_file(data):
    path, content = data
    if Path(path).exists():  # Added () to call the method
        pass
    else:
        Path(path).write_text(content)


def main():
    args = get_args()
    protein_df = get_docking_list(args.protein_list)

    OUTDIR: Path = Path.home() / "docking_refined" / "results" / args.batch_name
    list_file = OUTDIR / f"{args.batch_name}_with_compound.csv"

    OUTDIR.mkdir(exist_ok=True)

    with open(args.ligand_list, "r") as file:
        directory = (
            Path.home() / "docking_refined" / "compound" / "protonated_minimized"
        )
        compound_path = [directory / (line.strip() + ".sdf") for line in file]

    compound_dict = {
        compound: {
            "Box_size": subprocess.check_output(
                ["eBoxSize-1.1.pl", compound], text=True
            ).strip(),
            "Compound_name": compound.stem,
        }
        for compound in compound_path
    }

    compounds_df = pd.DataFrame.from_dict(compound_dict, orient="index")
    compounds_df.reset_index(inplace=True, names="Compound_path")

    if protein_df is not None:
        print(f"There are {len(protein_df)} proteins for docking")
        protein_df_expanded = protein_df.assign(
            Compound_path=pd.Series([compound_path] * len(protein_df), dtype=object)
        ).explode("Compound_path")

        protein_df_expanded = protein_df_expanded.merge(
            compounds_df, on="Compound_path", how="left"
        )

        protein_df_expanded["Out_file"] = protein_df_expanded.apply(
            lambda row: str(
                OUTDIR / f"{row['Name']}_{row['Site']}_{row['Compound_name']}.sdf"
            ),
            axis=1,
        )
        protein_df_expanded["Config_file"] = protein_df_expanded.apply(
            lambda row: str(
                OUTDIR
                / f"{row['Name']}_{row['Site']}_{row['Compound_name']}_config.txt"
            ),
            axis=1,
        )
        protein_df_expanded["Config_content"] = (
            "receptor = "
            + protein_df_expanded["PDB_path"].astype(str)
            + "\n"
            + "ligand = "
            + protein_df_expanded["Compound_path"].astype(str)
            + "\n"
            + "out = "
            + protein_df_expanded["Out_file"].astype(str)
            + "\n"
            + "center_x = "
            + protein_df_expanded["Center_x"].astype(str)
            + "\n"
            + "center_y = "
            + protein_df_expanded["Center_y"].astype(str)
            + "\n"
            + "center_z = "
            + protein_df_expanded["Center_z"].astype(str)
            + "\n"
            + "size_x = "
            + protein_df_expanded["Box_size"].astype(str)
            + "\n"
            + "size_y = "
            + protein_df_expanded["Box_size"].astype(str)
            + "\n"
            + "size_z = "
            + protein_df_expanded["Box_size"].astype(str)
            + "\n"
            + "exhaustiveness = "
            + "4"
            + "\n"
            + "cnn_scoring = "
            + "rescore"
            + "\n"
            + "cnn = "
            + "fast"
            + "\n"
            + "cpu = "
            + "1"
        )

        tasks = [
            (r.Config_file, r.Config_content) for r in protein_df_expanded.itertuples()
        ]

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(save_config_file, tasks)

        protein_df_expanded.to_csv(list_file, index=True)
        print(f"There are {len(protein_df_expanded)} docking runs")
    else:
        print("Protein list is empty")


if __name__ == "__main__":
    main()
