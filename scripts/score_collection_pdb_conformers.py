#!/usr/bin/env python3

import sdfrust
import pandas as pd
import argparse
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def get_args():
    parser = argparse.ArgumentParser(description="Get score from SDF files")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        metavar=" ",
        type=str,
        help="Input batch name containing SDF files",
    )
    return parser


def get_best_pose(file: Path):
    mol = next(sdfrust.iter_sdf_file(str(file)), None)
    if mol is None:
        print(f"{file} is empty")
        return None
    try:
        best_pose = {
            "Compound": str(mol.name).split(" ")[0],
            "minimizedAffinity": mol.get_property("minimizedAffinity"),
            "CNNscore": mol.get_property("CNNscore"),
            "CNNaffinity": mol.get_property("CNNaffinity"),
            "CNN_VS": mol.get_property("CNN_VS"),
            "File_Path": file,
        }
        # print(f"Getting result from {file.name}")
    except Exception:
        return None
    return best_pose


def get_best_pose_conformer(file: Path):
    best_mol = None
    best_cnnscore = float("-inf")

    for mol in sdfrust.iter_sdf_file(str(file)):
        try:
            cnnscore = float(mol.get_property("CNNscore"))

            if math.isfinite(cnnscore) and cnnscore > best_cnnscore:
                best_cnnscore = cnnscore
                best_mol = mol

        except (TypeError, ValueError, KeyError):
            # Skip poses with missing or invalid CNNscore values.
            continue

    if best_mol is None:
        print(f"{file} is empty or contains no valid CNNscore")
        return None

    try:
        return {
            "Compound": str(best_mol.name).split()[0],
            "minimizedAffinity": float(best_mol.get_property("minimizedAffinity")),
            "CNNscore": float(best_mol.get_property("CNNscore")),
            "CNNaffinity": float(best_mol.get_property("CNNaffinity")),
            "CNN_VS": float(best_mol.get_property("CNN_VS")),
            "File_Path": str(file),
        }

    except (TypeError, ValueError, KeyError):
        return None


def main():
    parser = get_args()
    args = parser.parse_args()

    WORK_DIR = Path("/home/vu2002123/target-elucidation/data/")
    RAW_DIR = WORK_DIR / "raw"
    HPC_DIR = WORK_DIR / "HPC_mount"
    INTERIM_DIR = WORK_DIR / "interim"

    DIR = INTERIM_DIR / args.input / "top3_poses"
    sdf_list = list(DIR.rglob("*.sdf"))

    with ProcessPoolExecutor() as executor:
        results = [r for r in executor.map(get_best_pose_conformer, sdf_list) if r is not None]

    result_df = pd.DataFrame(results)
    result_df.to_csv(INTERIM_DIR / f"{args.input}_out.csv", index=False)


if __name__ == "__main__":
    main()
