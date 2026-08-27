#!/usr/bin/env python3

"""Calculate heavy-atom RMSD for EGFR cognate ligand docking poses."""

import argparse
from pathlib import Path
import re
import subprocess

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_DOCKING_FILE = DATA_DIR / "interim" / "EGFR_PDB_out.csv"
DEFAULT_METADATA_FILE = DATA_DIR / "raw" / "PDB" / "PDB_IDs.csv"
DEFAULT_PDB_DIR = DATA_DIR / "raw" / "PDB"
DEFAULT_OUTPUT = DATA_DIR / "interim" / "EGFR_PDB_cognate_rmsd.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docking-file", type=Path, default=DEFAULT_DOCKING_FILE)
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA_FILE)
    parser.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--conda-environment",
        default="aqme",
        help="Conda environment containing obrms (default: aqme).",
    )
    return parser.parse_args()


def run_obrms(
    docked_file: Path,
    cognate_file: Path,
    conda_environment: str,
) -> float:
    """Calculate the first (top-ranked) docked pose RMSD with obrms."""
    command = [
        "conda",
        "run",
        "-n",
        conda_environment,
        "obrms",
        "-f",
        str(docked_file),
        str(cognate_file),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"obrms failed for {docked_file}: {message}")
    match = re.search(r"([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s*$", completed.stdout)
    if not match:
        raise ValueError(f"Could not parse obrms output: {completed.stdout.strip()}")
    return float(match.group(1))


def main() -> None:
    args = parse_args()
    docking = pd.read_csv(args.docking_file)
    metadata = pd.read_csv(args.metadata_file)
    required_docking = {"Compound", "File_Path"}
    required_metadata = {"ID", "Chain", "cognate", "name", "name.1"}
    if missing := required_docking - set(docking.columns):
        raise ValueError(f"Missing docking columns: {sorted(missing)}")
    if missing := required_metadata - set(metadata.columns):
        raise ValueError(f"Missing PDB metadata columns: {sorted(missing)}")

    docking["PDB_ID"] = docking["File_Path"].map(
        lambda value: Path(value).parent.name.split("_")[0].upper()
    )
    metadata["ID"] = metadata["ID"].astype(str).str.strip().str.upper()
    metadata["name.1"] = metadata["name.1"].astype(str).str.strip()
    merged = docking.merge(metadata, left_on="PDB_ID", right_on="ID", how="inner")
    cognate_pairs = merged[
        merged["Compound"].str.casefold() == merged["name.1"].str.casefold()
    ].copy()
    if cognate_pairs.empty:
        raise ValueError("No cognate compound–PDB docking pairs were found")

    rows = []
    for row in cognate_pairs.itertuples(index=False):
        docked_file = Path(row.File_Path)
        cognate_file = args.pdb_dir / (
            f"{row.PDB_ID}_{row.Chain}_{row.cognate}_cognate.pdb"
        )
        if not docked_file.is_file():
            raise FileNotFoundError(docked_file)
        if not cognate_file.is_file():
            raise FileNotFoundError(cognate_file)
        rmsd = run_obrms(docked_file, cognate_file, args.conda_environment)
        rows.append(
            {
                "PDB_ID": row.PDB_ID,
                "EGFR_variant": row.name,
                "Compound": row.Compound,
                "Cognate_CCD": row.cognate,
                "RMSD_Angstrom": rmsd,
                "Docked_pose": str(docked_file),
                "Cognate_structure": str(cognate_file),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["EGFR_variant", "PDB_ID"], kind="stable"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))
    print(f"\nRMSD results saved to: {args.output}")


if __name__ == "__main__":
    main()
