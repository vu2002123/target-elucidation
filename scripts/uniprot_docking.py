#!/usr/bin/env python3

import argparse
from pathlib import Path
import shutil
import subprocess

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"
DEFAULT_LIGAND_DIR = DATA_DIR / "HPC_compound" / "protonated_minimized"
DEFAULT_AF_DIR = DATA_DIR / "HPC_input" / "AF_v6"
DEFAULT_POCKET_CSV = DEFAULT_AF_DIR / "all_pockets.csv"
DEFAULT_RECEPTOR_DIR = DEFAULT_AF_DIR / "prepared_pdbqt"
EBOX_SIZE_SCRIPT = PROJECT_DIR / "scripts" / "eBoxSize-1.1.pl"
EXHAUSTIVENESS = 16

POCKET_COLUMNS = {"ID", "Name", "Site", "Center_x", "Center_y", "Center_z"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dock a list of ligands against AlphaFold structures selected by "
            "UniProt ID and pocket centers from AF_v6/all_pockets.csv."
        )
    )
    parser.add_argument(
        "--drug_file",
        required=True,
        type=Path,
        help="Text file containing one ligand name per line.",
    )
    parser.add_argument(
        "--uniprot_file",
        required=True,
        type=Path,
        help="Text file containing one UniProt ID per line.",
    )
    parser.add_argument(
        "--output_dir",
        default="uniprot_docking",
        help="Relative output folder under data/interim (default: uniprot_docking).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run docking again when an output SDF already exists.",
    )
    return parser.parse_args()


def read_nonempty_lines(path: Path) -> list[str]:
    """Read unique non-empty, non-NaN values while preserving input order."""
    if not path.is_file():
        raise FileNotFoundError(path)

    values = []
    seen = set()
    with path.open() as file:
        for line in file:
            value = line.strip()
            if not value or value.lower() == "nan" or value in seen:
                continue
            seen.add(value)
            values.append(value)
    return values


def read_uniprot_ids(path: Path) -> list[str]:
    uniprot_ids = []
    seen = set()
    for value in read_nonempty_lines(path):
        uniprot_id = value.upper()
        if uniprot_id not in seen:
            seen.add(uniprot_id)
            uniprot_ids.append(uniprot_id)
    return uniprot_ids


def read_drug_names(path: Path) -> list[str]:
    names = []
    seen = set()
    for value in read_nonempty_lines(path):
        name = Path(value).stem if value.lower().endswith(".sdf") else value
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def load_matching_pockets(pocket_csv: Path, uniprot_ids: list[str]) -> pd.DataFrame:
    """Return all indexed structure/site rows matching the requested UniProt IDs."""
    if not pocket_csv.is_file():
        raise FileNotFoundError(pocket_csv)

    pockets = pd.read_csv(pocket_csv)
    missing_columns = POCKET_COLUMNS - set(pockets.columns)
    if missing_columns:
        raise ValueError(f"Missing pocket-index columns: {sorted(missing_columns)}")

    pockets["ID"] = pockets["ID"].astype("string").str.strip().str.upper()
    for column in ["Center_x", "Center_y", "Center_z"]:
        pockets[column] = pd.to_numeric(pockets[column], errors="coerce")

    matching = pockets[pockets["ID"].isin(uniprot_ids)].copy()
    invalid_coordinates = matching[["Center_x", "Center_y", "Center_z"]].isna().any(axis=1)
    if invalid_coordinates.any():
        invalid_count = int(invalid_coordinates.sum())
        print(f"Skipping {invalid_count} pocket rows with invalid center coordinates")
        matching = matching[~invalid_coordinates]

    return matching.sort_values(["ID", "Name", "Site"]).reset_index(drop=True)


def validate_output_dir(output_dir: str) -> Path:
    relative_path = Path(output_dir)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("--output_dir must be a relative folder under data/interim")
    return INTERIM_DIR / relative_path


def calculate_ligand_box_size(ligand_file: Path) -> float:
    """Calculate a cubic docking-box length from a ligand SDF using eBoxSize."""
    if not EBOX_SIZE_SCRIPT.is_file():
        raise FileNotFoundError(EBOX_SIZE_SCRIPT)

    completed = subprocess.run(
        [str(EBOX_SIZE_SCRIPT), str(ligand_file)],
        check=True,
        text=True,
        capture_output=True,
    )
    output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise ValueError(f"eBoxSize returned no box size for {ligand_file}")
    try:
        box_size = float(output_lines[-1])
    except ValueError as error:
        raise ValueError(
            f"Could not parse eBoxSize output for {ligand_file}: {completed.stdout!r}"
        ) from error
    if box_size <= 0:
        raise ValueError(f"Invalid eBoxSize value for {ligand_file}: {box_size}")
    return box_size


def gnina_command(
    receptor_file: Path,
    ligand_file: Path,
    output_file: Path,
    log_file: Path,
    center_x: float,
    center_y: float,
    center_z: float,
    box_size: float,
) -> list[str]:
    return [
        "gnina",
        "--receptor",
        str(receptor_file),
        "--ligand",
        str(ligand_file),
        "--center_x",
        str(center_x),
        "--center_y",
        str(center_y),
        "--center_z",
        str(center_z),
        "--size_x",
        str(box_size),
        "--size_y",
        str(box_size),
        "--size_z",
        str(box_size),
        "--exhaustiveness",
        str(EXHAUSTIVENESS),
        "--cnn_scoring",
        "rescore",
        "--out",
        str(output_file),
        "--log",
        str(log_file),
    ]


def main() -> None:
    args = parse_args()
    if shutil.which("gnina") is None:
        raise FileNotFoundError("gnina was not found on PATH")

    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uniprot_ids = read_uniprot_ids(args.uniprot_file)
    drug_names = read_drug_names(args.drug_file)
    if not uniprot_ids:
        raise ValueError("The UniProt ID file contains no usable IDs")
    if not drug_names:
        raise ValueError("The drug file contains no usable ligand names")

    pockets = load_matching_pockets(DEFAULT_POCKET_CSV, uniprot_ids)
    matched_ids = set(pockets["ID"])
    unmatched_ids = [uniprot_id for uniprot_id in uniprot_ids if uniprot_id not in matched_ids]
    for uniprot_id in unmatched_ids:
        print(f"No matching pocket structure for UniProt ID: {uniprot_id}")

    print(
        f"Resolved {len(matched_ids):,}/{len(uniprot_ids):,} UniProt IDs to "
        f"{len(pockets):,} structure-site rows"
    )

    ligand_files = {
        drug_name: DEFAULT_LIGAND_DIR / f"{drug_name}.sdf" for drug_name in drug_names
    }
    ligand_box_sizes = {}
    for drug_name, ligand_file in ligand_files.items():
        if not ligand_file.is_file():
            print(f"Missing ligand: {ligand_file}")
            continue
        try:
            ligand_box_sizes[drug_name] = calculate_ligand_box_size(ligand_file)
            print(
                f"{drug_name}: eBoxSize={ligand_box_sizes[drug_name]:.3f} Angstrom"
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print(f"Could not calculate box size for {ligand_file}: {error}")

    manifest_rows = []
    for pocket in pockets.itertuples(index=False):
        receptor_file = DEFAULT_RECEPTOR_DIR / f"{pocket.Name}_prepared.pdbqt"
        pocket_name = f"{pocket.Name}_{pocket.Site}"
        pocket_output_dir = output_dir / str(pocket.ID) / pocket_name
        pocket_output_dir.mkdir(parents=True, exist_ok=True)

        for drug_name in drug_names:
            ligand_file = ligand_files[drug_name]
            output_file = pocket_output_dir / f"{drug_name}_docked.sdf"
            log_file = pocket_output_dir / f"{drug_name}_gnina.log"
            status = "pending"
            return_code = pd.NA
            box_size = ligand_box_sizes.get(drug_name, pd.NA)

            if not receptor_file.is_file():
                status = "missing_receptor"
                print(f"Missing receptor: {receptor_file}")
            elif not ligand_file.is_file():
                status = "missing_ligand"
            elif pd.isna(box_size):
                status = "ebox_size_error"
            elif output_file.is_file() and not args.overwrite:
                status = "existing_output"
                print(f"Skipping existing output: {output_file}")
            else:
                command = gnina_command(
                    receptor_file=receptor_file,
                    ligand_file=ligand_file,
                    output_file=output_file,
                    log_file=log_file,
                    center_x=float(pocket.Center_x),
                    center_y=float(pocket.Center_y),
                    center_z=float(pocket.Center_z),
                    box_size=float(box_size),
                )
                print(
                    f"Docking {drug_name} against {pocket.ID} / "
                    f"{pocket.Name} / {pocket.Site}"
                )
                completed = subprocess.run(command, check=False)
                return_code = completed.returncode
                status = "success" if completed.returncode == 0 else "failed"

            manifest_rows.append(
                {
                    "uniprot_id": pocket.ID,
                    "structure_name": pocket.Name,
                    "site": pocket.Site,
                    "center_x": pocket.Center_x,
                    "center_y": pocket.Center_y,
                    "center_z": pocket.Center_z,
                    "box_size": box_size,
                    "drug": drug_name,
                    "receptor_file": str(receptor_file),
                    "ligand_file": str(ligand_file),
                    "output_file": str(output_file),
                    "log_file": str(log_file),
                    "status": status,
                    "return_code": return_code,
                }
            )

    for uniprot_id in unmatched_ids:
        manifest_rows.append(
            {
                "uniprot_id": uniprot_id,
                "status": "unmatched_uniprot_id",
            }
        )

    manifest_file = output_dir / "docking_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_file, index=False)
    status_counts = pd.Series(
        [row["status"] for row in manifest_rows], dtype="string"
    ).value_counts()
    print("\nDocking status summary:")
    print(status_counts.to_string())
    print(f"Manifest saved to: {manifest_file}")
    print(f"Docking results saved under: {output_dir}")


if __name__ == "__main__":
    main()
