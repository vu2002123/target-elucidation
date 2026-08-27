#!/usr/bin/env python3

"""Collect the best GNINA pose from AlphaFold docking results."""

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import sdfrust


PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
MANIFEST_COLUMNS = {
    "uniprot_id",
    "structure_name",
    "site",
    "drug",
    "output_file",
    "status",
}
RESULT_COLUMNS = [
    "Compound",
    "UNIPROT_ID",
    "Structure_Name",
    "Site",
    "Center_x",
    "Center_y",
    "Center_z",
    "Box_Size",
    "minimizedAffinity",
    "CNNscore",
    "CNNaffinity",
    "CNN_VS",
    "Best_Pose",
    "Pose_Count",
    "Docking_Status",
    "Receptor_File",
    "Ligand_File",
    "File_Path",
    "Log_File",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the highest-CNNscore pose from AlphaFold docking outputs "
            "created by uniprot_docking.py."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Relative AlphaFold docking folder under data/interim.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: system ProcessPoolExecutor setting).",
    )
    return parser.parse_args()


def resolve_input_dir(value: str) -> Path:
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("--input must be a relative folder under data/interim")
    return INTERIM_DIR / relative_path


def optional_value(row: dict, column: str):
    value = row.get(column, pd.NA)
    return pd.NA if pd.isna(value) else value


def resolve_output_file(row: dict, input_dir: Path) -> Path | None:
    """Use the manifest path, with a hierarchy-based fallback if data were moved."""
    manifest_path = row.get("output_file")
    if isinstance(manifest_path, str) and manifest_path.strip():
        path = Path(manifest_path)
        if path.is_file():
            return path

    required = ("uniprot_id", "structure_name", "site", "drug")
    if any(pd.isna(row.get(column)) for column in required):
        return Path(manifest_path) if isinstance(manifest_path, str) else None

    fallback = (
        input_dir
        / str(row["uniprot_id"])
        / f"{row['structure_name']}_{row['site']}"
        / f"{row['drug']}_docked.sdf"
    )
    return fallback


def collect_best_pose(task: dict) -> dict:
    row = task["row"]
    input_dir = Path(task["input_dir"])
    output_file = resolve_output_file(row, input_dir)
    report = {
        "Compound": optional_value(row, "drug"),
        "UNIPROT_ID": optional_value(row, "uniprot_id"),
        "Structure_Name": optional_value(row, "structure_name"),
        "Site": optional_value(row, "site"),
        "Docking_Status": optional_value(row, "status"),
        "File_Path": str(output_file) if output_file is not None else pd.NA,
        "Collection_Status": "pending",
        "Collection_Error": pd.NA,
    }

    if output_file is None or not output_file.is_file():
        report["Collection_Status"] = "missing_output"
        return {"result": None, "report": report}

    best_mol = None
    best_pose_number = None
    best_cnnscore = float("-inf")
    pose_count = 0
    try:
        for pose_number, mol in enumerate(
            sdfrust.iter_sdf_file(str(output_file)), start=1
        ):
            pose_count += 1
            try:
                cnnscore = float(mol.get_property("CNNscore"))
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(cnnscore) and cnnscore > best_cnnscore:
                best_mol = mol
                best_pose_number = pose_number
                best_cnnscore = cnnscore
    except Exception as error:
        report["Collection_Status"] = "read_error"
        report["Collection_Error"] = str(error)
        return {"result": None, "report": report}

    if best_mol is None:
        report["Collection_Status"] = (
            "empty_output" if pose_count == 0 else "no_valid_cnnscore"
        )
        return {"result": None, "report": report}

    try:
        minimized_affinity = float(best_mol.get_property("minimizedAffinity"))
        cnn_affinity = float(best_mol.get_property("CNNaffinity"))
        cnn_vs = float(best_mol.get_property("CNN_VS"))
    except (KeyError, TypeError, ValueError) as error:
        report["Collection_Status"] = "invalid_score_fields"
        report["Collection_Error"] = str(error)
        return {"result": None, "report": report}

    compound = optional_value(row, "drug")
    if pd.isna(compound):
        compound = str(best_mol.name).split()[0]

    result = {
        "Compound": compound,
        "UNIPROT_ID": optional_value(row, "uniprot_id"),
        "Structure_Name": optional_value(row, "structure_name"),
        "Site": optional_value(row, "site"),
        "Center_x": optional_value(row, "center_x"),
        "Center_y": optional_value(row, "center_y"),
        "Center_z": optional_value(row, "center_z"),
        "Box_Size": optional_value(row, "box_size"),
        "minimizedAffinity": minimized_affinity,
        "CNNscore": best_cnnscore,
        "CNNaffinity": cnn_affinity,
        "CNN_VS": cnn_vs,
        "Best_Pose": best_pose_number,
        "Pose_Count": pose_count,
        "Docking_Status": optional_value(row, "status"),
        "Receptor_File": optional_value(row, "receptor_file"),
        "Ligand_File": optional_value(row, "ligand_file"),
        "File_Path": str(output_file),
        "Log_File": optional_value(row, "log_file"),
    }
    report["Collection_Status"] = "collected"
    report["Best_Pose"] = best_pose_number
    report["Pose_Count"] = pose_count
    return {"result": result, "report": report}


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input)
    manifest_file = input_dir / "docking_manifest.csv"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Docking manifest not found: {manifest_file}")

    manifest = pd.read_csv(manifest_file)
    missing_columns = MANIFEST_COLUMNS - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Missing manifest columns: {sorted(missing_columns)}")

    tasks = [
        {"row": row, "input_dir": str(input_dir)}
        for row in manifest.to_dict(orient="records")
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        collected = list(executor.map(collect_best_pose, tasks))

    results = [item["result"] for item in collected if item["result"] is not None]
    reports = [item["report"] for item in collected]

    output_file = INTERIM_DIR / f"{args.input}_out.csv"
    report_file = INTERIM_DIR / f"{args.input}_collection_report.csv"
    pd.DataFrame(results, columns=RESULT_COLUMNS).to_csv(output_file, index=False)
    report_df = pd.DataFrame(reports)
    report_df.to_csv(report_file, index=False)

    status_counts = report_df["Collection_Status"].value_counts(dropna=False)
    print("Collection status summary:")
    print(status_counts.to_string())
    print(f"Collected {len(results):,}/{len(manifest):,} manifest rows")
    print(f"Scores saved to: {output_file}")
    print(f"Collection report saved to: {report_file}")


if __name__ == "__main__":
    main()
