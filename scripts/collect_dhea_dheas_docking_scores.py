#!/usr/bin/env python3

"""Collect and normalize DHEA/DHEAS docking scores from three databases."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
UNIPROT_FILE = PROJECT_DIR / "data" / "raw" / "uniprot_ids.tsv"

COMPOUNDS = ("DHEA", "DHEAS")
DATASETS = (1, 2, 3)
OUTPUT_COLUMNS = [
    "Compound",
    "Dataset",
    "rank",
    "UNIPROT_ID",
    "Gene_Name",
    "minimizedAffinity",
    "CNNscore",
    "CNNaffinity",
    "CNN_VS",
    "source_record",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INTERIM_DIR,
        help=f"Directory for the six output CSV files (default: {INTERIM_DIR}).",
    )
    return parser.parse_args()


def load_gene_map() -> dict[str, str]:
    """Return a UniProt-to-primary-gene mapping."""
    uniprot = pd.read_csv(UNIPROT_FILE, sep="\t", usecols=["Entry", "Gene Names"])
    uniprot["Entry"] = uniprot["Entry"].astype("string").str.strip().str.upper()
    uniprot["Gene_Name"] = (
        uniprot["Gene Names"].astype("string").str.strip().str.split().str[0]
    )
    return (
        uniprot.dropna(subset=["Entry", "Gene_Name"])
        .drop_duplicates("Entry")
        .set_index("Entry")["Gene_Name"]
        .to_dict()
    )


def dataset_path(compound: str, dataset: int) -> Path:
    """Resolve the available source file for one compound/database pair."""
    paths = {
        (compound, 1): INTERIM_DIR / f"docking_{compound}_best_per_gene.tsv",
        (compound, 2): INTERIM_DIR / f"{compound}_AF2-PD_annotated_out.csv",
        (compound, 3): INTERIM_DIR / "DHEA_DS_out.csv",
    }
    path = paths[(compound, dataset)]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_dataset_1(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    return pd.DataFrame(
        {
            "UNIPROT_ID": table["ID"],
            "Gene_Name": table.get("Gene_Name"),
            "minimizedAffinity": table["affinity"],
            "CNNscore": table["CNN_pose_score"],
            "CNNaffinity": table["CNN_affinity"],
            "source_record": table["FilePath"],
        }
    )


def load_dataset_2(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    return pd.DataFrame(
        {
            "UNIPROT_ID": table["ID"],
            "Gene_Name": table.get("Gene_Name"),
            "minimizedAffinity": table["Delta_G"],
            "CNNscore": table["CNN_score"],
            "CNNaffinity": table["CNN_affinity"],
            "source_record": table["Domain"],
        }
    )


def load_dataset_3(path: Path, compound: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table[
        table["Compound"].astype("string").str.strip().str.casefold()
        == compound.casefold()
    ].copy()
    return pd.DataFrame(
        {
            "UNIPROT_ID": table["File_Name"].str.split("_").str[0],
            "Gene_Name": pd.Series(pd.NA, index=table.index, dtype="string"),
            "minimizedAffinity": table["minimizedAffinity"],
            "CNNscore": table["CNNscore"],
            "CNNaffinity": table["CNNaffinity"],
            "CNN_VS": table["CNN_VS"],
            "source_record": table["File_Name"],
        }
    )


def normalize_scores(
    table: pd.DataFrame,
    compound: str,
    dataset: int,
    gene_map: dict[str, str],
) -> pd.DataFrame:
    """Apply the score and target-selection logic from top_target_extraction.py."""
    table["UNIPROT_ID"] = (
        table["UNIPROT_ID"].astype("string").str.strip().str.upper()
    )
    for column in ("minimizedAffinity", "CNNscore", "CNNaffinity"):
        table[column] = pd.to_numeric(table[column], errors="coerce")

    if "CNN_VS" in table:
        table["CNN_VS"] = pd.to_numeric(table["CNN_VS"], errors="coerce")
    else:
        table["CNN_VS"] = table["CNNscore"] * table["CNNaffinity"]

    # Retain the highest-CNN_VS row for each UniProt target. Unlike
    # top_target_extraction.py, keep targets regardless of minimizedAffinity.
    table = (
        table.dropna(subset=["UNIPROT_ID", "CNN_VS"])
        .sort_values("CNN_VS", ascending=False, kind="stable")
        .drop_duplicates("UNIPROT_ID", keep="first")
        .reset_index(drop=True)
    )
    mapped_genes = table["UNIPROT_ID"].map(gene_map)
    table["Gene_Name"] = table["Gene_Name"].replace("", pd.NA).fillna(mapped_genes)
    table.insert(0, "Dataset", dataset)
    table.insert(0, "Compound", compound)
    table.insert(2, "rank", np.arange(1, len(table) + 1))
    return table[OUTPUT_COLUMNS]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gene_map = load_gene_map()

    for compound in COMPOUNDS:
        for dataset in DATASETS:
            source = dataset_path(compound, dataset)
            if dataset == 1:
                scores = load_dataset_1(source)
            elif dataset == 2:
                scores = load_dataset_2(source)
            else:
                scores = load_dataset_3(source, compound)

            scores = normalize_scores(scores, compound, dataset, gene_map)
            output = args.output_dir / f"{compound}_dataset_{dataset}_docking_scores.csv"
            scores.to_csv(output, index=False)
            print(
                f"{compound}, dataset {dataset}: {len(scores):,} targets -> {output}"
            )


if __name__ == "__main__":
    main()
