#!/usr/bin/env python3

"""Export ranked docking results and Top-K/percent UniProt ID lists."""

import argparse
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
REGISTRY_FILE = INTERIM_DIR / "file_locations.csv"
UNIPROT_FILE = PROJECT_DIR / "data" / "raw" / "uniprot_ids.tsv"
DEFAULT_OUTPUT_DIR = INTERIM_DIR / "multidrug_docking_rankings"

DRUGS = (
    "Lenvatinib",
    "Erlotinib",
    "Afatinib",
    "Gefitinib",
    "Osimertinib",
    "Crizotinib",
    "Ruxolitinib",
    "Sunitinib",
    "Imatinib",
)
DATASETS = (1, 2, 3)
TOP_K_VALUES = (10, 20, 50, 100)
TOP_PERCENT_VALUES = (1, 5, 10, 20)
CNN_AFFINITY_THRESHOLDS = (6, 7)
OUTPUT_COLUMNS = (
    "Compound",
    "Dataset",
    "Rank",
    "UNIPROT_ID",
    "Gene_Name",
    "minimizedAffinity",
    "CNNscore",
    "CNNaffinity",
    "CNN_VS",
    "File_Name",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def derive_uniprot_id(file_name: object, dataset: int) -> str | pd.NA:
    value = str(file_name)
    if dataset == 1:
        match = re.search(r"(?:^|/)AF-([A-Za-z0-9]+)-", value)
        if match:
            return match.group(1).upper()
    prefix = Path(value).name.split("_", maxsplit=1)[0].strip().upper()
    return prefix or pd.NA


def load_ranked_result(
    path: Path,
    extension: str,
    drug: str,
    dataset: int,
) -> pd.DataFrame:
    """Normalize a docking table and retain each protein's best CNN_VS pocket."""
    separator = "\t" if extension.lower() == "tsv" or path.suffix.lower() == ".tsv" else ","
    table = pd.read_csv(path, sep=separator)
    table = table.rename(
        columns={
            "ID": "UNIPROT_ID",
            "Gene_name": "Gene_Name",
            "CNN_pose_score": "CNNscore",
            "CNN_score": "CNNscore",
            "CNN_affinity": "CNNaffinity",
            "affinity": "minimizedAffinity",
        }
    )
    if "Compound" in table.columns:
        table = table[
            table["Compound"].astype("string").str.strip().str.casefold()
            == drug.casefold()
        ].copy()
    if table.empty:
        raise ValueError(f"No rows for {drug} in {path}")

    if "UNIPROT_ID" not in table.columns:
        if "File_Name" not in table.columns:
            raise ValueError(f"No UNIPROT_ID, ID, or File_Name column in {path}")
        table["UNIPROT_ID"] = table["File_Name"].map(
            lambda value: derive_uniprot_id(value, dataset)
        )
    table["UNIPROT_ID"] = (
        table["UNIPROT_ID"].astype("string").str.strip().str.upper()
    )

    for column in ("minimizedAffinity", "CNNscore", "CNNaffinity", "CNN_VS"):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    if "CNN_VS" not in table.columns:
        if not {"CNNscore", "CNNaffinity"}.issubset(table.columns):
            raise ValueError(f"Cannot calculate CNN_VS for {path}")
        table["CNN_VS"] = table["CNNscore"] * table["CNNaffinity"]

    ranked = (
        table.dropna(subset=["UNIPROT_ID", "CNN_VS"])
        .sort_values("CNN_VS", ascending=False, kind="stable")
        .drop_duplicates("UNIPROT_ID", keep="first")
        .reset_index(drop=True)
    )
    ranked["Compound"] = drug
    ranked["Dataset"] = dataset
    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    for column in OUTPUT_COLUMNS:
        if column not in ranked.columns:
            ranked[column] = pd.NA
    return ranked[list(OUTPUT_COLUMNS)]


def write_ids(ids: pd.Series, path: Path) -> None:
    ids.to_csv(path, index=False, header=False)


def load_gene_map() -> dict[str, str]:
    """Map UniProt IDs to their primary gene symbols."""
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(REGISTRY_FILE)
    gene_map = load_gene_map()
    manifest_rows = []
    filtered_manifest_rows = []

    for drug in DRUGS:
        file_stem = drug.lower().replace(" ", "_")
        drug_output_dir = args.output_dir / file_stem
        drug_output_dir.mkdir(parents=True, exist_ok=True)
        for dataset in DATASETS:
            matches = registry[
                registry["Compound"].eq(drug)
                & registry["Dataset"].eq(dataset)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one registry entry for {drug}, dataset {dataset}; "
                    f"found {len(matches)}"
                )
            location = matches.iloc[0]
            source_file = INTERIM_DIR / str(location["File_location"])
            if not source_file.is_file():
                raise FileNotFoundError(source_file)
            ranked = load_ranked_result(
                source_file,
                str(location.get("Extension", source_file.suffix.lstrip("."))),
                drug,
                dataset,
            )
            ranked["Gene_Name"] = (
                ranked["Gene_Name"]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
                .fillna(ranked["UNIPROT_ID"].map(gene_map))
            )
            prefix = f"{file_stem}_dataset_{dataset}"
            result_file = drug_output_dir / f"{prefix}_docking_results.csv"
            ranked.to_csv(result_file, index=False)

            generated_lists = []
            for top_k in TOP_K_VALUES:
                list_file = drug_output_dir / f"{prefix}_top_{top_k}_ids.txt"
                write_ids(ranked.head(top_k)["UNIPROT_ID"], list_file)
                generated_lists.append(str(list_file))
            for percentage in TOP_PERCENT_VALUES:
                count = min(
                    len(ranked),
                    max(1, math.ceil(len(ranked) * percentage / 100)),
                )
                list_file = (
                    drug_output_dir / f"{prefix}_top_{percentage}percent_ids.txt"
                )
                write_ids(ranked.head(count)["UNIPROT_ID"], list_file)
                generated_lists.append(str(list_file))

            manifest_rows.append(
                {
                    "Compound": drug,
                    "Dataset": dataset,
                    "Ranked_protein_count": len(ranked),
                    "Source_file": str(source_file),
                    "Docking_result_file": str(result_file),
                    "Top_1percent_count": math.ceil(len(ranked) * 0.01),
                    "Top_5percent_count": math.ceil(len(ranked) * 0.05),
                    "Top_10percent_count": math.ceil(len(ranked) * 0.10),
                    "Top_20percent_count": math.ceil(len(ranked) * 0.20),
                    "Generated_ID_list_count": len(generated_lists),
                }
            )

            for threshold in CNN_AFFINITY_THRESHOLDS:
                filtered_output_dir = (
                    drug_output_dir / f"cnn_affinity_gt_{threshold}"
                )
                filtered_output_dir.mkdir(parents=True, exist_ok=True)
                filtered = (
                    ranked[ranked["CNNaffinity"].gt(threshold)]
                    .copy()
                    .sort_values("CNNaffinity", ascending=False, kind="stable")
                    .reset_index(drop=True)
                )
                filtered["Rank"] = np.arange(1, len(filtered) + 1)
                filtered_prefix = f"{prefix}_cnn_affinity_gt_{threshold}"
                filtered_file = (
                    filtered_output_dir
                    / f"{filtered_prefix}_docking_results.csv"
                )
                filtered.to_csv(filtered_file, index=False)

                missing_gene_count = int(filtered["Gene_Name"].isna().sum())
                for percentage in TOP_PERCENT_VALUES:
                    count = (
                        min(
                            len(filtered),
                            max(1, math.ceil(len(filtered) * percentage / 100)),
                        )
                        if len(filtered)
                        else 0
                    )
                    top_rows = filtered.head(count)
                    id_file = (
                        filtered_output_dir
                        / (
                            f"{filtered_prefix}_top_{percentage}percent_"
                            "uniprot_ids.txt"
                        )
                    )
                    gene_file = (
                        filtered_output_dir
                        / (
                            f"{filtered_prefix}_top_{percentage}percent_"
                            "gene_names.txt"
                        )
                    )
                    write_ids(top_rows["UNIPROT_ID"], id_file)
                    write_ids(top_rows["Gene_Name"].fillna("NA"), gene_file)

                filtered_manifest_rows.append(
                    {
                        "Compound": drug,
                        "Dataset": dataset,
                        "CNNaffinity_threshold": f"> {threshold}",
                        "Filtered_protein_count": len(filtered),
                        "Missing_gene_name_count": missing_gene_count,
                        "Filtered_result_file": str(filtered_file),
                        "Top_1percent_count": (
                            math.ceil(len(filtered) * 0.01) if len(filtered) else 0
                        ),
                        "Top_5percent_count": (
                            math.ceil(len(filtered) * 0.05) if len(filtered) else 0
                        ),
                        "Top_10percent_count": (
                            math.ceil(len(filtered) * 0.10) if len(filtered) else 0
                        ),
                        "Top_20percent_count": (
                            math.ceil(len(filtered) * 0.20) if len(filtered) else 0
                        ),
                    }
                )
            print(f"{drug}, Dataset {dataset}: {len(ranked):,} ranked proteins")

    manifest = pd.DataFrame(manifest_rows)
    manifest_file = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_file, index=False)
    filtered_manifest = pd.DataFrame(filtered_manifest_rows)
    filtered_manifest_file = args.output_dir / "cnn_affinity_filter_manifest.csv"
    filtered_manifest.to_csv(filtered_manifest_file, index=False)
    print(f"\nExported {len(manifest)} docking result files.")
    print(
        f"Exported {len(manifest) * (len(TOP_K_VALUES) + len(TOP_PERCENT_VALUES))} "
        "ID list files."
    )
    print(f"Manifest saved to: {manifest_file}")
    print(
        f"Exported {len(filtered_manifest)} CNNaffinity-filtered result files and "
        f"{len(filtered_manifest) * len(TOP_PERCENT_VALUES) * 2} filtered lists."
    )
    print(f"Filtered-result manifest saved to: {filtered_manifest_file}")


if __name__ == "__main__":
    main()
