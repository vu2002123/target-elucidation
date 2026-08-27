"""Export known-target ranks across three docking datasets to a Word-ready CSV.

Each row represents one drug/known-target pair. Dataset-specific rank, rank
percentile, score, and best-pocket output filename are stored in wide columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
BINDER_DIR = PROJECT_DIR / "data" / "raw" / "pubchem"
DEFAULT_FILE_LOCATIONS = INTERIM_DIR / "file_locations.csv"
DEFAULT_OUTPUT = INTERIM_DIR / "known_target_ranks_all_datasets.csv"

DEFAULT_DRUGS = [
    "Lenvatinib",
    "Erlotinib",
    "Afatinib",
    "Gefitinib",
    "Osimertinib",
    "Crizotinib",
    "Ruxolitinib",
    "Sunitinib",
    "Tamoxifen",
    "Thioridazine",
    "Trifluoperazine",
    "Imatinib",
    "Venetoclax",
    "Olaparib",
    "Prinomastat",
    "Thalidomide",
    "Lenalidomide",
    "Pomalidomide",
    "Sorafenib",
    "Dacomitinib",
    "Dasatinib",
    "Nilotinib",
    "Prochlorperazine",
    "Niclosamide",
]
DATASETS = (1, 2, 3)
RANKING_METHODS = {
    "minimizedAffinity": {
        "label": "smina docking score",
        "ascending": True,
    },
    "CNNaffinity": {
        "label": "GNINA predicted affinity",
        "ascending": False,
    },
    "CNN_VS": {
        "label": "GNINA combined score",
        "ascending": False,
    },
}


def read_binders(path: Path) -> set[str]:
    """Read unique, non-empty UniProt IDs from a known-binder file."""
    with path.open() as file:
        return {
            binder_id
            for line in file
            if (binder_id := line.strip().upper()) and binder_id != "NAN"
        }


def read_ranked_scores(
    path: Path,
    extension: str,
    drug: str,
    dataset: int,
    ranking_method: str,
    compound_name: str | None = None,
) -> pd.DataFrame:
    """Rank proteins and retain the best-scoring pocket for each UniProt ID."""
    separator = "\t" if extension.lower() == "tsv" or path.suffix.lower() == ".tsv" else ","
    scores = pd.read_csv(path, sep=separator)
    scores = scores.rename(
        columns={
            "CNN_affinity": "CNNaffinity",
            "CNN_pose_score": "CNNscore",
            "CNN_score": "CNNscore",
            "affinity": "minimizedAffinity",
            "ID": "UNIPROT_ID",
            "FilePath": "File_Name",
        }
    )

    if "Compound" in scores.columns:
        file_compound_name = compound_name
        if file_compound_name is None:
            file_compound_name = "PCP" if drug == "Prochlorperazine" else drug
        scores = scores[scores["Compound"] == file_compound_name].copy()

    if "UNIPROT_ID" not in scores.columns:
        if "File_Name" not in scores.columns:
            raise ValueError("no UNIPROT_ID, ID, or File_Name column")
        if dataset == 1:
            scores["UNIPROT_ID"] = scores["File_Name"].str.split("-").str[1]
        else:
            scores["UNIPROT_ID"] = scores["File_Name"].str.split("_").str[0]

    required_scores = {"CNNscore", "CNNaffinity", "minimizedAffinity"}
    missing_scores = required_scores.difference(scores.columns)
    if missing_scores:
        raise ValueError(f"missing score columns: {sorted(missing_scores)}")
    if "File_Name" not in scores.columns:
        raise ValueError("missing output filename column: File_Name")
    scores["File_Name"] = scores["File_Name"].map(lambda value: Path(str(value)).name)

    scores["UNIPROT_ID"] = scores["UNIPROT_ID"].astype("string").str.strip().str.upper()
    for column in required_scores:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    scores["CNN_VS"] = scores["CNNscore"] * scores["CNNaffinity"]

    ascending = RANKING_METHODS[ranking_method]["ascending"]
    ranked = (
        scores[scores["minimizedAffinity"].lt(0)]
        .dropna(subset=["UNIPROT_ID", ranking_method])
        .sort_values(ranking_method, ascending=ascending, kind="stable")
        .drop_duplicates(subset="UNIPROT_ID", keep="first")
        .reset_index(drop=True)
    )
    ranked["Rank"] = ranked.index + 1
    number_of_proteins = len(ranked)
    if number_of_proteins <= 1:
        ranked["Rank percentile (%)"] = 100.0
    else:
        ranked["Rank percentile (%)"] = (
            100 * (number_of_proteins - ranked["Rank"]) / (number_of_proteins - 1)
        )
    ranked["Ranked protein count"] = number_of_proteins
    return ranked


def dataset_target_table(
    ranked_scores: pd.DataFrame,
    binders: set[str],
    dataset: int,
    ranking_method: str,
) -> pd.DataFrame:
    """Select known targets and label columns for a wide cross-dataset join."""
    target_scores = ranked_scores[ranked_scores["UNIPROT_ID"].isin(binders)].copy()
    target_scores = target_scores[
        [
            "UNIPROT_ID",
            "Rank",
            "Rank percentile (%)",
            ranking_method,
            "CNNscore",
            "CNNaffinity",
            "File_Name",
            "Ranked protein count",
        ]
    ]
    return target_scores.rename(
        columns={
            "Rank": f"Dataset {dataset} rank",
            "Rank percentile (%)": f"Dataset {dataset} rank percentile (%)",
            ranking_method: f"Dataset {dataset} score",
            "CNNscore": f"Dataset {dataset} CNNscore",
            "CNNaffinity": f"Dataset {dataset} CNNaffinity",
            "File_Name": f"Dataset {dataset} output filename",
            "Ranked protein count": f"Dataset {dataset} protein count",
        }
    )


def export_known_target_ranks(
    drugs: list[str],
    file_locations_file: Path,
    binder_dir: Path,
    ranking_method: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Build one wide, high-to-low-rank table for all requested drugs."""
    locations = pd.read_csv(file_locations_file)
    required_location_columns = {"Compound", "Dataset", "File_location"}
    missing = required_location_columns.difference(locations.columns)
    if missing:
        raise ValueError(f"{file_locations_file} is missing columns: {sorted(missing)}")

    all_drug_tables = []
    messages = []
    for drug_order, drug in enumerate(drugs):
        binder_file = binder_dir / f"{drug}_filtered_total.txt"
        if not binder_file.is_file():
            messages.append(f"Skipping {drug}: missing binder file {binder_file}")
            continue
        binders = read_binders(binder_file)
        if not binders:
            messages.append(f"Skipping {drug}: binder file is empty")
            continue

        drug_table = pd.DataFrame({"UNIPROT_ID": sorted(binders)})
        for dataset in DATASETS:
            matching_locations = locations[
                (locations["Compound"] == drug) & (locations["Dataset"] == dataset)
            ]
            if matching_locations.empty:
                messages.append(f"{drug}, Dataset {dataset}: no file-location entry")
                continue
            location = matching_locations.iloc[0]
            dock_file = INTERIM_DIR / str(location["File_location"])
            if not dock_file.is_file():
                messages.append(f"{drug}, Dataset {dataset}: missing {dock_file}")
                continue
            try:
                ranked_scores = read_ranked_scores(
                    dock_file,
                    str(location.get("Extension", dock_file.suffix.lstrip("."))),
                    drug,
                    dataset,
                    ranking_method,
                )
            except (OSError, ValueError, pd.errors.ParserError) as error:
                messages.append(f"{drug}, Dataset {dataset}: {error}")
                continue
            drug_table = drug_table.merge(
                dataset_target_table(ranked_scores, binders, dataset, ranking_method),
                on="UNIPROT_ID",
                how="left",
            )

        percentile_columns = [
            f"Dataset {dataset} rank percentile (%)"
            for dataset in DATASETS
            if f"Dataset {dataset} rank percentile (%)" in drug_table.columns
        ]
        drug_table.insert(0, "Drug", drug)
        drug_table["Datasets containing target"] = drug_table[percentile_columns].notna().sum(axis=1)
        drug_table["Best rank percentile (%)"] = drug_table[percentile_columns].max(axis=1)
        drug_table["Mean rank percentile (%)"] = drug_table[percentile_columns].mean(axis=1)
        drug_table["Ranking method"] = RANKING_METHODS[ranking_method]["label"]
        drug_table["_drug_order"] = drug_order
        all_drug_tables.append(drug_table)

    if not all_drug_tables:
        raise ValueError("No requested drugs could be processed")

    output = pd.concat(all_drug_tables, ignore_index=True, sort=False)
    output = output.sort_values(
        ["_drug_order", "Mean rank percentile (%)", "Best rank percentile (%)", "UNIPROT_ID"],
        ascending=[True, False, False, True],
        na_position="last",
        kind="stable",
    ).drop(columns="_drug_order")
    output = output.rename(columns={"UNIPROT_ID": "UniProt ID"})
    dataset_columns = [
        column
        for dataset in DATASETS
        for column in (
            f"Dataset {dataset} rank",
            f"Dataset {dataset} rank percentile (%)",
            f"Dataset {dataset} score",
            f"Dataset {dataset} CNNscore",
            f"Dataset {dataset} CNNaffinity",
            f"Dataset {dataset} output filename",
            f"Dataset {dataset} protein count",
        )
    ]
    summary_columns = [
        "Datasets containing target",
        "Best rank percentile (%)",
        "Mean rank percentile (%)",
        "Ranking method",
    ]
    output = output[["Drug", "UniProt ID", *dataset_columns, *summary_columns]]
    integer_columns = [
        column
        for dataset in DATASETS
        for column in (f"Dataset {dataset} rank", f"Dataset {dataset} protein count")
    ]
    output[integer_columns] = output[integer_columns].astype("Int64")
    return output, messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drugs",
        nargs="+",
        default=DEFAULT_DRUGS,
        help="Drug names to export (default: validation drugs from average_roc_pr_curves.py)",
    )
    parser.add_argument("--file-locations", type=Path, default=DEFAULT_FILE_LOCATIONS)
    parser.add_argument("--binder-dir", type=Path, default=BINDER_DIR)
    parser.add_argument(
        "--ranking-method",
        choices=RANKING_METHODS,
        default="CNN_VS",
        help="Score used to rank proteins (default: CNN_VS, GNINA combined score)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, messages = export_known_target_ranks(
        args.drugs,
        args.file_locations,
        args.binder_dir,
        args.ranking_method,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, float_format="%.4f", encoding="utf-8-sig")

    for message in messages:
        print(message)
    print(f"\nRows exported: {len(output):,}")
    print(f"Drugs exported: {output['Drug'].nunique():,}")
    print(f"Ranking method: {RANKING_METHODS[args.ranking_method]['label']}")
    print(f"CSV written to: {args.output}")


if __name__ == "__main__":
    main()
