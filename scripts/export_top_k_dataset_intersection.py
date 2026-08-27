"""Export per-dataset top-K UniProt IDs and their cross-dataset intersection."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from export_known_target_ranks import (
    DATASETS,
    DEFAULT_FILE_LOCATIONS,
    INTERIM_DIR,
    RANKING_METHODS,
    read_ranked_scores,
)


DEFAULT_COMPOUND = "Prochlorperazine"
DEFAULT_TOP_K = 100
DEFAULT_GENE_MAP = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "uniprot_ids.tsv"
)
COMPOUND_ALIASES = {
    "Prochlorperazine": "PCP",
    "N-desmethyl Prochlorperazine": "NPCP",
    "Prochlorperazine Sulfoxide": "PCPS",
}


def filename_slug(value: str) -> str:
    """Convert a compound name to a stable, filesystem-safe stem."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "compound"


def write_id_list(ids: list[str], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("".join(f"{uniprot_id}\n" for uniprot_id in ids))


def read_gene_map(path: Path) -> dict[str, str]:
    """Read UniProt accessions and primary gene symbols from a local TSV."""
    gene_data = pd.read_csv(path, sep="\t", dtype="string")
    if {"Entry", "Gene Names"}.issubset(gene_data.columns):
        accessions = gene_data["Entry"]
        gene_names = gene_data["Gene Names"]
    elif {"From", "To"}.issubset(gene_data.columns):
        accessions = gene_data["From"]
        gene_names = gene_data["To"]
    else:
        raise ValueError(
            f"{path} must contain Entry/Gene Names or From/To columns"
        )

    mapping = {}
    for accession, names in zip(accessions, gene_names):
        if pd.isna(accession) or pd.isna(names):
            continue
        primary_gene = str(names).strip().split()[0]
        if primary_gene:
            mapping[str(accession).strip().upper()] = primary_gene
    return mapping


def load_top_k_by_dataset(
    compound: str,
    compound_label: str,
    top_k: int,
    ranking_method: str,
    file_locations_file: Path,
) -> dict[int, pd.DataFrame]:
    """Load the top unique proteins for each of the three datasets."""
    locations = pd.read_csv(file_locations_file)
    required_columns = {"Compound", "Dataset", "File_location"}
    missing_columns = required_columns.difference(locations.columns)
    if missing_columns:
        raise ValueError(
            f"{file_locations_file} is missing columns: {sorted(missing_columns)}"
        )

    top_tables = {}
    for dataset in DATASETS:
        matching_locations = locations[
            (locations["Compound"] == compound)
            & (locations["Dataset"] == dataset)
        ]
        if matching_locations.empty:
            raise ValueError(f"No file-location entry for {compound}, Dataset {dataset}")
        location = matching_locations.iloc[0]
        dock_file = INTERIM_DIR / str(location["File_location"])
        if not dock_file.is_file():
            raise FileNotFoundError(dock_file)

        ranked = read_ranked_scores(
            dock_file,
            str(location.get("Extension", dock_file.suffix.lstrip("."))),
            compound,
            dataset,
            ranking_method,
            compound_name=compound_label,
        )
        if len(ranked) < top_k:
            raise ValueError(
                f"{compound}, Dataset {dataset} has only {len(ranked):,} ranked "
                f"proteins, fewer than requested top {top_k:,}"
            )
        top_tables[dataset] = ranked.head(top_k).copy()
    return top_tables


def build_intersection_table(
    top_tables: dict[int, pd.DataFrame],
    ranking_method: str,
    gene_map: dict[str, str],
) -> pd.DataFrame:
    """Return proteins occurring in at least two dataset top-K lists."""
    all_ids = sorted(
        set().union(*(set(table["UNIPROT_ID"]) for table in top_tables.values()))
    )
    rows = []
    for uniprot_id in all_ids:
        row: dict[str, object] = {
            "UniProt ID": uniprot_id,
            "Gene Name": gene_map.get(uniprot_id, pd.NA),
        }
        memberships = []
        ranks = []
        for dataset in DATASETS:
            match = top_tables[dataset][
                top_tables[dataset]["UNIPROT_ID"] == uniprot_id
            ]
            present = not match.empty
            row[f"In Dataset {dataset} top K"] = present
            row[f"Dataset {dataset} rank"] = (
                int(match.iloc[0]["Rank"]) if present else pd.NA
            )
            row[f"Dataset {dataset} score"] = (
                float(match.iloc[0][ranking_method]) if present else pd.NA
            )
            if present:
                memberships.append(dataset)
                ranks.append(int(match.iloc[0]["Rank"]))

        if len(memberships) < 2:
            continue
        membership_text = " & ".join(f"Dataset {dataset}" for dataset in memberships)
        row["Intersection"] = membership_text
        row["Dataset count"] = len(memberships)
        row["Best rank"] = min(ranks)
        row["Mean rank"] = sum(ranks) / len(ranks)
        row["In Dataset 1 & 2 intersection"] = all(
            row[f"In Dataset {dataset} top K"] for dataset in (1, 2)
        )
        row["In Dataset 1 & 3 intersection"] = all(
            row[f"In Dataset {dataset} top K"] for dataset in (1, 3)
        )
        row["In Dataset 2 & 3 intersection"] = all(
            row[f"In Dataset {dataset} top K"] for dataset in (2, 3)
        )
        row["In all 3 datasets"] = len(memberships) == 3
        rows.append(row)

    intersection = pd.DataFrame(rows)
    if intersection.empty:
        return intersection
    intersection = intersection.sort_values(
        ["Dataset count", "Mean rank", "Best rank", "UniProt ID"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    rank_columns = [f"Dataset {dataset} rank" for dataset in DATASETS]
    intersection[rank_columns] = intersection[rank_columns].astype("Int64")
    return intersection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compound", default=DEFAULT_COMPOUND)
    parser.add_argument(
        "--compound-label",
        help=(
            "Value in the docking files' Compound column. Defaults to the compound "
            "name, with built-in PCP/NPCP/PCPS aliases."
        ),
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--ranking-method",
        choices=RANKING_METHODS,
        default="CNN_VS",
        help="Score used for ranking (default: GNINA combined score)",
    )
    parser.add_argument("--file-locations", type=Path, default=DEFAULT_FILE_LOCATIONS)
    parser.add_argument("--gene-map", type=Path, default=DEFAULT_GENE_MAP)
    parser.add_argument("--output-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument(
        "--output-prefix",
        help="Filename prefix (default: a slug generated from the compound name)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    compound_label = args.compound_label or COMPOUND_ALIASES.get(
        args.compound, args.compound
    )
    output_prefix = args.output_prefix or filename_slug(args.compound)
    top_tables = load_top_k_by_dataset(
        args.compound,
        compound_label,
        args.top_k,
        args.ranking_method,
        args.file_locations,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    text_files = []
    for dataset in DATASETS:
        ids = top_tables[dataset]["UNIPROT_ID"].tolist()
        output_file = (
            args.output_dir
            / f"{output_prefix}_dataset{dataset}_top{args.top_k}_uniprot_ids.txt"
        )
        write_id_list(ids, output_file)
        text_files.append(output_file)

    gene_map = read_gene_map(args.gene_map)
    intersection = build_intersection_table(
        top_tables, args.ranking_method, gene_map
    )
    if not intersection.empty:
        intersection.insert(0, "Compound", args.compound)
        intersection.insert(1, "Top K", args.top_k)
        intersection.insert(
            2,
            "Ranking method",
            RANKING_METHODS[args.ranking_method]["label"],
        )
    csv_file = (
        args.output_dir
        / f"{output_prefix}_top{args.top_k}_dataset_intersection.csv"
    )
    intersection.to_csv(csv_file, index=False, float_format="%.4f", encoding="utf-8-sig")

    sets = {
        dataset: set(top_tables[dataset]["UNIPROT_ID"]) for dataset in DATASETS
    }
    print(f"Compound: {args.compound} (file label: {compound_label})")
    print(f"Ranking method: {RANKING_METHODS[args.ranking_method]['label']}")
    print(f"Dataset 1 & 2 intersection: {len(sets[1] & sets[2])}")
    print(f"Dataset 1 & 3 intersection: {len(sets[1] & sets[3])}")
    print(f"Dataset 2 & 3 intersection: {len(sets[2] & sets[3])}")
    print(f"All 3 datasets: {len(sets[1] & sets[2] & sets[3])}")
    print(f"Intersection CSV written to: {csv_file}")
    for output_file in text_files:
        print(f"Top-{args.top_k} ID list written to: {output_file}")


if __name__ == "__main__":
    main()
