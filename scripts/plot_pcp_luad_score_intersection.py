#!/usr/bin/env python3

"""Plot intersections of PCP/LUAD targets scoring at least 5 in Datasets 1–3."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from venn import venn


PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "reports" / "figures" / "top_target_scores_PCP_LUAD_intersection_ge5.png"
)
DEFAULT_REGION_DATA = INTERIM_DIR / "top_target_scores_PCP_LUAD_intersection_regions_ge5.csv"
INPUT_TEMPLATE = "top_target_scores_PCP_LUAD_D{dataset}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=5,
        help="Minimum total score included in each set (default: 5).",
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--region-data", type=Path, default=DEFAULT_REGION_DATA)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def calculate_regions(
    threshold: float,
) -> tuple[dict[int, set[str]], dict[str, set[str]], dict[str, str]]:
    tables = {
        dataset: pd.read_csv(INTERIM_DIR / INPUT_TEMPLATE.format(dataset=dataset))
        for dataset in (1, 2, 3)
    }
    sets = {
        dataset: set(
            table.loc[table["total_score"].ge(threshold), "uniprot_id"].dropna().astype(str)
        )
        for dataset, table in tables.items()
    }
    gene_map = (
        pd.concat(
            [table[["uniprot_id", "gene_name"]] for table in tables.values()],
            ignore_index=True,
        )
        .dropna(subset=["uniprot_id", "gene_name"])
        .drop_duplicates("uniprot_id")
        .set_index("uniprot_id")["gene_name"]
        .to_dict()
    )
    regions = {
        "Dataset 1 only": sets[1] - sets[2] - sets[3],
        "Dataset 2 only": sets[2] - sets[1] - sets[3],
        "Dataset 3 only": sets[3] - sets[1] - sets[2],
        "Datasets 1 & 2 only": (sets[1] & sets[2]) - sets[3],
        "Datasets 1 & 3 only": (sets[1] & sets[3]) - sets[2],
        "Datasets 2 & 3 only": (sets[2] & sets[3]) - sets[1],
        "All three datasets": sets[1] & sets[2] & sets[3],
    }
    return sets, regions, gene_map


def save_region_data(
    regions: dict[str, set[str]],
    gene_map: dict[str, str],
    output: Path,
) -> None:
    rows = []
    for region, proteins in regions.items():
        for protein in sorted(proteins):
            rows.append(
                {
                    "intersection_region": region,
                    "uniprot_id": protein,
                    "gene_name": gene_map.get(protein, ""),
                }
            )
        if not proteins:
            rows.append(
                {
                    "intersection_region": region,
                    "uniprot_id": pd.NA,
                    "gene_name": pd.NA,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def draw_intersection(
    dataset_sets: dict[int, set[str]],
    threshold: float,
    output: Path,
    dpi: int,
) -> None:
    figure, ax = plt.subplots(figsize=(9, 8))
    venn(
        {f"Dataset {dataset}": dataset_sets[dataset] for dataset in (1, 2, 3)},
        ax=ax,
        cmap=["#1f77b4", "#ff7f0e", "#2ca02c"],
        alpha=0.5,
        fontsize=15,
        legend_loc="upper right",
    )
    # ax.set_title(
    #     f"PCP LUAD target overlap (total score ≥ {threshold:g})",
    #     fontsize=20,
    #     pad=18,
    # )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be at least 1")
    dataset_sets, regions, gene_map = calculate_regions(args.threshold)
    save_region_data(regions, gene_map, args.region_data)
    draw_intersection(
        dataset_sets,
        args.threshold,
        args.output,
        args.dpi,
    )
    print(", ".join(f"{region}: {len(proteins)}" for region, proteins in regions.items()))
    print(f"Intersection plot saved to: {args.output}")
    print(f"Intersection members saved to: {args.region_data}")


if __name__ == "__main__":
    main()
