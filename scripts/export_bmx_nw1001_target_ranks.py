"""Export BMX known-target ranks for BMX and NW1001 across three datasets.

This is a compound-specific counterpart to
``export_parent_metabolite_target_ranks.py``.  It handles the distinct schemas
of BMX_D1_out.csv and BMX_D2_out.csv, applies the BMX binder list to NW1001,
and writes the same Word-ready CSV and rank-percentile heatmap.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from export_known_target_ranks import (
    DATASETS,
    INTERIM_DIR,
    RANKING_METHODS,
    dataset_target_table,
    read_binders,
)
from export_parent_metabolite_target_ranks import (
    combine_rank_and_percentile_columns,
)
from export_top_k_dataset_intersection import DEFAULT_GENE_MAP, read_gene_map


PROJECT_DIR = Path(__file__).resolve().parents[1]
BINDER_DIR = PROJECT_DIR / "data" / "raw" / "pubchem"
DEFAULT_BINDER_FILE = BINDER_DIR / "BMX_filtered_total.txt"
DEFAULT_OUTPUT = INTERIM_DIR / "BMX_NW1001_known_target_ranks.csv"
DEFAULT_FIGURE = PROJECT_DIR / "reports" / "figures" / "BMX_NW1001_rank_metrics.png"
COMPOUNDS = ("BMX", "NW1001")
DATA_FILES = {
    1: INTERIM_DIR / "BMX_D1_out.csv",
    2: INTERIM_DIR / "BMX_D2_out.csv",
    3: INTERIM_DIR / "DS3_more_out.csv",
}


def extract_uniprot_id(value: object) -> str | pd.NA:
    """Extract a canonical-looking UniProt accession from a target name."""
    if pd.isna(value):
        return pd.NA
    for token in re.split(r"[-_]", Path(str(value)).name):
        accession = token.split(".", 1)[0].upper()
        if re.fullmatch(
            r"(?:[A-Z][0-9][A-Z0-9]{3}[0-9]|"
            r"[A-Z][0-9][A-Z0-9]{3}[0-9][A-Z0-9]{4})",
            accession,
        ):
            return accession
    return pd.NA


def read_bmx_scores(path: Path, dataset: int, compound: str) -> pd.DataFrame:
    """Normalize one BMX/NW1001 docking table to the shared score schema."""
    scores = pd.read_csv(path)

    if dataset == 1:
        # ID example: AF-P98164-F11-model_v4
        scores = scores[scores["Compound"].eq(compound)].copy()
        scores["File_Name"] = scores["ID"].astype("string")
        scores["UNIPROT_ID"] = scores["ID"].map(extract_uniprot_id)
    elif dataset == 2:
        # This file's nominal Compound field is actually the final portion of
        # its target name, e.g. 523_fpocket_1_NW1001. Recover the compound from
        # the suffix and retain a reconstructed target identifier for output.
        target_tail = scores["Compound"].astype("string")
        recovered_compound = target_tail.str.rsplit("_", n=1).str[-1]
        scores = scores[recovered_compound.eq(compound)].copy()
        scores["File_Name"] = (
            scores["ID"].astype("string")
            + "_"
            + scores["Site"].astype("string")
            + "_"
            + scores["Compound"].astype("string")
        )
        scores["UNIPROT_ID"] = scores["ID"].map(extract_uniprot_id)
    elif dataset == 3:
        scores = scores[scores["Compound"].eq(compound)].copy()
        scores["UNIPROT_ID"] = scores["File_Name"].map(extract_uniprot_id)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    required = {"minimizedAffinity", "CNNscore", "CNNaffinity", "File_Name"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if scores.empty:
        raise ValueError(f"{path} contains no rows for Compound={compound!r}")

    scores["UNIPROT_ID"] = scores["UNIPROT_ID"].astype("string").str.upper()
    for column in ("minimizedAffinity", "CNNscore", "CNNaffinity"):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    scores["CNN_VS"] = scores["CNNscore"] * scores["CNNaffinity"]
    return scores


def rank_scores(scores: pd.DataFrame, ranking_method: str) -> pd.DataFrame:
    """Keep the best pocket per protein and calculate rank percentiles."""
    ranked = (
        scores.dropna(subset=["UNIPROT_ID", ranking_method])
        .sort_values(
            ranking_method,
            ascending=RANKING_METHODS[ranking_method]["ascending"],
            kind="stable",
        )
        .drop_duplicates("UNIPROT_ID", keep="first")
        .reset_index(drop=True)
    )
    ranked["Rank"] = ranked.index + 1
    protein_count = len(ranked)
    ranked["Rank percentile (%)"] = (
        100.0
        if protein_count <= 1
        else 100 * (protein_count - ranked["Rank"]) / (protein_count - 1)
    )
    ranked["Ranked protein count"] = protein_count
    return ranked


def build_output(
    binder_file: Path,
    gene_map_file: Path,
    ranking_method: str,
) -> pd.DataFrame:
    """Build the wide BMX/NW1001 known-target result table."""
    if not binder_file.is_file():
        raise FileNotFoundError(f"BMX binder file not found: {binder_file}")
    binders = read_binders(binder_file)
    if not binders:
        raise ValueError(f"BMX binder file is empty: {binder_file}")
    gene_map = read_gene_map(gene_map_file)

    compound_tables = []
    for compound_order, compound in enumerate(COMPOUNDS):
        table = pd.DataFrame({"UNIPROT_ID": sorted(binders)})
        for dataset in DATASETS:
            path = DATA_FILES[dataset]
            if not path.is_file():
                raise FileNotFoundError(path)
            ranked = rank_scores(read_bmx_scores(path, dataset, compound), ranking_method)
            table = table.merge(
                dataset_target_table(ranked, binders, dataset, ranking_method),
                on="UNIPROT_ID",
                how="left",
            )

        percentile_columns = [
            f"Dataset {dataset} rank percentile (%)" for dataset in DATASETS
        ]
        table.insert(0, "Compound", compound)
        table.insert(1, "Binder source drug", "BMX")
        table["Datasets containing target"] = table[percentile_columns].notna().sum(axis=1)
        table["Best rank percentile (%)"] = table[percentile_columns].max(axis=1)
        table["Mean rank percentile (%)"] = table[percentile_columns].mean(axis=1)
        table["Ranking method"] = RANKING_METHODS[ranking_method]["label"]
        table["_compound_order"] = compound_order
        compound_tables.append(table)

    output = pd.concat(compound_tables, ignore_index=True)
    output = output.sort_values(
        ["_compound_order", "Mean rank percentile (%)", "UNIPROT_ID"],
        ascending=[True, False, True],
        na_position="last",
        kind="stable",
    ).drop(columns="_compound_order")
    output = output.rename(columns={"UNIPROT_ID": "UniProt ID"})
    output.insert(
        output.columns.get_loc("UniProt ID") + 1,
        "Gene Name",
        output["UniProt ID"].map(gene_map),
    )

    ordered_columns = ["Compound", "Binder source drug", "UniProt ID", "Gene Name"]
    for dataset in DATASETS:
        ordered_columns.extend(
            [
                f"Dataset {dataset} rank",
                f"Dataset {dataset} rank percentile (%)",
                f"Dataset {dataset} score",
                f"Dataset {dataset} CNNscore",
                f"Dataset {dataset} CNNaffinity",
                f"Dataset {dataset} output filename",
                f"Dataset {dataset} protein count",
            ]
        )
    ordered_columns.extend(
        [
            "Datasets containing target",
            "Best rank percentile (%)",
            "Mean rank percentile (%)",
            "Ranking method",
        ]
    )
    return output[ordered_columns]


def plot_rank_metrics(output: pd.DataFrame, output_file: Path) -> None:
    """Compare rank percentile, CNNscore, and CNNaffinity as grouped bars."""
    metrics = (
        ("rank percentile (%)", "Rank percentile (%)", (0, 100)),
        ("CNNscore", "CNNscore", (0, 1)),
        ("CNNaffinity", "CNNaffinity", None),
    )
    dataset_positions = np.arange(len(DATASETS), dtype=float)
    bar_width = 0.34
    offsets = (-bar_width / 2, bar_width / 2)
    colors = ("tab:blue", "tab:orange")

    figure, axes = plt.subplots(1, 3, figsize=(18, 7))
    for axis, (column_suffix, ylabel, limits) in zip(axes, metrics):
        for compound, offset, color in zip(COMPOUNDS, offsets, colors):
            compound_row = output[output["Compound"].eq(compound)].iloc[0]
            values = [
                float(compound_row[f"Dataset {dataset} {column_suffix}"])
                for dataset in DATASETS
            ]
            bars = axis.bar(
                dataset_positions + offset,
                values,
                width=bar_width,
                color=color,
                label=compound,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.1f}" if column_suffix == "rank percentile (%)" else f"{value:.2f}" for value in values],
                padding=4,
                fontsize=12,
                rotation=0,
            )

        axis.set_xticks(dataset_positions, [f"Dataset {dataset}" for dataset in DATASETS])
        axis.set_ylabel(ylabel, fontsize=17)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(axis="y", alpha=0.25)
        if limits is not None:
            axis.set_ylim(*limits)
        else:
            maximum = max(patch.get_height() for patch in axis.patches)
            axis.set_ylim(0, maximum * 1.18)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(COMPOUNDS),
        frameon=False,
        fontsize=16,
    )
    gene_label = output.iloc[0]["Gene Name"]
    target_label = output.iloc[0]["UniProt ID"]
    if pd.notna(gene_label):
        target_label = f"{target_label} ({gene_label})"
    figure.suptitle(
        f"BMX and NW1001 ranking metrics for known target {target_label}",
        fontsize=22,
        y=1.00,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=600, bbox_inches="tight")
    figure.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binder-file", type=Path, default=DEFAULT_BINDER_FILE)
    parser.add_argument("--gene-map", type=Path, default=DEFAULT_GENE_MAP)
    parser.add_argument("--ranking-method", choices=RANKING_METHODS, default="CNN_VS")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_output(args.binder_file, args.gene_map, args.ranking_method)
    plot_rank_metrics(output, args.figure_output)
    csv_output = combine_rank_and_percentile_columns(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.to_csv(args.output, index=False, float_format="%.4f", encoding="utf-8-sig")

    print(f"BMX binders: {output['UniProt ID'].nunique():,}")
    print(f"Rows exported: {len(output):,}")
    for dataset in DATASETS:
        column = f"Dataset {dataset} rank percentile (%)"
        print(f"Dataset {dataset} ranks available: {output[column].notna().sum():,}/{len(output):,}")
    print(f"CSV written to: {args.output}")
    print(f"Figure written to: {args.figure_output}")
    print(f"Figure SVG written to: {args.figure_output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
