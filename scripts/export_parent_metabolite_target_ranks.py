"""Export one parent drug and its metabolites using the parent's known binders.

The output is a Word-ready wide CSV with one row per compound/known-target pair
and rank, higher-is-better percentile, score, and output filename for all three
datasets. Metabolites are evaluated against the parent drug's binder list.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from export_known_target_ranks import (
    BINDER_DIR,
    DATASETS,
    DEFAULT_FILE_LOCATIONS,
    INTERIM_DIR,
    RANKING_METHODS,
    dataset_target_table,
    read_binders,
)
from export_top_k_dataset_intersection import (
    DEFAULT_GENE_MAP,
    filename_slug,
    read_gene_map,
)

# DEFAULT_PARENT = "Prochlorperazine"
# DEFAULT_METABOLITES = [
#     "N-desmethyl Prochlorperazine",
#     "Prochlorperazine Sulfoxide",
# ]
# DEFAULT_COMPOUND_ALIASES = {
#     "Prochlorperazine": "PCP",
#     "N-desmethyl Prochlorperazine": "NPCP",
#     "Prochlorperazine Sulfoxide": "PCPS",
# }
# DEFAULT_OUTPUT = INTERIM_DIR / "prochlorperazine_metabolite_known_target_ranks.csv"

# DEFAULT_PARENT = "Niclosamide"
# DEFAULT_METABOLITES = [
#     "CCL-7293q",
#     "CCL-7411k",
#     "CCL-7284h",
#     "CCL-7291o",
#     "CCL-7286j",
#     "CCL-7414n",
#     "CCL-7415o",
# ]
# DEFAULT_COMPOUND_ALIASES = {
#     "Niclosamide": "Niclosamide",
#     "CCL-7293q": "7293q",
#     "CCL-7411k": "7411k",
#     "CCL-7284h": "7284h",
#     "CCL-7286j": "7286j",
#     "CCL-7291o": "7291o",
#     "CCL-7414n": "7414n",
#     "CCL-7415o": "7415o",
# }
# DEFAULT_OUTPUT = INTERIM_DIR / "niclosamide_metabolite_known_target_ranks.csv"

# DEFAULT_PARENT = "DHEA"
# DEFAULT_METABOLITES = ["DHEAS"]
# DEFAULT_COMPOUND_ALIASES = {
#     "DHEA": "DHEA",
#     "DHEAS": "DHEAS",
# }
# DEFAULT_OUTPUT = INTERIM_DIR / "DHEA_metabolite_known_target_ranks.csv"

DEFAULT_PARENT = "BMX"
DEFAULT_METABOLITES = ["NW1001"]
DEFAULT_COMPOUND_ALIASES = {
    "BMX": "BMX",
    "NW1001": "NW1001",
}
DEFAULT_OUTPUT = INTERIM_DIR / "BMX_metabolite_known_target_ranks.csv"

DEFAULT_FIGURE_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIGURE_DPI = 600


def read_delimited_table(path: Path) -> pd.DataFrame:
    """Read a CSV or TSV table, using its suffix and delimiter as hints."""
    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    table = pd.read_csv(path, sep=separator)
    if len(table.columns) == 1:
        alternate_separator = "," if separator == "\t" else "\t"
        alternate = pd.read_csv(path, sep=alternate_separator)
        if len(alternate.columns) > 1:
            table = alternate
    return table


def read_niclosamide_ranked_scores(
    path: Path,
    extension: str,
    compound_name: str,
    ranking_method: str,
) -> pd.DataFrame:
    """Read and rank the CSV/TSV formats used by Niclosamide and CCL analogues."""
    separator = "\t" if extension.lower() == "tsv" or path.suffix.lower() == ".tsv" else ","
    scores = pd.read_csv(path, sep=separator)
    score_aliases = {
        "affinity": "minimizedAffinity",
        "CNN_affinity": "CNNaffinity",
        "CNN_pose_score": "CNNscore",
        "CNN_score": "CNNscore",
        "FilePath": "File_Name",
    }

    # The 7293q and 7411k Dataset 2 TSVs have no header and begin with an
    # empty index field. Re-read those files without consuming the first pose.
    recognized = {"minimizedAffinity", "affinity", "CNNaffinity", "CNN_affinity"}
    if separator == "\t" and not recognized.intersection(scores.columns):
        scores = pd.read_csv(path, sep=separator, header=None).dropna(axis="columns", how="all")
        if scores.shape[1] != 6:
            raise ValueError(f"unsupported headerless TSV layout with {scores.shape[1]} columns")
        scores.columns = [
            "Pose",
            "minimizedAffinity",
            "intramol",
            "CNNscore",
            "CNNaffinity",
            "File_Name",
        ]
    else:
        scores = scores.rename(columns=score_aliases)

    if "Compound" in scores.columns:
        scores = scores[scores["Compound"].astype("string").str.strip() == compound_name].copy()

    if "UNIPROT_ID" not in scores.columns:
        for id_column in ("ID", "UniProtKB", "UniProt ID", "UniProt_ID", "Entry"):
            if id_column in scores.columns:
                scores["UNIPROT_ID"] = scores[id_column]
                break
    if "UNIPROT_ID" not in scores.columns:
        if "File_Name" not in scores.columns:
            raise ValueError("missing both a UniProt ID column and File_Name")

        def accession_from_filename(value: object) -> str | pd.NA:
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

        scores["UNIPROT_ID"] = scores["File_Name"].map(accession_from_filename)

    required = {"minimizedAffinity", "CNNscore", "CNNaffinity", "File_Name"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    scores["File_Name"] = scores["File_Name"].map(lambda value: Path(str(value)).name)
    scores["UNIPROT_ID"] = scores["UNIPROT_ID"].astype("string").str.strip().str.upper()
    for column in ("minimizedAffinity", "CNNscore", "CNNaffinity"):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    scores["CNN_VS"] = scores["CNNscore"] * scores["CNNaffinity"]

    ranked = (
        scores[scores["minimizedAffinity"].lt(0)]
        .dropna(subset=["UNIPROT_ID", ranking_method])
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


def parse_aliases(alias_arguments: list[str]) -> dict[str, str]:
    """Parse repeated 'display name=file label' compound aliases."""
    aliases = dict(DEFAULT_COMPOUND_ALIASES)
    for argument in alias_arguments:
        if "=" not in argument:
            raise ValueError(
                f"Invalid compound alias {argument!r}; expected 'compound name=file label'"
            )
        compound, file_label = (part.strip() for part in argument.split("=", 1))
        if not compound or not file_label:
            raise ValueError(f"Invalid compound alias {argument!r}")
        aliases[compound] = file_label
    return aliases


def export_parent_and_metabolites(
    parent_drug: str,
    metabolites: list[str],
    file_locations_file: Path,
    binder_dir: Path,
    ranking_method: str,
    compound_aliases: dict[str, str],
    gene_map: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """Build a cross-dataset table using the parent binder set for all compounds."""
    binder_file = binder_dir / f"{parent_drug}_filtered_total.txt"
    if not binder_file.is_file():
        raise FileNotFoundError(f"Parent binder file not found: {binder_file}")
    parent_binders = read_binders(binder_file)
    if not parent_binders:
        raise ValueError(f"Parent binder file is empty: {binder_file}")

    locations = read_delimited_table(file_locations_file)
    required_columns = {"Compound", "Dataset", "File_location"}
    missing_columns = required_columns.difference(locations.columns)
    if missing_columns:
        raise ValueError(f"{file_locations_file} is missing columns: {sorted(missing_columns)}")

    compounds = [parent_drug, *metabolites]
    if len(set(compounds)) != len(compounds):
        raise ValueError("The parent drug and metabolite names must be unique")

    compound_tables = []
    messages = []
    for compound_order, compound in enumerate(compounds):
        compound_table = pd.DataFrame({"UNIPROT_ID": sorted(parent_binders)})
        file_compound_name = compound_aliases.get(compound, compound)

        for dataset in DATASETS:
            matching_locations = locations[
                (locations["Compound"] == compound) & (locations["Dataset"] == dataset)
            ]
            if matching_locations.empty:
                messages.append(f"{compound}, Dataset {dataset}: no file-location entry")
                continue
            location = matching_locations.iloc[0]
            dock_file = INTERIM_DIR / str(location["File_location"])
            if not dock_file.is_file():
                messages.append(f"{compound}, Dataset {dataset}: missing {dock_file}")
                continue

            try:
                ranked_scores = read_niclosamide_ranked_scores(
                    dock_file,
                    str(location.get("Extension", dock_file.suffix.lstrip("."))),
                    file_compound_name,
                    ranking_method,
                )
            except (OSError, ValueError, pd.errors.ParserError) as error:
                messages.append(f"{compound}, Dataset {dataset}: {error}")
                continue
            if ranked_scores.empty:
                messages.append(
                    f"{compound}, Dataset {dataset}: no rows matched Compound={file_compound_name!r}"
                )
                continue

            compound_table = compound_table.merge(
                dataset_target_table(
                    ranked_scores,
                    parent_binders,
                    dataset,
                    ranking_method,
                ),
                on="UNIPROT_ID",
                how="left",
            )

        # Keep a stable output schema even when a dataset entry is absent or
        # its source table cannot be parsed.
        for dataset in DATASETS:
            for column in (
                f"Dataset {dataset} rank",
                f"Dataset {dataset} rank percentile (%)",
                f"Dataset {dataset} score",
                f"Dataset {dataset} CNNscore",
                f"Dataset {dataset} CNNaffinity",
                f"Dataset {dataset} output filename",
                f"Dataset {dataset} protein count",
            ):
                if column not in compound_table.columns:
                    compound_table[column] = pd.NA

        percentile_columns = [
            f"Dataset {dataset} rank percentile (%)"
            for dataset in DATASETS
            if f"Dataset {dataset} rank percentile (%)" in compound_table.columns
        ]
        compound_table.insert(0, "Compound", compound)
        compound_table.insert(1, "Binder source drug", parent_drug)
        compound_table["Datasets containing target"] = (
            compound_table[percentile_columns].notna().sum(axis=1)
        )
        compound_table["Best rank percentile (%)"] = compound_table[percentile_columns].max(axis=1)
        compound_table["Mean rank percentile (%)"] = compound_table[percentile_columns].mean(
            axis=1
        )
        compound_table["Ranking method"] = RANKING_METHODS[ranking_method]["label"]
        compound_table["_compound_order"] = compound_order
        compound_tables.append(compound_table)

    output = pd.concat(compound_tables, ignore_index=True, sort=False)
    output = output.sort_values(
        [
            "_compound_order",
            "Mean rank percentile (%)",
            "Best rank percentile (%)",
            "UNIPROT_ID",
        ],
        ascending=[True, False, False, True],
        na_position="last",
        kind="stable",
    ).drop(columns="_compound_order")
    output = output.rename(columns={"UNIPROT_ID": "UniProt ID"})
    output.insert(
        output.columns.get_loc("UniProt ID") + 1,
        "Gene Name",
        output["UniProt ID"].map(gene_map),
    )

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
    output = output[
        [
            "Compound",
            "Binder source drug",
            "UniProt ID",
            "Gene Name",
            *dataset_columns,
            *summary_columns,
        ]
    ]
    integer_columns = [
        column
        for dataset in DATASETS
        for column in (f"Dataset {dataset} rank", f"Dataset {dataset} protein count")
    ]
    output[integer_columns] = output[integer_columns].astype("Int64")
    return output, messages


def plot_rank_percentile_heatmap(
    output: pd.DataFrame,
    compounds: list[str],
    parent_drug: str,
    ranking_method: str,
    output_file: Path,
) -> None:
    """Plot rank percentiles for the parent and metabolites across all datasets."""
    dataset_heatmaps = []
    for dataset in DATASETS:
        percentile_column = f"Dataset {dataset} rank percentile (%)"
        dataset_data = output.pivot(
            index="UniProt ID",
            columns="Compound",
            values=percentile_column,
        ).reindex(columns=compounds)
        dataset_heatmaps.append(dataset_data)

    heatmap_data = pd.concat(dataset_heatmaps, axis=1)
    heatmap_data = heatmap_data.loc[
        heatmap_data.mean(axis=1, skipna=True)
        .sort_values(ascending=False, na_position="last")
        .index
    ]
    gene_map = (
        output[["UniProt ID", "Gene Name"]]
        .drop_duplicates("UniProt ID")
        .set_index("UniProt ID")["Gene Name"]
    )
    heatmap_data.index = [
        f"{uniprot_id} ({gene_map.loc[uniprot_id]})"
        if pd.notna(gene_map.loc[uniprot_id])
        else uniprot_id
        for uniprot_id in heatmap_data.index
    ]
    annotations = heatmap_data.map(lambda value: "" if pd.isna(value) else f"{value:.1f}")

    figure_width = max(16, 0.85 * len(heatmap_data.columns) + 4)
    figure_height = max(12, 0.50 * len(heatmap_data) + 3)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    sns.heatmap(
        heatmap_data,
        ax=axis,
        cmap="Blues",
        vmin=0,
        vmax=100,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 12},
        linewidths=0.6,
        linecolor="white",
        mask=heatmap_data.isna(),
        cbar_kws={"label": "Rank percentile (%)"},
    )
    columns_per_dataset = len(compounds)
    for boundary in range(columns_per_dataset, len(heatmap_data.columns), columns_per_dataset):
        axis.axvline(boundary, color="black", linewidth=2)
    for dataset_index, dataset in enumerate(DATASETS):
        block_center = dataset_index * columns_per_dataset + columns_per_dataset / 2
        axis.text(
            block_center,
            1.02,
            f"Dataset {dataset}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
        )
    axis.set_title(
        f"{parent_drug} and metabolites: protein rank percentiles\n"
        f"{RANKING_METHODS[ranking_method]['label']}",
        fontsize=26,
        pad=52,
    )
    axis.set_xlabel("")
    axis.set_ylabel("Known target (gene)", fontsize=20)
    axis.tick_params(axis="x", labelsize=15, rotation=45)
    axis.tick_params(axis="y", labelsize=14, rotation=0)
    colorbar = axis.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=14)
    colorbar.set_label("Rank percentile (%)", fontsize=17)
    plt.setp(axis.get_xticklabels(), ha="right", rotation_mode="anchor")
    figure.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    figure.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def combine_rank_and_percentile_columns(output: pd.DataFrame) -> pd.DataFrame:
    """Format each dataset rank and percentile as ``rank (percentile%)``."""
    formatted = output.copy()
    for dataset in DATASETS:
        rank_column = f"Dataset {dataset} rank"
        percentile_column = f"Dataset {dataset} rank percentile (%)"
        combined_column = f"Dataset {dataset} rank (percentile)"
        insert_at = formatted.columns.get_loc(rank_column)
        combined_values = [
            (pd.NA if pd.isna(rank) or pd.isna(percentile) else f"{int(rank)} ({percentile:.2f}%)")
            for rank, percentile in zip(formatted[rank_column], formatted[percentile_column])
        ]
        formatted = formatted.drop(columns=[rank_column, percentile_column])
        formatted.insert(insert_at, combined_column, combined_values)
    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-drug", default=DEFAULT_PARENT)
    parser.add_argument("--metabolites", nargs="+", default=DEFAULT_METABOLITES)
    parser.add_argument(
        "--compound-alias",
        action="append",
        default=[],
        metavar="NAME=LABEL",
        help=(
            "Map a display name to its Compound-column label; may be repeated. "
            "Defaults include PCP, NPCP, and PCPS."
        ),
    )
    parser.add_argument("--file-locations", type=Path, default=DEFAULT_FILE_LOCATIONS)
    parser.add_argument("--binder-dir", type=Path, default=BINDER_DIR)
    parser.add_argument("--gene-map", type=Path, default=DEFAULT_GENE_MAP)
    parser.add_argument(
        "--ranking-method",
        choices=RANKING_METHODS,
        default="CNN_VS",
        help="Score used to rank proteins (default: GNINA combined score)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        help="Heatmap PNG path (default: reports/figures/<parent>_metabolite_rank_percentiles.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, messages = export_parent_and_metabolites(
        args.parent_drug,
        args.metabolites,
        args.file_locations,
        args.binder_dir,
        args.ranking_method,
        parse_aliases(args.compound_alias),
        read_gene_map(args.gene_map),
    )
    heatmap_output = args.heatmap_output or (
        DEFAULT_FIGURE_DIR / f"{filename_slug(args.parent_drug)}_metabolite_rank_percentiles.png"
    )
    plot_rank_percentile_heatmap(
        output,
        [args.parent_drug, *args.metabolites],
        args.parent_drug,
        args.ranking_method,
        heatmap_output,
    )
    csv_output = combine_rank_and_percentile_columns(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.to_csv(
        args.output,
        index=False,
        float_format="%.4f",
        encoding="utf-8-sig",
    )

    for message in messages:
        print(message)
    print(f"\nParent binder source: {args.parent_drug}")
    print(f"Known binders applied to every compound: {output['UniProt ID'].nunique():,}")
    print(f"Compounds exported: {output['Compound'].nunique():,}")
    print(f"Rows exported: {len(output):,}")
    print(f"Ranking method: {RANKING_METHODS[args.ranking_method]['label']}")
    print(f"CSV written to: {args.output}")
    print(f"Heatmap written to: {heatmap_output}")
    print(f"Heatmap SVG written to: {heatmap_output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
