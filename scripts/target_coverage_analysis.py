"""Calculate and visualize known-target coverage across docking datasets.

Target coverage is the number of known binders available in a docking dataset
divided by the drug's total number of known binders. Coverage differences are
paired percentage-point changes for Dataset 3 versus Datasets 1 and 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

DEFAULT_FILE_LOCATIONS = INTERIM_DIR / "file_locations.csv"
DEFAULT_BINDER_DIR = RAW_DIR / "pubchem"
DEFAULT_COVERAGE_TABLE = INTERIM_DIR / "target_coverage_by_dataset.csv"
DEFAULT_DIFFERENCE_TABLE = INTERIM_DIR / "target_coverage_differences.csv"
DEFAULT_MACRO_TABLE = INTERIM_DIR / "target_coverage_macro_differences.csv"
DEFAULT_RECALL_TABLE = INTERIM_DIR / "target_recall_at_k.csv"
DEFAULT_MACRO_RECALL_TABLE = INTERIM_DIR / "macro_average_target_recall_at_k.csv"
DEFAULT_PRECISION_TABLE = INTERIM_DIR / "target_precision_at_k.csv"
DEFAULT_MACRO_PRECISION_TABLE = INTERIM_DIR / "macro_average_target_precision_at_k.csv"
DEFAULT_ENRICHMENT_TABLE = INTERIM_DIR / "target_enrichment_factor_at_k.csv"
DEFAULT_ENRICHMENT_SUMMARY_TABLE = INTERIM_DIR / "median_target_enrichment_factor_at_k.csv"
DEFAULT_COVERAGE_FIGURE = FIGURE_DIR / "target_coverage_heatmap.png"
DEFAULT_DIFFERENCE_FIGURE = FIGURE_DIR / "dataset3_target_coverage_difference_heatmap.png"
DEFAULT_RECALL_FIGURE_DIR = FIGURE_DIR
DEFAULT_MACRO_RECALL_FIGURE = FIGURE_DIR / "macro_average_target_recall_at_k_heatmap.png"
DEFAULT_PRECISION_FIGURE_DIR = FIGURE_DIR
DEFAULT_MACRO_PRECISION_FIGURE = FIGURE_DIR / "macro_average_target_precision_at_k_heatmap.png"
DEFAULT_ENRICHMENT_FIGURE_DIR = FIGURE_DIR
DEFAULT_ENRICHMENT_SUMMARY_FIGURE = FIGURE_DIR / "median_target_enrichment_factor_at_k_heatmap.png"
FIGURE_DPI = 600

DRUGS = [
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
TOP_K_VALUES = (1, 10, 50, 100, 200, 500)
COMPARISONS = {
    "Dataset 3 − Dataset 1": (3, 1),
    "Dataset 3 − Dataset 2": (3, 2),
}


def read_binders(path: Path) -> set[str]:
    """Read unique, non-empty UniProt IDs from a known-binder file."""
    with path.open() as file:
        return {
            binder_id
            for line in file
            if (binder_id := line.strip().upper()) and binder_id != "NAN"
        }


def read_ranked_target_ids(
    path: Path,
    extension: str,
    drug: str,
    dataset: int,
) -> list[str]:
    """Rank unique proteins by GNINA combined score as in sr_av_calculation.py."""
    separator = "\t" if extension.lower() == "tsv" or path.suffix.lower() == ".tsv" else ","
    dock_scores = pd.read_csv(path, sep=separator)
    dock_scores = dock_scores.rename(
        columns={
            "CNN_affinity": "CNNaffinity",
            "CNN_pose_score": "CNNscore",
            "CNN_score": "CNNscore",
            "affinity": "minimizedAffinity",
            "ID": "UNIPROT_ID",
        }
    )

    if "Compound" in dock_scores.columns:
        compound_name = "PCP" if drug == "Prochlorperazine" else drug
        dock_scores = dock_scores[dock_scores["Compound"] == compound_name].copy()

    if "UNIPROT_ID" not in dock_scores.columns:
        if "File_Name" not in dock_scores.columns:
            raise ValueError("no UNIPROT_ID, ID, or File_Name column")
        if dataset == 1:
            dock_scores["UNIPROT_ID"] = dock_scores["File_Name"].str.split("-").str[1]
        else:
            dock_scores["UNIPROT_ID"] = dock_scores["File_Name"].str.split("_").str[0]

    required_scores = {"CNNscore", "CNNaffinity", "minimizedAffinity"}
    missing_scores = required_scores.difference(dock_scores.columns)
    if missing_scores:
        raise ValueError(f"missing score columns: {sorted(missing_scores)}")

    for column in required_scores:
        dock_scores[column] = pd.to_numeric(dock_scores[column], errors="coerce")
    dock_scores["CNN_VS"] = dock_scores["CNNscore"] * dock_scores["CNNaffinity"]
    dock_scores["UNIPROT_ID"] = dock_scores["UNIPROT_ID"].astype("string").str.strip().str.upper()

    # Match sr_av_calculation.py by excluding failed/non-binding poses.
    ranked_scores = (
        dock_scores[dock_scores["minimizedAffinity"].lt(0)]
        .dropna(subset=["UNIPROT_ID", "CNN_VS"])
        .sort_values("CNN_VS", ascending=False)
        .drop_duplicates(subset="UNIPROT_ID")
    )
    return ranked_scores["UNIPROT_ID"].tolist()


def missing_coverage_row(
    drug: str,
    dataset: int,
    binder_file: Path,
    status: str,
    total_binders: int | object = pd.NA,
) -> dict:
    return {
        "drug": drug,
        "dataset": dataset,
        "status": status,
        "available_binders": pd.NA,
        "total_binders": total_binders,
        "target_coverage": pd.NA,
        "target_coverage_percent": pd.NA,
        "binder_file": str(binder_file),
        "dock_file": pd.NA,
    }


def calculate_target_coverage(
    file_locations_file: Path,
    binder_dir: Path,
) -> pd.DataFrame:
    """Calculate available/total known-binder coverage for every drug/dataset."""
    file_locations = pd.read_csv(file_locations_file)
    required_columns = {"Compound", "Dataset", "File_location"}
    missing_columns = required_columns.difference(file_locations.columns)
    if missing_columns:
        raise ValueError(f"{file_locations_file} is missing columns: {sorted(missing_columns)}")

    rows = []
    for drug in DRUGS:
        binder_file = binder_dir / f"{drug}_filtered_total.txt"
        if not binder_file.is_file():
            for dataset in DATASETS:
                rows.append(
                    missing_coverage_row(drug, dataset, binder_file, "missing_binder_file")
                )
            continue

        binders = read_binders(binder_file)
        if not binders:
            for dataset in DATASETS:
                rows.append(
                    missing_coverage_row(drug, dataset, binder_file, "empty_binder_file", 0)
                )
            continue

        for dataset in DATASETS:
            locations = file_locations[
                (file_locations["Compound"] == drug) & (file_locations["Dataset"] == dataset)
            ]
            if locations.empty:
                rows.append(
                    missing_coverage_row(
                        drug,
                        dataset,
                        binder_file,
                        "missing_dock_file_entry",
                        len(binders),
                    )
                )
                continue

            location = locations.iloc[0]
            dock_file = INTERIM_DIR / str(location["File_location"])
            if not dock_file.is_file():
                row = missing_coverage_row(
                    drug, dataset, binder_file, "missing_dock_file", len(binders)
                )
                row["dock_file"] = str(dock_file)
                rows.append(row)
                continue

            try:
                ranked_target_ids = read_ranked_target_ids(
                    dock_file,
                    str(location.get("Extension", dock_file.suffix.lstrip("."))),
                    drug,
                    dataset,
                )
            except (OSError, ValueError, pd.errors.ParserError) as error:
                row = missing_coverage_row(
                    drug,
                    dataset,
                    binder_file,
                    f"dock_file_error: {error}",
                    len(binders),
                )
                row["dock_file"] = str(dock_file)
                rows.append(row)
                continue

            available_binders = binders & set(ranked_target_ids)
            coverage = len(available_binders) / len(binders)
            rows.append(
                {
                    "drug": drug,
                    "dataset": dataset,
                    "status": "ok",
                    "available_binders": len(available_binders),
                    "total_binders": len(binders),
                    "target_coverage": coverage,
                    "target_coverage_percent": 100 * coverage,
                    "binder_file": str(binder_file),
                    "dock_file": str(dock_file),
                }
            )

    return pd.DataFrame(rows)


def calculate_target_recall(
    file_locations_file: Path,
    binder_dir: Path,
) -> pd.DataFrame:
    """Calculate GNINA combined-score recall against all known binders at each K."""
    file_locations = pd.read_csv(file_locations_file)
    rows = []

    for drug in DRUGS:
        binder_file = binder_dir / f"{drug}_filtered_total.txt"
        if not binder_file.is_file():
            continue
        binders = read_binders(binder_file)
        if not binders:
            continue

        for dataset in DATASETS:
            locations = file_locations[
                (file_locations["Compound"] == drug) & (file_locations["Dataset"] == dataset)
            ]
            if locations.empty:
                continue
            location = locations.iloc[0]
            dock_file = INTERIM_DIR / str(location["File_location"])
            if not dock_file.is_file():
                continue

            try:
                ranked_target_ids = read_ranked_target_ids(
                    dock_file,
                    str(location.get("Extension", dock_file.suffix.lstrip("."))),
                    drug,
                    dataset,
                )
            except (OSError, ValueError, pd.errors.ParserError) as error:
                print(f"Could not calculate recall for {drug}, Dataset {dataset}: {error}")
                continue

            available_binder_count = len(binders & set(ranked_target_ids))
            for top_k in TOP_K_VALUES:
                hit_count = len(binders & set(ranked_target_ids[:top_k]))
                recall = hit_count / len(binders)
                rows.append(
                    {
                        "drug": drug,
                        "dataset": dataset,
                        "top_k": top_k,
                        "hit_count": hit_count,
                        "total_binders": len(binders),
                        "available_binder_count": available_binder_count,
                        "ranked_target_count": len(ranked_target_ids),
                        "evaluated_target_count": min(top_k, len(ranked_target_ids)),
                        "recall": recall,
                        "recall_percent": 100 * recall,
                        "ranking_method": "GNINA combined score",
                    }
                )

    return pd.DataFrame(rows)


def calculate_target_precision(recall_table: pd.DataFrame) -> pd.DataFrame:
    """Calculate hit precision among the ranked proteins evaluated at each K."""
    precision_table = recall_table[
        [
            "drug",
            "dataset",
            "top_k",
            "hit_count",
            "evaluated_target_count",
            "ranking_method",
        ]
    ].copy()
    precision_table["precision"] = (
        precision_table["hit_count"] / precision_table["evaluated_target_count"]
    )
    precision_table["precision_percent"] = 100 * precision_table["precision"]
    return precision_table


def calculate_enrichment_factor(recall_table: pd.DataFrame) -> pd.DataFrame:
    """Calculate EF@K relative to available-binder prevalence in each dataset."""
    enrichment_table = recall_table[
        [
            "drug",
            "dataset",
            "top_k",
            "hit_count",
            "evaluated_target_count",
            "available_binder_count",
            "ranked_target_count",
            "ranking_method",
        ]
    ].copy()
    enrichment_table["top_k_hit_rate"] = (
        enrichment_table["hit_count"] / enrichment_table["evaluated_target_count"]
    )
    enrichment_table["background_hit_rate"] = (
        enrichment_table["available_binder_count"] / enrichment_table["ranked_target_count"]
    )
    enrichment_table["enrichment_factor"] = (
        enrichment_table["top_k_hit_rate"] / enrichment_table["background_hit_rate"]
    )
    enrichment_table.loc[enrichment_table["background_hit_rate"].eq(0), "enrichment_factor"] = (
        pd.NA
    )
    return enrichment_table


def summarize_enrichment_factor(enrichment_table: pd.DataFrame) -> pd.DataFrame:
    """Calculate median and interquartile enrichment factors by dataset and K."""
    summary = (
        enrichment_table.groupby(["dataset", "top_k"])["enrichment_factor"]
        .agg(
            median_enrichment_factor="median",
            q1_enrichment_factor=lambda values: values.quantile(0.25),
            q3_enrichment_factor=lambda values: values.quantile(0.75),
            analyzed_drug_count="count",
        )
        .reset_index()
        .sort_values(["dataset", "top_k"])
        .reset_index(drop=True)
    )
    summary["interquartile_range"] = (
        summary["q3_enrichment_factor"] - summary["q1_enrichment_factor"]
    )
    return summary


def calculate_macro_average_recall(recall_table: pd.DataFrame) -> pd.DataFrame:
    """Macro-average per-drug recall within each dataset and K."""
    return (
        recall_table.groupby(["dataset", "top_k"], as_index=False)
        .agg(
            macro_average_recall=("recall", "mean"),
            macro_average_recall_percent=("recall_percent", "mean"),
            analyzed_drug_count=("drug", "nunique"),
        )
        .sort_values(["dataset", "top_k"])
        .reset_index(drop=True)
    )


def calculate_macro_average_precision(precision_table: pd.DataFrame) -> pd.DataFrame:
    """Macro-average per-drug precision within each dataset and K."""
    return (
        precision_table.groupby(["dataset", "top_k"], as_index=False)
        .agg(
            macro_average_precision=("precision", "mean"),
            macro_average_precision_percent=("precision_percent", "mean"),
            analyzed_drug_count=("drug", "nunique"),
        )
        .sort_values(["dataset", "top_k"])
        .reset_index(drop=True)
    )


def calculate_coverage_differences(
    coverage_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate paired per-drug and macro-average coverage differences."""
    coverage_wide = coverage_table.pivot(
        index="drug", columns="dataset", values="target_coverage_percent"
    ).reindex(index=DRUGS, columns=DATASETS)

    difference_table = pd.DataFrame(index=DRUGS)
    for comparison, (minuend, subtrahend) in COMPARISONS.items():
        difference_table[comparison] = coverage_wide[minuend] - coverage_wide[subtrahend]
    difference_table.index.name = "drug"
    difference_table = difference_table.reset_index()

    macro_rows = []
    for comparison in COMPARISONS:
        valid_differences = difference_table[comparison].dropna()
        macro_rows.append(
            {
                "comparison": comparison,
                "macro_average_difference_percentage_points": valid_differences.mean(),
                "paired_drug_count": len(valid_differences),
                "total_drug_count": len(DRUGS),
            }
        )
    return difference_table, pd.DataFrame(macro_rows)


def save_figure(figure: plt.Figure, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    figure.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_target_coverage(coverage_table: pd.DataFrame, output_file: Path) -> None:
    """Plot available/total target coverage by drug and dataset."""
    heatmap_data = (
        coverage_table.pivot(index="drug", columns="dataset", values="target_coverage_percent")
        .reindex(index=DRUGS, columns=DATASETS)
        .rename(columns={dataset: f"Dataset {dataset}" for dataset in DATASETS})
        .apply(pd.to_numeric, errors="coerce")
    )
    available = coverage_table.pivot(
        index="drug", columns="dataset", values="available_binders"
    ).reindex(index=DRUGS, columns=DATASETS)
    total = coverage_table.pivot(index="drug", columns="dataset", values="total_binders").reindex(
        index=DRUGS, columns=DATASETS
    )

    annotations = heatmap_data.copy().astype("object")
    for drug_index, drug in enumerate(DRUGS):
        for dataset_index, dataset in enumerate(DATASETS):
            coverage = heatmap_data.iloc[drug_index, dataset_index]
            numerator = available.loc[drug, dataset]
            denominator = total.loc[drug, dataset]
            annotations.iloc[drug_index, dataset_index] = (
                ""
                if pd.isna(coverage) or pd.isna(numerator) or pd.isna(denominator)
                else f"{int(numerator)}/{int(denominator)}\n({coverage:.1f}%)"
            )

    figure, axis = plt.subplots(figsize=(10, 13))
    sns.heatmap(
        heatmap_data,
        ax=axis,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 12},
        linewidths=0.7,
        linecolor="white",
        mask=heatmap_data.isna(),
        cbar_kws={"label": "Target coverage (%)"},
    )
    axis.set_title("Known-target coverage by dataset", fontsize=23, pad=16)
    axis.set_xlabel("")
    axis.set_ylabel("Drug", fontsize=17)
    axis.tick_params(axis="x", labelsize=15, rotation=0)
    axis.tick_params(axis="y", labelsize=13, rotation=0)
    figure.tight_layout()
    save_figure(figure, output_file)


def plot_coverage_differences(
    difference_table: pd.DataFrame,
    macro_table: pd.DataFrame,
    output_file: Path,
) -> None:
    """Plot Dataset 3's per-drug and macro-average coverage differences."""
    heatmap_data = (
        difference_table.set_index("drug")
        .reindex(DRUGS)
        .apply(pd.to_numeric, errors="coerce")
    )
    macro_values = macro_table.set_index("comparison")[
        "macro_average_difference_percentage_points"
    ].reindex(COMPARISONS)
    heatmap_data.loc["Macro average"] = macro_values
    maximum = np.nanmax(np.abs(heatmap_data.to_numpy(dtype=float)))
    color_limit = max(1.0, float(maximum)) if np.isfinite(maximum) else 1.0
    annotations = heatmap_data.map(lambda value: "" if pd.isna(value) else f"{value:+.1f} pp")

    figure, axis = plt.subplots(figsize=(9, 13))
    sns.heatmap(
        heatmap_data,
        ax=axis,
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        center=0,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 12},
        linewidths=0.7,
        linecolor="white",
        mask=heatmap_data.isna(),
        cbar_kws={"label": "Coverage difference (percentage points)"},
    )
    axis.hlines(
        len(DRUGS),
        *axis.get_xlim(),
        colors="black",
        linewidth=2,
    )
    axis.set_title("Dataset 3 target-coverage difference", fontsize=23, pad=16)
    axis.set_xlabel("")
    axis.set_ylabel("Drug", fontsize=17)
    axis.tick_params(axis="x", labelsize=14, rotation=0)
    axis.tick_params(axis="y", labelsize=13, rotation=0)
    figure.tight_layout()
    save_figure(figure, output_file)


def plot_dataset_recall_heatmaps(
    recall_table: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Write one per-drug recall-at-K heatmap for each dataset."""
    output_files = []
    for dataset in DATASETS:
        dataset_table = recall_table[recall_table["dataset"] == dataset]
        heatmap_data = (
            dataset_table.pivot(index="drug", columns="top_k", values="recall_percent")
            .reindex(index=DRUGS, columns=TOP_K_VALUES)
            .rename(columns={top_k: f"Top {top_k}" for top_k in TOP_K_VALUES})
        )
        hits = dataset_table.pivot(index="drug", columns="top_k", values="hit_count").reindex(
            index=DRUGS, columns=TOP_K_VALUES
        )
        totals = (
            dataset_table[["drug", "total_binders"]]
            .drop_duplicates("drug")
            .set_index("drug")["total_binders"]
            .reindex(DRUGS)
        )
        annotations = heatmap_data.copy().astype("object")
        for drug_index, drug in enumerate(DRUGS):
            for top_k_index, top_k in enumerate(TOP_K_VALUES):
                recall = heatmap_data.iloc[drug_index, top_k_index]
                hit_count = hits.loc[drug, top_k]
                total_binders = totals.loc[drug]
                annotations.iloc[drug_index, top_k_index] = (
                    ""
                    if pd.isna(recall) or pd.isna(hit_count) or pd.isna(total_binders)
                    else f"{int(hit_count)}/{int(total_binders)}\n({recall:.1f}%)"
                )

        figure, axis = plt.subplots(figsize=(16, 13))
        sns.heatmap(
            heatmap_data,
            ax=axis,
            cmap="YlOrRd",
            vmin=0,
            vmax=100,
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 11},
            linewidths=0.7,
            linecolor="white",
            mask=heatmap_data.isna(),
            cbar_kws={"label": "Recall across all known binders (%)"},
        )
        axis.set_title(
            f"Dataset {dataset}: known-target recall at K",
            fontsize=23,
            pad=16,
        )
        axis.set_xlabel("Protein ranking cutoff", fontsize=17)
        axis.set_ylabel("Drug", fontsize=17)
        axis.tick_params(axis="x", labelsize=14, rotation=0)
        axis.tick_params(axis="y", labelsize=13, rotation=0)
        figure.tight_layout()

        output_file = output_dir / f"dataset{dataset}_target_recall_at_k_heatmap.png"
        save_figure(figure, output_file)
        output_files.append(output_file)

    return output_files


def plot_macro_average_recall(
    macro_recall_table: pd.DataFrame,
    output_file: Path,
) -> None:
    """Plot macro-averaged recall at each K for all datasets."""
    heatmap_data = (
        macro_recall_table.pivot(
            index="dataset",
            columns="top_k",
            values="macro_average_recall_percent",
        )
        .reindex(index=DATASETS, columns=TOP_K_VALUES)
        .rename(
            index={dataset: f"Dataset {dataset}" for dataset in DATASETS},
            columns={top_k: f"Top {top_k}" for top_k in TOP_K_VALUES},
        )
    )
    annotations = heatmap_data.map(lambda value: "" if pd.isna(value) else f"{value:.1f}%")

    figure, axis = plt.subplots(figsize=(15, 6))
    sns.heatmap(
        heatmap_data,
        ax=axis,
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 16},
        linewidths=0.8,
        linecolor="white",
        mask=heatmap_data.isna(),
        cbar_kws={"label": "Macro-averaged recall (%)"},
    )
    axis.set_title("Macro-averaged known-target recall at K", fontsize=23, pad=16)
    axis.set_xlabel("Protein ranking cutoff", fontsize=17)
    axis.set_ylabel("")
    axis.tick_params(axis="x", labelsize=15, rotation=0)
    axis.tick_params(axis="y", labelsize=15, rotation=0)
    figure.tight_layout()
    save_figure(figure, output_file)


def plot_dataset_precision_heatmaps(
    precision_table: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Write one per-drug precision-at-K heatmap for each dataset."""
    output_files = []
    for dataset in DATASETS:
        dataset_table = precision_table[precision_table["dataset"] == dataset]
        heatmap_data = (
            dataset_table.pivot(index="drug", columns="top_k", values="precision_percent")
            .reindex(index=DRUGS, columns=TOP_K_VALUES)
            .rename(columns={top_k: f"Top {top_k}" for top_k in TOP_K_VALUES})
        )
        hits = dataset_table.pivot(index="drug", columns="top_k", values="hit_count").reindex(
            index=DRUGS, columns=TOP_K_VALUES
        )
        evaluated = dataset_table.pivot(
            index="drug", columns="top_k", values="evaluated_target_count"
        ).reindex(index=DRUGS, columns=TOP_K_VALUES)
        annotations = heatmap_data.copy().astype("object")
        for drug_index, drug in enumerate(DRUGS):
            for top_k_index, top_k in enumerate(TOP_K_VALUES):
                precision = heatmap_data.iloc[drug_index, top_k_index]
                hit_count = hits.loc[drug, top_k]
                evaluated_count = evaluated.loc[drug, top_k]
                annotations.iloc[drug_index, top_k_index] = (
                    ""
                    if pd.isna(precision) or pd.isna(hit_count) or pd.isna(evaluated_count)
                    else f"{int(hit_count)}/{int(evaluated_count)}\n({precision:.1f}%)"
                )

        figure, axis = plt.subplots(figsize=(16, 13))
        sns.heatmap(
            heatmap_data,
            ax=axis,
            cmap="YlGnBu",
            vmin=0,
            vmax=30,
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 11},
            linewidths=0.7,
            linecolor="white",
            mask=heatmap_data.isna(),
            cbar_kws={"label": "Precision among ranked proteins (%)"},
        )
        axis.set_title(
            f"Dataset {dataset}: known-target precision at K",
            fontsize=23,
            pad=16,
        )
        axis.set_xlabel("Protein ranking cutoff", fontsize=17)
        axis.set_ylabel("Drug", fontsize=17)
        axis.tick_params(axis="x", labelsize=14, rotation=0)
        axis.tick_params(axis="y", labelsize=13, rotation=0)
        figure.tight_layout()

        output_file = output_dir / f"dataset{dataset}_target_precision_at_k_heatmap.png"
        save_figure(figure, output_file)
        output_files.append(output_file)

    return output_files


def plot_macro_average_precision(
    macro_precision_table: pd.DataFrame,
    output_file: Path,
) -> None:
    """Plot macro-averaged precision at each K for all datasets."""
    heatmap_data = (
        macro_precision_table.pivot(
            index="dataset",
            columns="top_k",
            values="macro_average_precision_percent",
        )
        .reindex(index=DATASETS, columns=TOP_K_VALUES)
        .rename(
            index={dataset: f"Dataset {dataset}" for dataset in DATASETS},
            columns={top_k: f"Top {top_k}" for top_k in TOP_K_VALUES},
        )
    )
    annotations = heatmap_data.map(lambda value: "" if pd.isna(value) else f"{value:.2f}%")

    figure, axis = plt.subplots(figsize=(15, 6))
    sns.heatmap(
        heatmap_data,
        ax=axis,
        cmap="YlGnBu",
        vmin=0,
        vmax=30,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 16},
        linewidths=0.8,
        linecolor="white",
        mask=heatmap_data.isna(),
        cbar_kws={"label": "Macro-averaged precision (%)"},
    )
    axis.set_title("Macro-averaged known-target precision at K", fontsize=23, pad=16)
    axis.set_xlabel("Protein ranking cutoff", fontsize=17)
    axis.set_ylabel("")
    axis.tick_params(axis="x", labelsize=15, rotation=0)
    axis.tick_params(axis="y", labelsize=15, rotation=0)
    figure.tight_layout()
    save_figure(figure, output_file)


def plot_dataset_enrichment_heatmaps(
    enrichment_table: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Write one per-drug enrichment-factor heatmap for each dataset."""
    color_limit = float(enrichment_table["enrichment_factor"].quantile(0.95))
    color_limit = max(1.0, color_limit)
    output_files = []

    for dataset in DATASETS:
        heatmap_data = (
            enrichment_table[enrichment_table["dataset"] == dataset]
            .pivot(index="drug", columns="top_k", values="enrichment_factor")
            .reindex(index=DRUGS, columns=TOP_K_VALUES)
            .rename(columns={top_k: f"Top {top_k}" for top_k in TOP_K_VALUES})
        )
        annotations = heatmap_data.map(lambda value: "" if pd.isna(value) else f"{value:.2f}×")

        figure, axis = plt.subplots(figsize=(16, 13))
        sns.heatmap(
            heatmap_data,
            ax=axis,
            cmap="viridis",
            vmin=0,
            vmax=color_limit,
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 11},
            linewidths=0.7,
            linecolor="white",
            mask=heatmap_data.isna(),
            cbar_kws={
                "label": f"Enrichment factor (colors capped at {color_limit:.1f}×)",
                "extend": "max",
            },
        )
        axis.set_title(
            f"Dataset {dataset}: known-target enrichment factor at K",
            fontsize=23,
            pad=16,
        )
        axis.set_xlabel("Protein ranking cutoff", fontsize=17)
        axis.set_ylabel("Drug", fontsize=17)
        axis.tick_params(axis="x", labelsize=14, rotation=0)
        axis.tick_params(axis="y", labelsize=13, rotation=0)
        figure.tight_layout()

        output_file = output_dir / f"dataset{dataset}_target_enrichment_at_k_heatmap.png"
        save_figure(figure, output_file)
        output_files.append(output_file)

    return output_files


def plot_enrichment_summary(
    enrichment_summary: pd.DataFrame,
    output_file: Path,
) -> None:
    """Plot median enrichment factors with Q1–Q3 annotations."""
    median_data = (
        enrichment_summary.pivot(
            index="dataset",
            columns="top_k",
            values="median_enrichment_factor",
        )
        .reindex(index=DATASETS, columns=TOP_K_VALUES)
        .rename(
            index={dataset: f"Dataset {dataset}" for dataset in DATASETS},
            columns={top_k: f"Top {top_k}" for top_k in TOP_K_VALUES},
        )
    )
    q1_data = enrichment_summary.pivot(
        index="dataset", columns="top_k", values="q1_enrichment_factor"
    ).reindex(index=DATASETS, columns=TOP_K_VALUES)
    q3_data = enrichment_summary.pivot(
        index="dataset", columns="top_k", values="q3_enrichment_factor"
    ).reindex(index=DATASETS, columns=TOP_K_VALUES)

    annotations = median_data.copy().astype("object")
    for dataset_index, dataset in enumerate(DATASETS):
        for top_k_index, top_k in enumerate(TOP_K_VALUES):
            median = median_data.iloc[dataset_index, top_k_index]
            q1 = q1_data.loc[dataset, top_k]
            q3 = q3_data.loc[dataset, top_k]
            annotations.iloc[dataset_index, top_k_index] = (
                ""
                if pd.isna(median) or pd.isna(q1) or pd.isna(q3)
                else f"{median:.2f}×\n[{q1:.2f}–{q3:.2f}]"
            )

    color_limit = max(1.0, float(np.nanmax(median_data.to_numpy(dtype=float))))
    figure, axis = plt.subplots(figsize=(15, 6))
    sns.heatmap(
        median_data,
        ax=axis,
        cmap="viridis",
        vmin=0,
        vmax=color_limit,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 14},
        linewidths=0.8,
        linecolor="white",
        mask=median_data.isna(),
        cbar_kws={"label": "Median enrichment factor"},
    )
    axis.set_title(
        "Median known-target enrichment factor at K (Q1–Q3)",
        fontsize=23,
        pad=16,
    )
    axis.set_xlabel("Protein ranking cutoff", fontsize=17)
    axis.set_ylabel("")
    axis.tick_params(axis="x", labelsize=15, rotation=0)
    axis.tick_params(axis="y", labelsize=15, rotation=0)
    figure.tight_layout()
    save_figure(figure, output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-locations", type=Path, default=DEFAULT_FILE_LOCATIONS)
    parser.add_argument("--binder-dir", type=Path, default=DEFAULT_BINDER_DIR)
    parser.add_argument("--coverage-table", type=Path, default=DEFAULT_COVERAGE_TABLE)
    parser.add_argument("--difference-table", type=Path, default=DEFAULT_DIFFERENCE_TABLE)
    parser.add_argument("--macro-table", type=Path, default=DEFAULT_MACRO_TABLE)
    parser.add_argument("--recall-table", type=Path, default=DEFAULT_RECALL_TABLE)
    parser.add_argument(
        "--macro-recall-table",
        type=Path,
        default=DEFAULT_MACRO_RECALL_TABLE,
    )
    parser.add_argument("--precision-table", type=Path, default=DEFAULT_PRECISION_TABLE)
    parser.add_argument(
        "--macro-precision-table",
        type=Path,
        default=DEFAULT_MACRO_PRECISION_TABLE,
    )
    parser.add_argument("--enrichment-table", type=Path, default=DEFAULT_ENRICHMENT_TABLE)
    parser.add_argument(
        "--enrichment-summary-table",
        type=Path,
        default=DEFAULT_ENRICHMENT_SUMMARY_TABLE,
    )
    parser.add_argument("--coverage-figure", type=Path, default=DEFAULT_COVERAGE_FIGURE)
    parser.add_argument("--difference-figure", type=Path, default=DEFAULT_DIFFERENCE_FIGURE)
    parser.add_argument(
        "--recall-figure-dir",
        type=Path,
        default=DEFAULT_RECALL_FIGURE_DIR,
    )
    parser.add_argument(
        "--macro-recall-figure",
        type=Path,
        default=DEFAULT_MACRO_RECALL_FIGURE,
    )
    parser.add_argument(
        "--precision-figure-dir",
        type=Path,
        default=DEFAULT_PRECISION_FIGURE_DIR,
    )
    parser.add_argument(
        "--macro-precision-figure",
        type=Path,
        default=DEFAULT_MACRO_PRECISION_FIGURE,
    )
    parser.add_argument(
        "--enrichment-figure-dir",
        type=Path,
        default=DEFAULT_ENRICHMENT_FIGURE_DIR,
    )
    parser.add_argument(
        "--enrichment-summary-figure",
        type=Path,
        default=DEFAULT_ENRICHMENT_SUMMARY_FIGURE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coverage_table = calculate_target_coverage(args.file_locations, args.binder_dir)
    difference_table, macro_table = calculate_coverage_differences(coverage_table)
    recall_table = calculate_target_recall(args.file_locations, args.binder_dir)
    macro_recall_table = calculate_macro_average_recall(recall_table)
    precision_table = calculate_target_precision(recall_table)
    macro_precision_table = calculate_macro_average_precision(precision_table)
    enrichment_table = calculate_enrichment_factor(recall_table)
    enrichment_summary = summarize_enrichment_factor(enrichment_table)

    for output_file, table in (
        (args.coverage_table, coverage_table),
        (args.difference_table, difference_table),
        (args.macro_table, macro_table),
        (args.recall_table, recall_table),
        (args.macro_recall_table, macro_recall_table),
        (args.precision_table, precision_table),
        (args.macro_precision_table, macro_precision_table),
        (args.enrichment_table, enrichment_table),
        (args.enrichment_summary_table, enrichment_summary),
    ):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_file, index=False)

    plot_target_coverage(coverage_table, args.coverage_figure)
    plot_coverage_differences(difference_table, macro_table, args.difference_figure)
    recall_figure_files = plot_dataset_recall_heatmaps(recall_table, args.recall_figure_dir)
    plot_macro_average_recall(macro_recall_table, args.macro_recall_figure)
    precision_figure_files = plot_dataset_precision_heatmaps(
        precision_table, args.precision_figure_dir
    )
    plot_macro_average_precision(macro_precision_table, args.macro_precision_figure)
    enrichment_figure_files = plot_dataset_enrichment_heatmaps(
        enrichment_table, args.enrichment_figure_dir
    )
    plot_enrichment_summary(enrichment_summary, args.enrichment_summary_figure)

    print("\nMacro-averaged coverage differences:")
    print(macro_table.to_string(index=False))
    print(f"\nCoverage table written to: {args.coverage_table}")
    print(f"Difference table written to: {args.difference_table}")
    print(f"Macro-average table written to: {args.macro_table}")
    print(f"Coverage heatmap written to: {args.coverage_figure}")
    print(f"Difference heatmap written to: {args.difference_figure}")
    print(f"Recall table written to: {args.recall_table}")
    print(f"Macro-average recall table written to: {args.macro_recall_table}")
    for output_file in recall_figure_files:
        print(f"Recall heatmap written to: {output_file}")
    print(f"Macro-average recall heatmap written to: {args.macro_recall_figure}")
    print(f"Precision table written to: {args.precision_table}")
    print(f"Macro-average precision table written to: {args.macro_precision_table}")
    for output_file in precision_figure_files:
        print(f"Precision heatmap written to: {output_file}")
    print(f"Macro-average precision heatmap written to: {args.macro_precision_figure}")
    print(f"Enrichment-factor table written to: {args.enrichment_table}")
    print(f"Enrichment summary table written to: {args.enrichment_summary_table}")
    for output_file in enrichment_figure_files:
        print(f"Enrichment heatmap written to: {output_file}")
    print(f"Median enrichment-factor heatmap written to: {args.enrichment_summary_figure}")


if __name__ == "__main__":
    main()
