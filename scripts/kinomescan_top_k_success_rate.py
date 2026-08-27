#!/usr/bin/env python3

"""Plot Top-K docking success rates using Erlotinib/Gefitinib KinomeScan binders."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from sr_av_calculation import read_binders, read_dock_scores


PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"
FILE_LOCATIONS_FILE = INTERIM_DIR / "file_locations.csv"
DEFAULT_OUTPUT_DATA = INTERIM_DIR / "kinomescan_top_k_success_rates.csv"
DEFAULT_PERCENT_OUTPUT_DATA = INTERIM_DIR / "kinomescan_top_percent_success_rates.csv"
DEFAULT_OUTPUT_FIGURE = FIGURE_DIR / "kinomescan_top_k_success_rates.png"
DEFAULT_HEATMAP_FIGURE = FIGURE_DIR / "kinomescan_top_k_success_rate_heatmaps.png"
DEFAULT_CNN_AFFINITY_HEATMAP_FIGURE = (
    FIGURE_DIR / "kinomescan_top_k_success_rate_heatmaps_cnn_affinity.png"
)
DEFAULT_CNN_AFFINITY_1000NM_HEATMAP_FIGURE = (
    FIGURE_DIR / "kinomescan_top_k_success_rate_heatmaps_cnn_affinity_1000nm.png"
)
DEFAULT_CNN_VS_1000NM_HEATMAP_FIGURE = (
    FIGURE_DIR / "kinomescan_top_k_success_rate_heatmaps_cnn_vs_1000nm.png"
)
DEFAULT_CNN_VS_100NM_HEATMAP_FIGURE = (
    FIGURE_DIR / "kinomescan_top_k_success_rate_heatmaps_cnn_vs_100nm.png"
)
DEFAULT_CNN_AFFINITY_100NM_HEATMAP_FIGURE = (
    FIGURE_DIR / "kinomescan_top_k_success_rate_heatmaps_cnn_affinity_100nm.png"
)
PERCENT_HEATMAP_OUTPUTS = {
    ("CNN_VS", "non-NA Kd"): FIGURE_DIR / "kinomescan_top_percent_heatmaps_cnn_vs.png",
    ("CNNaffinity", "non-NA Kd"): FIGURE_DIR / "kinomescan_top_percent_heatmaps_cnn_affinity.png",
    ("CNN_VS", "Kd <= 1000 nM"): FIGURE_DIR / "kinomescan_top_percent_heatmaps_cnn_vs_1000nm.png",
    ("CNNaffinity", "Kd <= 1000 nM"): FIGURE_DIR / "kinomescan_top_percent_heatmaps_cnn_affinity_1000nm.png",
    ("CNN_VS", "Kd <= 100 nM"): FIGURE_DIR / "kinomescan_top_percent_heatmaps_cnn_vs_100nm.png",
    ("CNNaffinity", "Kd <= 100 nM"): FIGURE_DIR / "kinomescan_top_percent_heatmaps_cnn_affinity_100nm.png",
}
DEFAULT_OVERLAP_FIGURE = FIGURE_DIR / "kinomescan_top_k_target_overlap.png"
DEFAULT_OVERLAP_DATA = INTERIM_DIR / "kinomescan_top_k_target_overlap.csv"

DRUGS = ("Erlotinib", "Gefitinib")
DATASETS = (1, 2, 3)
MAX_K = 100
HEATMAP_K_VALUES = (1, 10, 50, 100)
TOP_PERCENT_VALUES = (1, 5, 10, 20)
BINDER_FILES = {
    "Erlotinib": INTERIM_DIR / "erlotinib_binding_uniprot_ids.txt",
    "Gefitinib": INTERIM_DIR / "gefitinib_binding_uniprot_ids.txt",
}
KINOMESCAN_FILES = {
    "Erlotinib": INTERIM_DIR / "erlotinib_kinomescan.csv",
    "Gefitinib": INTERIM_DIR / "gefitinib_kinomescan.csv",
}
DATASET_COLORS = {1: "#377eb8", 2: "#ff7f00", 3: "#4daf4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-k",
        type=int,
        default=MAX_K,
        help=f"Largest protein-ranking cutoff to calculate (maximum: {MAX_K}).",
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=DEFAULT_OUTPUT_DATA,
        help=f"Output long-format CSV (default: {DEFAULT_OUTPUT_DATA}).",
    )
    parser.add_argument(
        "--percent-output-data",
        type=Path,
        default=DEFAULT_PERCENT_OUTPUT_DATA,
        help=f"Output percentage-cutoff CSV (default: {DEFAULT_PERCENT_OUTPUT_DATA}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FIGURE,
        help=f"Output plot (default: {DEFAULT_OUTPUT_FIGURE}).",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=DEFAULT_HEATMAP_FIGURE,
        help=f"Output stacked heatmaps (default: {DEFAULT_HEATMAP_FIGURE}).",
    )
    parser.add_argument(
        "--cnn-affinity-heatmap-output",
        type=Path,
        default=DEFAULT_CNN_AFFINITY_HEATMAP_FIGURE,
        help=(
            "Output stacked heatmaps ranked by CNNaffinity "
            f"(default: {DEFAULT_CNN_AFFINITY_HEATMAP_FIGURE})."
        ),
    )
    parser.add_argument(
        "--cnn-affinity-1000nm-heatmap-output",
        type=Path,
        default=DEFAULT_CNN_AFFINITY_1000NM_HEATMAP_FIGURE,
        help=(
            "Output CNNaffinity-ranked heatmaps using only KINOMEscan targets "
            f"with Kd <= 1000 nM (default: {DEFAULT_CNN_AFFINITY_1000NM_HEATMAP_FIGURE})."
        ),
    )
    parser.add_argument(
        "--cnn-vs-1000nm-heatmap-output",
        type=Path,
        default=DEFAULT_CNN_VS_1000NM_HEATMAP_FIGURE,
        help=(
            "Output CNN_VS-ranked heatmaps using only KINOMEscan targets "
            f"with Kd <= 1000 nM (default: {DEFAULT_CNN_VS_1000NM_HEATMAP_FIGURE})."
        ),
    )
    parser.add_argument(
        "--cnn-vs-100nm-heatmap-output",
        type=Path,
        default=DEFAULT_CNN_VS_100NM_HEATMAP_FIGURE,
        help=f"Output CNN_VS heatmaps for Kd <= 100 nM binders.",
    )
    parser.add_argument(
        "--cnn-affinity-100nm-heatmap-output",
        type=Path,
        default=DEFAULT_CNN_AFFINITY_100NM_HEATMAP_FIGURE,
        help=f"Output CNNaffinity heatmaps for Kd <= 100 nM binders.",
    )
    parser.add_argument(
        "--overlap-output",
        type=Path,
        default=DEFAULT_OVERLAP_FIGURE,
        help=f"Output target-overlap plot (default: {DEFAULT_OVERLAP_FIGURE}).",
    )
    parser.add_argument(
        "--overlap-data",
        type=Path,
        default=DEFAULT_OVERLAP_DATA,
        help=f"Output target-overlap CSV (default: {DEFAULT_OVERLAP_DATA}).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_kinomescan_binders(max_kd_nm: float) -> dict[str, set[str]]:
    """Read targets with a numeric KINOMEscan Kd at or below the threshold."""
    binder_sets = {}
    for drug, path in KINOMESCAN_FILES.items():
        table = pd.read_csv(path)
        kd_column = "Kd (nM)" if "Kd (nM)" in table.columns else "Kd"
        required = {"UniProt ID", kd_column}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        kd_nm = pd.to_numeric(table[kd_column], errors="coerce")
        binder_sets[drug] = set(
            table.loc[kd_nm.le(max_kd_nm), "UniProt ID"].dropna().astype(str)
        )
    return binder_sets


def calculate_curves(
    max_k: int,
    ranking_column: str = "CNN_VS",
    binder_sets: dict[str, set[str]] | None = None,
    binder_filter: str = "non-NA Kd",
) -> pd.DataFrame:
    """Calculate cumulative KinomeScan binder recovery for each docking dataset."""
    registry = pd.read_csv(FILE_LOCATIONS_FILE)
    rows = []

    for drug in DRUGS:
        binder_file = BINDER_FILES[drug]
        if binder_sets is None:
            if not binder_file.is_file():
                raise FileNotFoundError(binder_file)
            binders = read_binders(binder_file)
        else:
            binders = binder_sets[drug]
        if not binders:
            raise ValueError(f"No UniProt IDs found in {binder_file}")

        for dataset in DATASETS:
            locations = registry[
                registry["Compound"].eq(drug) & registry["Dataset"].eq(dataset)
            ]
            if locations.empty:
                raise FileNotFoundError(
                    f"No docking-file registry entry for {drug}, dataset {dataset}"
                )
            location = locations.iloc[0]
            dock_file = INTERIM_DIR / str(location["File_location"])
            dock_scores = read_dock_scores(
                dock_file,
                str(location.get("Extension", dock_file.suffix.lstrip("."))),
                drug,
                dataset,
                ranking_column=ranking_column,
                ascending=False,
            )

            available_binders = binders & set(dock_scores["UNIPROT_ID"])
            cumulative_count = 0
            for rank, uniprot_id in enumerate(
                dock_scores.head(max_k)["UNIPROT_ID"], start=1
            ):
                if uniprot_id in binders:
                    cumulative_count += 1
                rows.append(
                    {
                        "drug": drug,
                        "dataset": dataset,
                        "ranking_method": ranking_column,
                        "binder_filter": binder_filter,
                        "top_k": rank,
                        "ranked_uniprot_id": uniprot_id,
                        "is_binder": uniprot_id in binders,
                        "retrieved_binder_count": cumulative_count,
                        "total_binder_count": len(binders),
                        "available_binder_count": len(available_binders),
                        "success_rate_all_percent": (
                            100 * cumulative_count / len(binders)
                        ),
                        "success_rate_available_percent": (
                            100 * cumulative_count / len(available_binders)
                            if available_binders
                            else np.nan
                        ),
                        "binder_file": str(binder_file),
                        "dock_file": str(dock_file),
                    }
                )
            print(
                f"{drug}, dataset {dataset}: {len(binders)} binders; "
                f"{len(available_binders)} available in {len(dock_scores)} ranked targets"
            )

    return pd.DataFrame(rows)


def calculate_percent_curves(
    ranking_column: str,
    binder_sets: dict[str, set[str]] | None = None,
    binder_filter: str = "non-NA Kd",
) -> pd.DataFrame:
    """Calculate recovery at percentage cutoffs using each complete ranking."""
    registry = pd.read_csv(FILE_LOCATIONS_FILE)
    rows = []
    for drug in DRUGS:
        if binder_sets is None:
            binders = read_binders(BINDER_FILES[drug])
        else:
            binders = binder_sets[drug]
        for dataset in DATASETS:
            location = registry[
                registry["Compound"].eq(drug) & registry["Dataset"].eq(dataset)
            ].iloc[0]
            dock_file = INTERIM_DIR / str(location["File_location"])
            dock_scores = read_dock_scores(
                dock_file,
                str(location.get("Extension", dock_file.suffix.lstrip("."))),
                drug,
                dataset,
                ranking_column=ranking_column,
                ascending=False,
            )
            ranked_ids = dock_scores["UNIPROT_ID"].tolist()
            available_binders = binders & set(ranked_ids)
            for top_percent in TOP_PERCENT_VALUES:
                top_k = min(
                    len(ranked_ids),
                    int(np.ceil(len(ranked_ids) * top_percent / 100)),
                )
                retrieved = binders & set(ranked_ids[:top_k])
                rows.append(
                    {
                        "drug": drug,
                        "dataset": dataset,
                        "ranking_method": ranking_column,
                        "binder_filter": binder_filter,
                        "top_percent": top_percent,
                        "top_k": top_k,
                        "ranked_protein_count": len(ranked_ids),
                        "retrieved_binder_count": len(retrieved),
                        "total_binder_count": len(binders),
                        "available_binder_count": len(available_binders),
                        "success_rate_all_percent": 100 * len(retrieved) / len(binders),
                    }
                )
    return pd.DataFrame(rows)


def draw_curves(curves: pd.DataFrame, output: Path, dpi: int, max_k: int) -> None:
    """Draw the two-drug Top-K success-rate figure."""
    figure, axes = plt.subplots(1, len(DRUGS), figsize=(13, 5.8), sharex=True, sharey=True)
    rate_settings = {
        "success_rate_all_percent": ("-", "All measured binders"),
        "success_rate_available_percent": ("--", "Available binders"),
    }

    for ax, drug in zip(axes, DRUGS):
        drug_curves = curves[curves["drug"].eq(drug)]
        for dataset in DATASETS:
            dataset_curve = drug_curves[drug_curves["dataset"].eq(dataset)]
            for rate_column, (line_style, _) in rate_settings.items():
                ax.plot(
                    dataset_curve["top_k"],
                    dataset_curve[rate_column],
                    color=DATASET_COLORS[dataset],
                    linestyle=line_style,
                    linewidth=2.2,
                )
        ax.set_title(drug, fontsize=18, fontweight="bold")
        ax.set_xlabel("Top K proteins ranked by CNN_VS", fontsize=13)
        ax.set_xlim(1, max_k)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=11)

    maximum_rate = curves[
        ["success_rate_all_percent", "success_rate_available_percent"]
    ].max().max()
    shared_y_max = max(5, 5 * np.ceil(maximum_rate / 5))
    axes[0].set_ylim(0, shared_y_max)
    axes[0].set_ylabel("Cumulative success rate (%)", fontsize=13)
    dataset_handles = [
        Line2D([0], [0], color=DATASET_COLORS[dataset], linewidth=2.5,
               label=f"Dataset {dataset}")
        for dataset in DATASETS
    ]
    rate_handles = [
        Line2D([0], [0], color="black", linestyle=line_style, linewidth=2.2,
               label=label)
        for line_style, label in rate_settings.values()
    ]
    figure.legend(
        handles=[*dataset_handles, *rate_handles],
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        frameon=False,
        fontsize=11,
    )
    figure.suptitle(
        "KinomeScan binder recovery among Top-K docking targets",
        fontsize=20,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.08, right=0.83, bottom=0.14, top=0.83, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def calculate_overlap(curves: pd.DataFrame) -> pd.DataFrame:
    """Count the seven exclusive Dataset 1/2/3 binder intersections."""
    region_definitions = (
        ("Dataset 1 only", lambda one, two, three: one - two - three),
        ("Dataset 2 only", lambda one, two, three: two - one - three),
        ("Dataset 3 only", lambda one, two, three: three - one - two),
        ("Datasets 1 & 2 only", lambda one, two, three: (one & two) - three),
        ("Datasets 1 & 3 only", lambda one, two, three: (one & three) - two),
        ("Datasets 2 & 3 only", lambda one, two, three: (two & three) - one),
        ("All three datasets", lambda one, two, three: one & two & three),
    )
    curves = curves[
        curves["ranking_method"].eq("CNN_VS")
        & curves["binder_filter"].eq("non-NA Kd")
    ]
    rows = []
    for drug in DRUGS:
        drug_rows = curves[curves["drug"].eq(drug)]
        for top_k in HEATMAP_K_VALUES:
            success_sets = {}
            for dataset in DATASETS:
                matches = drug_rows[
                    drug_rows["dataset"].eq(dataset)
                    & drug_rows["top_k"].le(top_k)
                    & drug_rows["is_binder"]
                ]
                success_sets[dataset] = set(matches["ranked_uniprot_id"])
            for region_order, (region, operation) in enumerate(
                region_definitions, start=1
            ):
                targets = operation(
                    success_sets[1],
                    success_sets[2],
                    success_sets[3],
                )
                rows.append(
                    {
                        "drug": drug,
                        "top_k": top_k,
                        "region_order": region_order,
                        "overlap_region": region,
                        "target_count": len(targets),
                        "uniprot_ids": ";".join(sorted(targets)),
                    }
                )
    return pd.DataFrame(rows)


def draw_overlap(overlap: pd.DataFrame, output: Path, dpi: int) -> None:
    """Plot exclusive recovered-binder intersections as stacked bars."""
    region_colors = {
        "Dataset 1 only": "#377eb8",
        "Dataset 2 only": "#ff7f00",
        "Dataset 3 only": "#4daf4a",
        "Datasets 1 & 2 only": "#984ea3",
        "Datasets 1 & 3 only": "#00a6a6",
        "Datasets 2 & 3 only": "#e6ab02",
        "All three datasets": "#d62728",
    }
    region_order = list(region_colors)
    figure, axes = plt.subplots(
        len(DRUGS),
        1,
        figsize=(10.5, 8),
        sharex=True,
        constrained_layout=True,
    )

    for ax, drug in zip(axes, DRUGS):
        matrix = (
            overlap[overlap["drug"].eq(drug)]
            .pivot(index="top_k", columns="overlap_region", values="target_count")
            .reindex(index=HEATMAP_K_VALUES, columns=region_order)
            .fillna(0)
        )
        bottom = np.zeros(len(matrix))
        for region in region_order:
            counts = matrix[region].to_numpy()
            bars = ax.bar(
                [str(top_k) for top_k in matrix.index],
                counts,
                bottom=bottom,
                color=region_colors[region],
                label=region,
                width=0.72,
            )
            for bar, count, base in zip(bars, counts, bottom):
                if count > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        base + count / 2,
                        str(int(count)),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if region == "All three datasets" else "black",
                    )
            bottom += counts
        for x_position, total in enumerate(bottom):
            ax.text(
                x_position,
                total + 0.5,
                f"Union: {int(total)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_title(drug, fontsize=17, fontweight="bold")
        ax.set_ylabel("Unique recovered binders", fontsize=12)
        ax.set_ylim(0, max(bottom) * 1.18 if max(bottom) else 1)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xlabel("Top K proteins ranked by CNN_VS", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=10,
    )
    figure.suptitle(
        "Overlap of KinomeScan binders recovered across docking datasets",
        fontsize=19,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def draw_heatmaps(
    curves: pd.DataFrame,
    output: Path,
    dpi: int,
    ranking_method: str = "CNN_VS",
    binder_filter: str = "non-NA Kd",
    shared_vmax: float | None = None,
) -> None:
    """Draw Erlotinib and Gefitinib Top-K heatmaps stacked vertically."""
    curves = curves[
        curves["ranking_method"].eq(ranking_method)
        & curves["binder_filter"].eq(binder_filter)
    ]
    if curves.empty:
        raise ValueError(
            f"No success-rate data found for {ranking_method}, {binder_filter}"
        )
    selected_k = [top_k for top_k in HEATMAP_K_VALUES if top_k <= curves["top_k"].max()]
    heatmap_values = {}
    global_maximum = 0.0

    for drug in DRUGS:
        drug_rows = curves[
            curves["drug"].eq(drug) & curves["top_k"].isin(selected_k)
        ]
        values = (
            drug_rows.pivot(
                index="dataset",
                columns="top_k",
                values="success_rate_all_percent",
            )
            .reindex(index=DATASETS, columns=selected_k)
        )
        heatmap_values[drug] = values
        global_maximum = max(global_maximum, values.max().max())
    if shared_vmax is not None:
        global_maximum = shared_vmax

    figure, axes = plt.subplots(
        len(DRUGS),
        1,
        figsize=(11, 7.5),
        sharex=True,
        constrained_layout=True,
    )
    colorbar_label = (
        f"Success rate across {binder_filter} binders (%)"
        if binder_filter != "non-NA Kd"
        else "Success rate across all measured binders (%)"
    )
    for index, (ax, drug) in enumerate(zip(axes, DRUGS)):
        values = heatmap_values[drug]
        annotations = values.copy().astype(object)
        drug_rows = curves[curves["drug"].eq(drug)]
        for dataset in DATASETS:
            for top_k in selected_k:
                match = drug_rows[
                    drug_rows["dataset"].eq(dataset)
                    & drug_rows["top_k"].eq(top_k)
                ]
                if match.empty:
                    annotations.loc[dataset, top_k] = ""
                    continue
                row = match.iloc[0]
                annotations.loc[dataset, top_k] = (
                    f"{int(row['retrieved_binder_count'])}/"
                    f"{int(row['total_binder_count'])}\n"
                    f"{row['success_rate_all_percent']:.1f}%"
                )

        sns.heatmap(
            values,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=global_maximum,
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 11},
            linewidths=1,
            linecolor="white",
            cbar=index == 0,
            cbar_kws={"label": colorbar_label}
            if index == 0
            else None,
        )
        ax.set_title(drug, fontsize=18, fontweight="bold", pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Docking dataset", fontsize=12)
        ax.set_yticklabels([f"Dataset {dataset}" for dataset in DATASETS], rotation=0)
        ax.tick_params(axis="both", labelsize=11)

    axes[-1].set_xlabel(
        f"Top K proteins ranked by {ranking_method}", fontsize=13
    )
    axes[-1].set_xticklabels(selected_k, rotation=0)
    if binder_filter != "non-NA Kd":
        figure.suptitle(
            f"KINOMEscan binding filter: {binder_filter}",
            fontsize=16,
            fontweight="bold",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def draw_percent_heatmaps(
    curves: pd.DataFrame,
    output: Path,
    dpi: int,
    ranking_method: str,
    binder_filter: str,
    shared_vmax: float,
) -> None:
    """Draw recovery heatmaps at Top 1%, 5%, 10%, and 20% cutoffs."""
    selected = curves[
        curves["ranking_method"].eq(ranking_method)
        & curves["binder_filter"].eq(binder_filter)
    ]
    figure, axes = plt.subplots(
        len(DRUGS), 1, figsize=(11, 7.5), sharex=True, constrained_layout=True
    )
    colorbar_label = (
        f"Success rate across {binder_filter} binders (%)"
        if binder_filter != "non-NA Kd"
        else "Success rate across all measured binders (%)"
    )
    for index, (ax, drug) in enumerate(zip(axes, DRUGS)):
        drug_rows = selected[selected["drug"].eq(drug)]
        values = (
            drug_rows.pivot(
                index="dataset",
                columns="top_percent",
                values="success_rate_all_percent",
            )
            .reindex(index=DATASETS, columns=TOP_PERCENT_VALUES)
        )
        annotations = values.copy().astype(object)
        for dataset in DATASETS:
            for top_percent in TOP_PERCENT_VALUES:
                row = drug_rows[
                    drug_rows["dataset"].eq(dataset)
                    & drug_rows["top_percent"].eq(top_percent)
                ].iloc[0]
                annotations.loc[dataset, top_percent] = (
                    f"{int(row['retrieved_binder_count'])}/"
                    f"{int(row['total_binder_count'])}\n"
                    f"{row['success_rate_all_percent']:.1f}%\n"
                    f"K={int(row['top_k'])}"
                )
        sns.heatmap(
            values,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=shared_vmax,
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 10},
            linewidths=1,
            linecolor="white",
            cbar=index == 0,
            cbar_kws={"label": colorbar_label} if index == 0 else None,
        )
        ax.set_title(drug, fontsize=18, fontweight="bold", pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Docking dataset", fontsize=12)
        ax.set_yticklabels([f"Dataset {dataset}" for dataset in DATASETS], rotation=0)
    axes[-1].set_xlabel(f"Top percentage ranked by {ranking_method}", fontsize=13)
    axes[-1].set_xticklabels(
        [f"{value}%" for value in TOP_PERCENT_VALUES], rotation=0
    )
    if binder_filter != "non-NA Kd":
        figure.suptitle(
            f"KINOMEscan binding filter: {binder_filter}",
            fontsize=16,
            fontweight="bold",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.max_k < 1:
        raise ValueError("--max-k must be at least 1")
    if args.max_k > MAX_K:
        raise ValueError(f"--max-k cannot exceed {MAX_K}")
    if args.dpi < 1:
        raise ValueError("--dpi must be at least 1")

    cnn_vs_curves = calculate_curves(args.max_k, ranking_column="CNN_VS")
    cnn_affinity_curves = calculate_curves(
        args.max_k, ranking_column="CNNaffinity"
    )
    threshold_binders = read_kinomescan_binders(max_kd_nm=1000)
    threshold_100nm_binders = read_kinomescan_binders(max_kd_nm=100)
    cnn_vs_1000nm_curves = calculate_curves(
        args.max_k,
        ranking_column="CNN_VS",
        binder_sets=threshold_binders,
        binder_filter="Kd <= 1000 nM",
    )
    cnn_affinity_1000nm_curves = calculate_curves(
        args.max_k,
        ranking_column="CNNaffinity",
        binder_sets=threshold_binders,
        binder_filter="Kd <= 1000 nM",
    )
    cnn_vs_100nm_curves = calculate_curves(
        args.max_k,
        ranking_column="CNN_VS",
        binder_sets=threshold_100nm_binders,
        binder_filter="Kd <= 100 nM",
    )
    cnn_affinity_100nm_curves = calculate_curves(
        args.max_k,
        ranking_column="CNNaffinity",
        binder_sets=threshold_100nm_binders,
        binder_filter="Kd <= 100 nM",
    )
    curves = pd.concat(
        [
            cnn_vs_curves,
            cnn_affinity_curves,
            cnn_vs_1000nm_curves,
            cnn_affinity_1000nm_curves,
            cnn_vs_100nm_curves,
            cnn_affinity_100nm_curves,
        ],
        ignore_index=True,
    )
    percent_curves = pd.concat(
        [
            calculate_percent_curves("CNN_VS"),
            calculate_percent_curves("CNNaffinity"),
            calculate_percent_curves(
                "CNN_VS", threshold_binders, "Kd <= 1000 nM"
            ),
            calculate_percent_curves(
                "CNNaffinity", threshold_binders, "Kd <= 1000 nM"
            ),
            calculate_percent_curves(
                "CNN_VS", threshold_100nm_binders, "Kd <= 100 nM"
            ),
            calculate_percent_curves(
                "CNNaffinity", threshold_100nm_binders, "Kd <= 100 nM"
            ),
        ],
        ignore_index=True,
    )
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(args.output_data, index=False)
    args.percent_output_data.parent.mkdir(parents=True, exist_ok=True)
    percent_curves.to_csv(args.percent_output_data, index=False)
    overlap = calculate_overlap(curves)
    args.overlap_data.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(args.overlap_data, index=False)
    non_na_vmax = curves.loc[
        curves["binder_filter"].eq("non-NA Kd"),
        "success_rate_all_percent",
    ].max()
    threshold_vmax = curves.loc[
        curves["binder_filter"].eq("Kd <= 1000 nM"),
        "success_rate_all_percent",
    ].max()
    threshold_100nm_vmax = curves.loc[
        curves["binder_filter"].eq("Kd <= 100 nM"),
        "success_rate_all_percent",
    ].max()
    percent_vmax = {
        binder_filter: percent_curves.loc[
            percent_curves["binder_filter"].eq(binder_filter),
            "success_rate_all_percent",
        ].max()
        for binder_filter in ("non-NA Kd", "Kd <= 1000 nM", "Kd <= 100 nM")
    }
    draw_curves(cnn_vs_curves, args.output, args.dpi, args.max_k)
    draw_heatmaps(
        curves,
        args.heatmap_output,
        args.dpi,
        ranking_method="CNN_VS",
        shared_vmax=non_na_vmax,
    )
    draw_heatmaps(
        curves,
        args.cnn_affinity_heatmap_output,
        args.dpi,
        ranking_method="CNNaffinity",
        shared_vmax=non_na_vmax,
    )
    draw_heatmaps(
        curves,
        args.cnn_vs_1000nm_heatmap_output,
        args.dpi,
        ranking_method="CNN_VS",
        binder_filter="Kd <= 1000 nM",
        shared_vmax=threshold_vmax,
    )
    draw_heatmaps(
        curves,
        args.cnn_affinity_1000nm_heatmap_output,
        args.dpi,
        ranking_method="CNNaffinity",
        binder_filter="Kd <= 1000 nM",
        shared_vmax=threshold_vmax,
    )
    draw_heatmaps(
        curves,
        args.cnn_vs_100nm_heatmap_output,
        args.dpi,
        ranking_method="CNN_VS",
        binder_filter="Kd <= 100 nM",
        shared_vmax=threshold_100nm_vmax,
    )
    draw_heatmaps(
        curves,
        args.cnn_affinity_100nm_heatmap_output,
        args.dpi,
        ranking_method="CNNaffinity",
        binder_filter="Kd <= 100 nM",
        shared_vmax=threshold_100nm_vmax,
    )
    for (ranking_method, binder_filter), output in PERCENT_HEATMAP_OUTPUTS.items():
        draw_percent_heatmaps(
            percent_curves,
            output,
            args.dpi,
            ranking_method,
            binder_filter,
            percent_vmax[binder_filter],
        )
    draw_overlap(overlap, args.overlap_output, args.dpi)
    print(f"Success-rate data saved to: {args.output_data}")
    print(f"Percentage-cutoff data saved to: {args.percent_output_data}")
    print(f"Success-rate plot saved to: {args.output}")
    print(f"Stacked heatmaps saved to: {args.heatmap_output}")
    print(
        "CNNaffinity-ranked stacked heatmaps saved to: "
        f"{args.cnn_affinity_heatmap_output}"
    )
    print(
        "CNN_VS-ranked <=1000 nM KINOMEscan heatmaps saved to: "
        f"{args.cnn_vs_1000nm_heatmap_output}"
    )
    print(
        "CNNaffinity-ranked <=1000 nM KINOMEscan heatmaps saved to: "
        f"{args.cnn_affinity_1000nm_heatmap_output}"
    )
    print(
        "CNN_VS-ranked <=100 nM KINOMEscan heatmaps saved to: "
        f"{args.cnn_vs_100nm_heatmap_output}"
    )
    print(
        "CNNaffinity-ranked <=100 nM KINOMEscan heatmaps saved to: "
        f"{args.cnn_affinity_100nm_heatmap_output}"
    )
    for output in PERCENT_HEATMAP_OUTPUTS.values():
        print(f"Percentage-cutoff heatmaps saved to: {output}")
    print(f"Target-overlap data saved to: {args.overlap_data}")
    print(f"Target-overlap plot saved to: {args.overlap_output}")


if __name__ == "__main__":
    main()
