from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
FIG_DIR = PROJECT_DIR / "reports" / "figures"

FILE_LOCATIONS_FILE = INTERIM_DIR / "file_locations.csv"
OUTPUT_FILE = INTERIM_DIR / "aim4_validation_sr.csv"
FIRST_TARGET_RANK_FILE = INTERIM_DIR / "aim4_first_target_rank_by_method.csv"
FIRST_TARGET_HEATMAP_FILE = FIG_DIR / "aim4_first_target_rank_heatmap.svg"
OVERALL_SUCCESS_HEATMAP_FILE = FIG_DIR / "aim4_overall_success_rate_heatmap.png"
CNN_VS_SUCCESS_FILE = INTERIM_DIR / "aim4_cnn_vs_success_rate_for_heatmap.csv"
CNN_VS_RECALL_FILE = INTERIM_DIR / "aim4_cnn_vs_recall_at_k_for_heatmap.csv"
TOTAL_TARGET_RECALL_HEATMAP_FILE = FIG_DIR / "aim4_cnn_vs_recall_total_targets_heatmap.png"
AVAILABLE_TARGET_RECALL_HEATMAP_FILE = FIG_DIR / "aim4_cnn_vs_recall_available_targets_heatmap.png"
TARGET_LIST_FILE = INTERIM_DIR / "aim4_drug_target_list_for_publication.csv"
UNIPROT_FILE = RAW_DIR / "uniprot_ids.tsv"
BINDER_DIR = RAW_DIR / "pubchem"

DRUGS = [
    "DHEA",
    "BMX",
    "NW1001",
    "Hydroxychloroquine",
    "Curcumin",
]
BINDER_SOURCE_DRUG = {"NW1001": "BMX"}

DATASETS = (1, 2, 3)
TOP_N_VALUES = (1, 10, 50, 100)
SUCCESS_TOP_K_VALUES = (1, 10, 50, 100)
RANKING_METHODS = {
    "CNNaffinity": {
        "label": "GNINA - Predicted affinity",
        "ascending": False,
    },
    "CNN_VS": {
        "label": "GNINA - Combined score",
        "ascending": False,
    },
    "minimizedAffinity": {
        "label": "smina",
        "ascending": True,
    },
}


def read_binders(path: Path) -> set[str]:
    """Read the unique, non-empty UniProt IDs from a binder file."""
    with path.open() as file:
        return {
            binder_id
            for line in file
            if (binder_id := line.strip().upper()) and binder_id != "NAN"
        }


def read_gene_names(path: Path) -> dict[str, str]:
    """Read UniProt accessions and their primary gene symbols."""
    uniprot = pd.read_csv(path, sep="\t", usecols=["Entry", "Gene Names"], dtype="string")
    uniprot["Entry"] = uniprot["Entry"].str.strip().str.upper()
    uniprot["Gene Name"] = uniprot["Gene Names"].str.strip().str.split().str[0]
    return (
        uniprot.dropna(subset=["Entry", "Gene Name"])
        .drop_duplicates("Entry")
        .set_index("Entry")["Gene Name"]
        .to_dict()
    )


def read_dock_scores(
    path: Path,
    extension: str,
    drug: str,
    dataset: int,
    ranking_column: str = "CNNaffinity",
    ascending: bool = False,
) -> pd.DataFrame:
    """Read, standardize, filter, and rank one docking-score dataframe."""
    separator = "\t" if extension.lower() == "tsv" or path.suffix.lower() == ".tsv" else ","
    dock_scores = pd.read_csv(path, sep=separator)

    recognized_columns = {
        "minimizedAffinity",
        "affinity",
        "CNNaffinity",
        "CNN_affinity",
    }
    if separator == "\t" and not recognized_columns.intersection(dock_scores.columns):
        dock_scores = pd.read_csv(path, sep=separator, header=None).dropna(
            axis="columns", how="all"
        )
        if dock_scores.shape[1] != 6:
            raise ValueError(
                f"unsupported headerless TSV layout with {dock_scores.shape[1]} columns"
            )
        dock_scores.columns = [
            "Pose",
            "minimizedAffinity",
            "intramol",
            "CNNscore",
            "CNNaffinity",
            "File_Name",
        ]

    rename_columns = {
        "CNN_affinity": "CNNaffinity",
        "CNN_pose_score": "CNNscore",
        "CNN_score": "CNNscore",
        "affinity": "minimizedAffinity",
    }
    dock_scores = dock_scores.rename(
        columns={old: new for old, new in rename_columns.items() if old in dock_scores.columns}
    )

    if "Compound" in dock_scores.columns:
        compound_name = "PCP" if drug == "Prochlorperazine" else drug
        compound_values = dock_scores["Compound"].astype("string").str.strip()
        if drug in {"BMX", "NW1001"} and dataset == 2:
            compound_values = compound_values.str.rsplit("_", n=1).str[-1]
        dock_scores = dock_scores[compound_values == compound_name].copy()

    if "UNIPROT_ID" not in dock_scores.columns:
        if "ID" in dock_scores.columns:
            id_values = dock_scores["ID"].astype("string")
            if dataset == 1 and drug in {"BMX", "NW1001"}:
                dock_scores["UNIPROT_ID"] = id_values.str.split("-").str[1]
            else:
                dock_scores["UNIPROT_ID"] = id_values.str.split("_").str[0]
        elif "File_Name" not in dock_scores.columns:
            raise ValueError("no UNIPROT_ID, ID, or File_Name column")
        elif dataset == 1:
            dock_scores["UNIPROT_ID"] = dock_scores["File_Name"].str.split("-").str[1]
        else:
            dock_scores["UNIPROT_ID"] = dock_scores["File_Name"].str.split("_").str[0]

    required_columns = {"CNNscore", "CNNaffinity", "minimizedAffinity"}
    missing_columns = required_columns - set(dock_scores.columns)
    if missing_columns:
        raise ValueError(f"missing score columns: {sorted(missing_columns)}")

    dock_scores["UNIPROT_ID"] = dock_scores["UNIPROT_ID"].astype("string").str.strip().str.upper()
    for column in required_columns:
        dock_scores[column] = pd.to_numeric(dock_scores[column], errors="coerce")
    dock_scores["CNN_VS"] = dock_scores["CNNscore"] * dock_scores["CNNaffinity"]

    return (
        dock_scores.dropna(subset=["UNIPROT_ID", ranking_column])
        .sort_values(ranking_column, ascending=ascending)
        .drop_duplicates(subset="UNIPROT_ID")
        .reset_index(drop=True)
    )


def empty_result(drug: str, dataset: int, binder_file: Path, status: str) -> dict:
    """Create a result row for an input that could not be analyzed."""
    row = {
        "drug": drug,
        "dataset": dataset,
        "status": status,
        "binder_file": str(binder_file),
        "dock_file": pd.NA,
        "binder_count": pd.NA,
        "dock_score_count": pd.NA,
        "available_binder_count": pd.NA,
        "availability_percent": pd.NA,
    }
    for top_n in TOP_N_VALUES:
        row[f"top_{top_n}_binder_count"] = pd.NA
        row[f"success_rate_top_{top_n}_percent"] = pd.NA
        row[f"success_rate_top_{top_n}_available_percent"] = pd.NA
    return row


def plot_top_n_success_rates(curve_table: pd.DataFrame) -> None:
    """Plot top-N success-rate curves for every drug using both denominators."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dataset_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    highlighted_top_n = set(TOP_N_VALUES)
    plot_types = {
        "available_binders": {
            "column": "success_rate_available_percent",
            "label": "Available binders",
            "marker": "o",
        },
        "all_binders": {
            "column": "success_rate_percent",
            "label": "Overall",
            "marker": "^",
        },
    }

    for drug in DRUGS:
        drug_results = curve_table[curve_table["drug"] == drug]

        if drug_results.empty:
            print(f"No plottable results: {drug}")
            continue

        fig, ax = plt.subplots(figsize=(10, 7.5))
        plotted_lines = 0

        for plot_config in plot_types.values():
            for dataset in DATASETS:
                dataset_results = (
                    drug_results[drug_results["dataset"] == dataset]
                    .dropna(subset=[plot_config["column"]])
                    .sort_values("top_n")
                )
                if dataset_results.empty:
                    continue

                color = dataset_colors.get(dataset, "tab:gray")
                ax.plot(
                    dataset_results["top_n"],
                    dataset_results[plot_config["column"]],
                    color=color,
                    linewidth=1.8,
                )

                highlighted_results = dataset_results[
                    dataset_results["top_n"].isin(highlighted_top_n)
                ]
                ax.scatter(
                    highlighted_results["top_n"],
                    highlighted_results[plot_config["column"]],
                    color=color,
                    s=90,
                    marker=plot_config["marker"],
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=3,
                )
                plotted_lines += 1

        if not plotted_lines:
            plt.close(fig)
            print(f"No plottable results: {drug}")
            continue

        for top_n in sorted(highlighted_top_n):
            ax.axvline(top_n, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

        ax.set_xticks(TOP_N_VALUES)
        for tick_label, top_n in zip(ax.get_xticklabels(), TOP_N_VALUES):
            if top_n in highlighted_top_n:
                tick_label.set_fontweight("bold")

        ax.set_xlabel("Top N docking results", fontsize=17)
        ax.set_ylabel("Success rate (%)", fontsize=17)
        ax.set_title(f"{drug}: docking success rates", fontsize=20)
        ax.set_xlim(0, max(TOP_N_VALUES) + 20)
        ax.set_ylim(0, 70)
        ax.tick_params(axis="both", labelsize=15)
        ax.grid(alpha=0.3)

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=dataset_colors[dataset],
                linewidth=2.5,
                label=f"Dataset {dataset}",
            )
            for dataset in DATASETS
        ]
        legend_handles.extend(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.8,
                marker=plot_config["marker"],
                markersize=8,
                label=plot_config["label"],
            )
            for plot_config in plot_types.values()
        )
        ax.legend(handles=legend_handles, loc="upper left", fontsize=13)

        fig.tight_layout()
        filename_drug = drug.lower().replace(" ", "_")
        fig.savefig(
            FIG_DIR / f"{filename_drug}_top_n_success_rates_combined.png",
            dpi=300,
        )
        plt.close(fig)


def plot_first_target_rank_heatmap(rank_table: pd.DataFrame) -> None:
    """Plot the first rank at which a known binder is retrieved."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    valid_ranks = rank_table["first_target_rank"].dropna()
    if valid_ranks.empty:
        print("No first-target ranks available for the heatmap")
        return

    binder_counts = (
        rank_table[["drug", "binder_count"]]
        .drop_duplicates("drug")
        .set_index("drug")["binder_count"]
        .to_dict()
    )
    global_max_rank = max(1, int(valid_ranks.max()))
    fig, axes = plt.subplots(1, 3, figsize=(20, 10), sharey=True)

    for index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        heatmap_data = (
            rank_table[rank_table["dataset"] == dataset]
            .pivot(index="drug", columns="ranking_method", values="first_target_rank")
            .reindex(index=DRUGS, columns=RANKING_METHODS)
            .rename(
                columns={method: config["label"] for method, config in RANKING_METHODS.items()}
            )
            .apply(pd.to_numeric, errors="coerce")
        )
        heatmap_data.index = [
            f"{drug} (n={binder_counts[drug]})" if drug in binder_counts else f"{drug} (n=NA)"
            for drug in heatmap_data.index
        ]
        annotations = heatmap_data.map(lambda value: "" if pd.isna(value) else f"{int(value):,}")

        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap="RdYlGn_r",
            norm=LogNorm(vmin=1, vmax=global_max_rank),
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 10},
            linewidths=0.5,
            linecolor="white",
            mask=heatmap_data.isna(),
            cbar=index == len(DATASETS) - 1,
            cbar_kws={"label": "First target rank (log scale)"}
            if index == len(DATASETS) - 1
            else None,
        )
        ax.set_title(f"Dataset {dataset}", fontsize=18)
        ax.set_xlabel("Sorting method", fontsize=15)
        ax.set_ylabel("Drug" if index == 0 else "", fontsize=15)
        ax.tick_params(axis="x", labelsize=11)
        plt.setp(
            ax.get_xticklabels(),
            rotation=30,
            ha="right",
            rotation_mode="anchor",
        )
        ax.tick_params(axis="y", labelrotation=0, labelsize=11)

    fig.suptitle("First known-target retrieval rank", fontsize=22, y=0.98)
    fig.subplots_adjust(left=0.15, right=0.94, bottom=0.18, top=0.90, wspace=0.08)
    fig.savefig(FIRST_TARGET_HEATMAP_FILE, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_overall_success_rate_heatmap(success_table: pd.DataFrame) -> None:
    """Plot the percentage of drugs retrieving at least one target by Top K."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_data = (
        success_table.pivot(index="dataset", columns="top_k", values="success_rate_percent")
        .reindex(index=DATASETS, columns=SUCCESS_TOP_K_VALUES)
        .rename(
            index={dataset: f"Dataset {dataset}" for dataset in DATASETS},
            columns={top_k: f"Top {top_k}" for top_k in SUCCESS_TOP_K_VALUES},
        )
    )
    counts = success_table.pivot(
        index="dataset", columns="top_k", values="successful_drug_count"
    ).reindex(index=DATASETS, columns=SUCCESS_TOP_K_VALUES)
    annotations = heatmap_data.copy().astype("object")
    for dataset_index, dataset in enumerate(DATASETS):
        for top_k_index, top_k in enumerate(SUCCESS_TOP_K_VALUES):
            rate = heatmap_data.iloc[dataset_index, top_k_index]
            count = counts.loc[dataset, top_k]
            annotations.iloc[dataset_index, top_k_index] = (
                "" if pd.isna(rate) else f"{int(count)}/{len(DRUGS)}\n({rate:.1f}%)"
            )

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 16},
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Successful drugs / all drugs (%)"},
    )
    ax.set_title(
        "Target-retrieval success rate at K",
        fontsize=24,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("Protein ranking cutoff", fontsize=19, labelpad=10)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=16, rotation=0)
    ax.tick_params(axis="y", labelsize=16, rotation=0)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=14)
    colorbar.set_label("Successful drugs / all drugs (%)", fontsize=16, labelpad=12)
    fig.tight_layout()
    fig.savefig(OVERALL_SUCCESS_HEATMAP_FILE, dpi=600, bbox_inches="tight")
    fig.savefig(OVERALL_SUCCESS_HEATMAP_FILE.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_recall_at_k_heatmap(
    drug_recall_table: pd.DataFrame,
    recall_column: str,
    denominator_column: str,
    title: str,
    colorbar_label: str,
    output_file: Path,
) -> None:
    """Plot per-drug CNN_VS recall at each Top-K cutoff by dataset."""
    fig = plt.figure(figsize=(25, 13))
    grid = fig.add_gridspec(
        1,
        len(DATASETS) + 1,
        width_ratios=[1] * len(DATASETS) + [0.045],
        wspace=0.08,
    )
    axes = []
    for index in range(len(DATASETS)):
        axes.append(
            fig.add_subplot(
                grid[0, index],
                sharey=axes[0] if axes else None,
            )
        )
    colorbar_axis = fig.add_subplot(grid[0, -1])
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        dataset_table = drug_recall_table[drug_recall_table["dataset"] == dataset]
        heatmap_data = (
            dataset_table.pivot(index="drug", columns="top_k", values=recall_column)
            .reindex(index=DRUGS, columns=SUCCESS_TOP_K_VALUES)
            .rename(columns={top_k: f"Top {top_k}" for top_k in SUCCESS_TOP_K_VALUES})
            .apply(pd.to_numeric, errors="coerce")
        )
        retrieved = dataset_table.pivot(
            index="drug", columns="top_k", values="retrieved_binder_count"
        ).reindex(index=DRUGS, columns=SUCCESS_TOP_K_VALUES)
        denominators = dataset_table[["drug", denominator_column]].drop_duplicates("drug")
        denominators = denominators.set_index("drug")[denominator_column].reindex(DRUGS)
        annotations = heatmap_data.copy().astype("object")
        for drug_index, drug in enumerate(DRUGS):
            for top_k_index, top_k in enumerate(SUCCESS_TOP_K_VALUES):
                recall = heatmap_data.iloc[drug_index, top_k_index]
                numerator = retrieved.loc[drug, top_k]
                denominator = denominators.loc[drug]
                annotations.iloc[drug_index, top_k_index] = (
                    ""
                    if pd.isna(recall) or pd.isna(denominator) or denominator == 0
                    else f"{int(numerator)}/{int(denominator)}\n({recall:.1f}%)"
                )

        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=50,
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 10},
            linewidths=0.6,
            linecolor="white",
            cbar=index == len(DATASETS) - 1,
            cbar_ax=colorbar_axis if index == len(DATASETS) - 1 else None,
            cbar_kws={"label": colorbar_label} if index == len(DATASETS) - 1 else None,
        )
        ax.set_title(f"Dataset {dataset}", fontsize=21, fontweight="bold", pad=14)
        ax.set_xlabel("Protein ranking cutoff", fontsize=17, labelpad=10)
        ax.set_ylabel("Drug" if index == 0 else "", fontsize=17)
        ax.tick_params(axis="x", labelsize=14, rotation=0)
        ax.tick_params(axis="y", labelsize=13, rotation=0)
        if index > 0:
            ax.tick_params(axis="y", labelleft=False, left=False)
        if index == len(DATASETS) - 1:
            colorbar = ax.collections[0].colorbar
            colorbar.ax.tick_params(labelsize=13)
            colorbar.set_label(colorbar_label, fontsize=15, labelpad=10)

    fig.suptitle(title, fontsize=26, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.15, right=0.94, bottom=0.10, top=0.90, wspace=0.08)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=600, bbox_inches="tight")
    fig.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    file_locations = pd.read_csv(FILE_LOCATIONS_FILE)
    gene_names = read_gene_names(UNIPROT_FILE)
    results = []
    curve_results = []
    first_target_results = []
    cnn_vs_heatmap_results = []
    missing_binder_drugs = []
    publication_targets = {}

    for drug in DRUGS:
        binder_source_drug = BINDER_SOURCE_DRUG.get(drug, drug)
        binder_file = BINDER_DIR / f"{binder_source_drug}_filtered_total.txt"

        if not binder_file.is_file():
            missing_binder_drugs.append(drug)
            print(f"Missing binder file: {binder_file}")
            for dataset in DATASETS:
                results.append(empty_result(drug, dataset, binder_file, "missing_binder_file"))
            continue

        binders = read_binders(binder_file)
        if not binders:
            print(f"Empty binder file: {binder_file}")
            for dataset in DATASETS:
                results.append(empty_result(drug, dataset, binder_file, "empty_binder_file"))
            continue

        for target in sorted(binders):
            publication_targets[(drug, target)] = {
                "Drug": drug,
                "Target UniProt ID": target,
                "Gene Name": gene_names.get(target, pd.NA),
                **{
                    f"Dataset {dataset} percentile (rank)": pd.NA
                    for dataset in DATASETS
                },
            }

        for dataset in DATASETS:
            location_rows = file_locations[
                (file_locations["Compound"] == drug) & (file_locations["Dataset"] == dataset)
            ]

            if location_rows.empty:
                print(f"Missing docking-file entry: {drug}, dataset {dataset}")
                row = empty_result(drug, dataset, binder_file, "missing_dock_file_entry")
                row["binder_count"] = len(binders)
                results.append(row)
                continue

            location = location_rows.iloc[0]
            dock_file = INTERIM_DIR / str(location["File_location"])

            if not dock_file.is_file():
                print(f"Missing docking file: {dock_file}")
                row = empty_result(drug, dataset, binder_file, "missing_dock_file")
                row["dock_file"] = str(dock_file)
                row["binder_count"] = len(binders)
                results.append(row)
                continue

            try:
                dock_scores = read_dock_scores(
                    dock_file,
                    str(location.get("Extension", dock_file.suffix.lstrip("."))),
                    drug,
                    dataset,
                )
            except (OSError, ValueError, pd.errors.ParserError) as error:
                print(f"Could not process {dock_file}: {error}")
                row = empty_result(drug, dataset, binder_file, f"dock_file_error: {error}")
                row["dock_file"] = str(dock_file)
                row["binder_count"] = len(binders)
                results.append(row)
                continue

            ranked_scores_by_method = {}
            for method, config in RANKING_METHODS.items():
                try:
                    ranked_scores = (
                        dock_scores
                        if method == "CNNaffinity"
                        else read_dock_scores(
                            dock_file,
                            str(location.get("Extension", dock_file.suffix.lstrip("."))),
                            drug,
                            dataset,
                            ranking_column=method,
                            ascending=config["ascending"],
                        )
                    )
                    ranked_scores_by_method[method] = ranked_scores
                    binder_positions = np.flatnonzero(
                        ranked_scores["UNIPROT_ID"].isin(binders).to_numpy()
                    )
                    first_target_rank = (
                        int(binder_positions[0] + 1) if binder_positions.size else pd.NA
                    )
                except (OSError, ValueError, pd.errors.ParserError) as error:
                    print(f"Could not rank {dock_file} using {method}: {error}")
                    first_target_rank = pd.NA

                first_target_results.append(
                    {
                        "drug": drug,
                        "dataset": dataset,
                        "ranking_method": method,
                        "sorting_method": config["label"],
                        "first_target_rank": first_target_rank,
                        "binder_count": len(binders),
                    }
                )

            cnn_vs_scores = ranked_scores_by_method.get("CNN_VS")
            if cnn_vs_scores is not None:
                protein_count = len(cnn_vs_scores)
                rank_by_id = {
                    uniprot_id: rank
                    for rank, uniprot_id in enumerate(
                        cnn_vs_scores["UNIPROT_ID"], start=1
                    )
                }
                for target in sorted(binders):
                    rank = rank_by_id.get(target)
                    if rank is None:
                        continue
                    percentile = (
                        100.0
                        if protein_count <= 1
                        else 100 * (protein_count - rank) / (protein_count - 1)
                    )
                    publication_targets[(drug, target)][
                        f"Dataset {dataset} percentile (rank)"
                    ] = f"{percentile:.2f}% ({rank})"
                available_binder_count = len(binders & set(cnn_vs_scores["UNIPROT_ID"]))
                heatmap_row = {
                    "drug": drug,
                    "dataset": dataset,
                    "total_binder_count": len(binders),
                    "available_binder_count": available_binder_count,
                }
                for top_k in SUCCESS_TOP_K_VALUES:
                    top_ids = set(cnn_vs_scores.head(top_k)["UNIPROT_ID"])
                    top_binder_count = len(binders & top_ids)
                    heatmap_row[f"success_top_{top_k}"] = int(top_binder_count > 0)
                    heatmap_row[f"top_{top_k}_binder_count"] = top_binder_count
                cnn_vs_heatmap_results.append(heatmap_row)

            dock_ids = set(dock_scores["UNIPROT_ID"])
            available_binders = binders & dock_ids

            row = {
                "drug": drug,
                "dataset": dataset,
                "status": "ok",
                "binder_file": str(binder_file),
                "dock_file": str(dock_file),
                "binder_count": len(binders),
                "dock_score_count": len(dock_scores),
                "available_binder_count": len(available_binders),
                "availability_percent": 100 * len(available_binders) / len(binders),
            }

            for top_n in TOP_N_VALUES:
                top_ids = set(dock_scores.head(top_n)["UNIPROT_ID"])
                top_binder_count = len(binders & top_ids)
                row[f"top_{top_n}_binder_count"] = top_binder_count
                row[f"success_rate_top_{top_n}_percent"] = 100 * top_binder_count / len(binders)
                row[f"success_rate_top_{top_n}_available_percent"] = (
                    100 * top_binder_count / len(available_binders) if available_binders else pd.NA
                )

            results.append(row)

            cumulative_binder_count = 0
            for rank, uniprot_id in enumerate(
                dock_scores.head(max(TOP_N_VALUES))["UNIPROT_ID"], start=1
            ):
                if uniprot_id in binders:
                    cumulative_binder_count += 1

                curve_results.append(
                    {
                        "drug": drug,
                        "dataset": dataset,
                        "top_n": rank,
                        "success_rate_percent": (100 * cumulative_binder_count / len(binders)),
                        "success_rate_available_percent": (
                            100 * cumulative_binder_count / len(available_binders)
                            if available_binders
                            else pd.NA
                        ),
                    }
                )

    percentage_columns = ["availability_percent"]
    for top_n in TOP_N_VALUES:
        percentage_columns.extend(
            [
                f"success_rate_top_{top_n}_percent",
                f"success_rate_top_{top_n}_available_percent",
            ]
        )

    count_columns = [
        "binder_count",
        "dock_score_count",
        "available_binder_count",
    ]
    count_columns.extend(f"top_{top_n}_binder_count" for top_n in TOP_N_VALUES)

    column_order = (
        ["drug", "dataset", "status"]
        + percentage_columns
        + count_columns
        + ["binder_file", "dock_file"]
    )

    result_table = pd.DataFrame(results)[column_order]
    result_table.to_csv(OUTPUT_FILE, index=False)
    publication_target_table = pd.DataFrame(publication_targets.values())
    publication_target_table["_drug_order"] = publication_target_table["Drug"].map(
        {drug: index for index, drug in enumerate(DRUGS)}
    )
    publication_target_table = publication_target_table.sort_values(
        ["_drug_order", "Target UniProt ID"], kind="stable"
    ).drop(columns="_drug_order")
    publication_target_table.to_csv(TARGET_LIST_FILE, index=False, encoding="utf-8-sig")
    curve_table = pd.DataFrame(curve_results)
    first_target_table = pd.DataFrame(first_target_results)
    first_target_table.to_csv(FIRST_TARGET_RANK_FILE, index=False)
    cnn_vs_drug_success = pd.DataFrame(cnn_vs_heatmap_results)
    cnn_vs_success_rows = []
    for dataset in DATASETS:
        dataset_success = cnn_vs_drug_success[cnn_vs_drug_success["dataset"] == dataset]
        for top_k in SUCCESS_TOP_K_VALUES:
            successful_drug_count = int(dataset_success[f"success_top_{top_k}"].sum())
            cnn_vs_success_rows.append(
                {
                    "dataset": dataset,
                    "top_k": top_k,
                    "successful_drug_count": successful_drug_count,
                    "all_drug_count": len(DRUGS),
                    "analyzed_drug_count": dataset_success["drug"].nunique(),
                    "success_rate_percent": (100 * successful_drug_count / len(DRUGS)),
                }
            )
    cnn_vs_success_table = pd.DataFrame(cnn_vs_success_rows)
    cnn_vs_success_table.to_csv(CNN_VS_SUCCESS_FILE, index=False)
    cnn_vs_recall_rows = []
    for row in cnn_vs_drug_success.itertuples(index=False):
        for top_k in SUCCESS_TOP_K_VALUES:
            retrieved_binder_count = getattr(row, f"top_{top_k}_binder_count")
            cnn_vs_recall_rows.append(
                {
                    "drug": row.drug,
                    "dataset": row.dataset,
                    "top_k": top_k,
                    "retrieved_binder_count": retrieved_binder_count,
                    "total_binder_count": row.total_binder_count,
                    "available_binder_count": row.available_binder_count,
                    "recall_total_targets_percent": (
                        100 * retrieved_binder_count / row.total_binder_count
                        if row.total_binder_count
                        else pd.NA
                    ),
                    "recall_available_targets_percent": (
                        100 * retrieved_binder_count / row.available_binder_count
                        if row.available_binder_count
                        else pd.NA
                    ),
                }
            )
    cnn_vs_recall_table = pd.DataFrame(cnn_vs_recall_rows)
    cnn_vs_recall_table.to_csv(CNN_VS_RECALL_FILE, index=False)
    plot_first_target_rank_heatmap(first_target_table)
    plot_overall_success_rate_heatmap(cnn_vs_success_table)
    plot_recall_at_k_heatmap(
        cnn_vs_recall_table,
        "recall_total_targets_percent",
        "total_binder_count",
        "Per-drug Top-K recall across all known targets (highest CNN_VS pocket per protein)",
        "Recall across all known targets (%)",
        TOTAL_TARGET_RECALL_HEATMAP_FILE,
    )
    plot_recall_at_k_heatmap(
        cnn_vs_recall_table,
        "recall_available_targets_percent",
        "available_binder_count",
        "Per-drug Top-K recall across available targets (highest CNN_VS pocket per protein)",
        "Recall across available targets (%)",
        AVAILABLE_TARGET_RECALL_HEATMAP_FILE,
    )

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Publication target list saved to: {TARGET_LIST_FILE}")
    print(f"First-target ranks saved to: {FIRST_TARGET_RANK_FILE}")
    print(f"CNN_VS Top-K success rates saved to: {CNN_VS_SUCCESS_FILE}")
    print(f"CNN_VS Top-K recall values saved to: {CNN_VS_RECALL_FILE}")
    print(f"Overall success-rate heatmap saved to: {OVERALL_SUCCESS_HEATMAP_FILE}")
    print(f"Total-target recall heatmap saved to: {TOTAL_TARGET_RECALL_HEATMAP_FILE}")
    print(f"Available-target recall heatmap saved to: {AVAILABLE_TARGET_RECALL_HEATMAP_FILE}")
    print(f"Plots saved to: {FIG_DIR}")
    if missing_binder_drugs:
        print("Drugs with missing binder files: " + ", ".join(missing_binder_drugs))


if __name__ == "__main__":
    main()
