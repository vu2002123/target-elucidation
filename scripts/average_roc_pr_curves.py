from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd
from scipy.stats import t
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
FIG_DIR = PROJECT_DIR / "reports" / "figures"

FILE_LOCATIONS_FILE = INTERIM_DIR / "file_locations.csv"
BINDER_DIR = RAW_DIR / "pubchem"
ROC_OUTPUT_FILE = FIG_DIR / "macro_averaged_auroc_comparison.png"
PR_OUTPUT_FILE = FIG_DIR / "micro_averaged_auprc_comparison.png"
METRICS_OUTPUT_FILE = INTERIM_DIR / "average_roc_pr_metrics.csv"
HEATMAP_OUTPUT_FILE = FIG_DIR / "drug_auroc_heatmap.png"
DRUG_AUC_OUTPUT_FILE = INTERIM_DIR / "drug_auroc_by_method.csv"
POCKET_REFERENCE_FILE = INTERIM_DIR / "D1_validation_90cp_out.csv"
ZSCORE_ROC_OUTPUT_FILE = FIG_DIR / "dataset3_zscore_cnn_affinity_auroc.png"
ZSCORE_PR_OUTPUT_FILE = FIG_DIR / "dataset3_zscore_cnn_affinity_auprc.png"
ZSCORE_METRICS_OUTPUT_FILE = INTERIM_DIR / "dataset3_zscore_cnn_affinity_metrics.csv"
ZSCORE_DRUG_AUC_OUTPUT_FILE = INTERIM_DIR / "dataset3_zscore_drug_auroc.csv"
ZSCORE_HEATMAP_OUTPUT_FILE = FIG_DIR / "dataset3_auroc_heatmap_with_zscore.png"
FIGURE_DPI = 600
BOOTSTRAP_REPLICATES = 1000
RANDOM_SEED = 42

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
DATASET_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
RANKING_METHODS = {
    "minimizedAffinity": {
        "title": "smina docking score",
        "ascending": True,
    },
    "CNNaffinity": {
        "title": "GNINA predicted affinity",
        "ascending": False,
    },
    "CNN_VS": {
        "title": "GNINA combined score",
        "ascending": False,
    },
}
ZSCORE_METHOD = "CNNaffinity_zscore"
ZSCORE_METHOD_LABEL = "GNINA - Pocket-normalized affinity"


def save_figure(fig: plt.Figure, output_file: Path) -> None:
    """Save a high-resolution PNG and a resolution-independent SVG copy."""
    fig.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")


def read_binders(path: Path) -> set[str]:
    with path.open() as file:
        return {
            binder_id
            for line in file
            if (binder_id := line.strip().upper()) and binder_id != "NAN"
        }


def pocket_id_from_filename(value: object) -> str:
    """Remove the final compound token from a docking pose filename."""
    parts = Path(str(value)).stem.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else parts[0]


def load_pocket_zscore_statistics(path: Path = POCKET_REFERENCE_FILE) -> pd.DataFrame:
    """Calculate per-pocket CNNaffinity mean and sample SD across reference drugs."""
    if not path.is_file():
        raise FileNotFoundError(path)
    reference = pd.read_csv(path, usecols=["CNNaffinity", "File_Name"])
    reference["CNNaffinity"] = pd.to_numeric(reference["CNNaffinity"], errors="coerce")
    reference["pocket_id"] = reference["File_Name"].map(pocket_id_from_filename)
    statistics = (
        reference.dropna(subset=["CNNaffinity", "pocket_id"])
        .groupby("pocket_id")["CNNaffinity"]
        .agg(pocket_mean="mean", pocket_sd="std", reference_count="count")
    )
    statistics = statistics[statistics["pocket_sd"].notna() & statistics["pocket_sd"].gt(0)]
    if statistics.empty:
        raise ValueError(f"No usable pocket CNNaffinity statistics in {path}")
    return statistics


def read_dock_scores(
    path: Path,
    extension: str,
    drug: str,
    dataset: int,
    ranking_column: str,
    ascending: bool,
    pocket_statistics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Read and rank a docking file, keeping the best row per UniProt ID."""
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

    required_score_columns = {"CNNscore", "CNNaffinity", "minimizedAffinity"}
    missing_columns = required_score_columns - set(dock_scores.columns)
    if missing_columns:
        raise ValueError(f"missing score columns: {sorted(missing_columns)}")

    dock_scores["UNIPROT_ID"] = dock_scores["UNIPROT_ID"].astype("string").str.strip().str.upper()
    for column in required_score_columns:
        dock_scores[column] = pd.to_numeric(dock_scores[column], errors="coerce")

    dock_scores["CNN_VS"] = dock_scores["CNNscore"] * dock_scores["CNNaffinity"]
    dock_scores = dock_scores[dock_scores["minimizedAffinity"] < 0]

    if ranking_column == ZSCORE_METHOD:
        if pocket_statistics is None:
            raise ValueError("pocket statistics are required for z-score ranking")
        if "File_Name" not in dock_scores.columns:
            raise ValueError("File_Name is required to identify pockets for z-score ranking")
        dock_scores["pocket_id"] = dock_scores["File_Name"].map(pocket_id_from_filename)
        dock_scores = dock_scores.join(pocket_statistics, on="pocket_id")
        dock_scores[ZSCORE_METHOD] = (
            dock_scores["CNNaffinity"] - dock_scores["pocket_mean"]
        ) / dock_scores["pocket_sd"]

    return (
        dock_scores.dropna(subset=["UNIPROT_ID", ranking_column])
        .sort_values(ranking_column, ascending=ascending)
        .drop_duplicates(subset="UNIPROT_ID")
        .reset_index(drop=True)
    )


def sklearn_scores(dock_scores: pd.DataFrame, ranking_column: str) -> np.ndarray:
    """Convert each ranking criterion to a score where larger is always better."""
    scores = dock_scores[ranking_column].to_numpy(dtype=float)
    if ranking_column == "minimizedAffinity":
        return -scores
    return scores


def collect_predictions() -> tuple[
    dict[str, dict[int, list[tuple[str, np.ndarray, np.ndarray]]]],
    list[tuple[str, np.ndarray, np.ndarray]],
]:
    """Collect labels and scores for every valid drug, dataset, and ranking method."""
    file_locations = pd.read_csv(FILE_LOCATIONS_FILE)
    predictions = {method: {dataset: [] for dataset in DATASETS} for method in RANKING_METHODS}
    zscore_predictions = []
    pocket_statistics = load_pocket_zscore_statistics()

    for drug in DRUGS:
        binder_file = BINDER_DIR / f"{drug}_filtered_total.txt"
        if not binder_file.is_file():
            print(f"Skipping {drug}: missing binder file {binder_file}")
            continue

        binders = read_binders(binder_file)
        if not binders:
            print(f"Skipping {drug}: empty binder file {binder_file}")
            continue

        for dataset in DATASETS:
            locations = file_locations[
                (file_locations["Compound"] == drug) & (file_locations["Dataset"] == dataset)
            ]
            if locations.empty:
                print(f"Skipping {drug}, dataset {dataset}: no file-location entry")
                continue

            location = locations.iloc[0]
            dock_file = INTERIM_DIR / str(location["File_location"])
            if not dock_file.is_file():
                print(f"Skipping {drug}, dataset {dataset}: missing {dock_file}")
                continue

            extension = str(location.get("Extension", dock_file.suffix.lstrip(".")))
            for method, config in RANKING_METHODS.items():
                try:
                    dock_scores = read_dock_scores(
                        dock_file,
                        extension,
                        drug,
                        dataset,
                        method,
                        config["ascending"],
                    )
                except (OSError, ValueError, pd.errors.ParserError) as error:
                    print(f"Skipping {drug}, dataset {dataset}, {method}: {error}")
                    continue

                y_true = dock_scores["UNIPROT_ID"].isin(binders).astype(int).to_numpy()
                if np.unique(y_true).size < 2:
                    print(
                        f"Skipping {drug}, dataset {dataset}, {method}: "
                        "labels contain only one class"
                    )
                    continue

                predictions[method][dataset].append(
                    (drug, y_true, sklearn_scores(dock_scores, method))
                )

            if dataset == 3:
                try:
                    zscore_scores = read_dock_scores(
                        dock_file,
                        extension,
                        drug,
                        dataset,
                        ZSCORE_METHOD,
                        ascending=False,
                        pocket_statistics=pocket_statistics,
                    )
                except (OSError, ValueError, pd.errors.ParserError) as error:
                    print(f"Skipping {drug}, dataset 3, {ZSCORE_METHOD}: {error}")
                    continue
                y_true = zscore_scores["UNIPROT_ID"].isin(binders).astype(int).to_numpy()
                if np.unique(y_true).size < 2:
                    print(
                        f"Skipping {drug}, dataset 3, {ZSCORE_METHOD}: "
                        "labels contain only one class"
                    )
                    continue
                zscore_predictions.append(
                    (
                        drug,
                        y_true,
                        sklearn_scores(zscore_scores, ZSCORE_METHOD),
                    )
                )

    return predictions, zscore_predictions


def macro_average_roc(
    drug_predictions: list[tuple[str, np.ndarray, np.ndarray]],
    fpr_grid: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Interpolate per-drug ROC curves and return their macro average."""
    interpolated_tprs = []
    drug_aucs = []

    for _, y_true, y_score in drug_predictions:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        interpolated_tpr = np.interp(fpr_grid, fpr, tpr)
        interpolated_tpr[0] = 0
        interpolated_tprs.append(interpolated_tpr)
        drug_aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_tpr[-1] = 1
    mean_auc = float(np.mean(drug_aucs))
    if len(drug_aucs) > 1:
        standard_error = float(np.std(drug_aucs, ddof=1) / np.sqrt(len(drug_aucs)))
        margin = float(t.ppf(0.975, df=len(drug_aucs) - 1) * standard_error)
        ci_low = mean_auc - margin
        ci_high = mean_auc + margin
    else:
        ci_low = np.nan
        ci_high = np.nan
    return mean_tpr, mean_auc, ci_low, ci_high


def micro_average_pr(
    drug_predictions: list[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Pool all drug-target predictions and calculate a micro-average PR curve."""
    y_true = np.concatenate([prediction[1] for prediction in drug_predictions])
    y_score = np.concatenate([prediction[2] for prediction in drug_predictions])
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    baseline = float(y_true.mean())
    return recall, precision, float(pr_auc), baseline


def bootstrap_micro_auprc(
    drug_predictions: list[tuple[str, np.ndarray, np.ndarray]],
    random_seed: int,
) -> tuple[float, float]:
    """Bootstrap drugs with replacement and return a percentile 95% AUPRC CI."""
    rng = np.random.default_rng(random_seed)
    bootstrap_auprcs = []
    number_of_drugs = len(drug_predictions)

    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_indices = rng.integers(0, number_of_drugs, size=number_of_drugs)
        y_true = np.concatenate([drug_predictions[index][1] for index in sampled_indices])
        y_score = np.concatenate([drug_predictions[index][2] for index in sampled_indices])
        if np.unique(y_true).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        bootstrap_auprcs.append(auc(recall, precision))

    if not bootstrap_auprcs:
        return np.nan, np.nan
    return tuple(np.percentile(bootstrap_auprcs, [2.5, 97.5]))


def draw_macro_roc(
    predictions: dict[str, dict[int, list[tuple[str, np.ndarray, np.ndarray]]]],
) -> list[dict]:
    fpr_grid = np.linspace(0, 1, 1001)
    fig, axes = plt.subplots(
        1,
        len(RANKING_METHODS),
        figsize=(7 * len(RANKING_METHODS), 6.5),
        sharex=True,
        sharey=True,
    )
    metric_rows = []

    for ax, (method, config) in zip(axes, RANKING_METHODS.items()):
        for dataset in DATASETS:
            drug_predictions = predictions[method][dataset]
            if not drug_predictions:
                continue
            mean_tpr, mean_auc, ci_low, ci_high = macro_average_roc(drug_predictions, fpr_grid)
            ax.plot(
                fpr_grid,
                mean_tpr,
                color=DATASET_COLORS[dataset],
                linewidth=2.5,
                label=(
                    f"Dataset {dataset} (AUROC={mean_auc:.3f}, 95% CI {ci_low:.3f}–{ci_high:.3f})"
                ),
            )
            metric_rows.append(
                {
                    "ranking_method": method,
                    "dataset": dataset,
                    "drug_count": len(drug_predictions),
                    "macro_auroc": mean_auc,
                    "macro_auroc_ci_95_low": ci_low,
                    "macro_auroc_ci_95_high": ci_high,
                }
            )

        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5)
        ax.set_title(config["title"], fontsize=18)
        ax.set_xlabel("False positive rate", fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=12)

    axes[0].set_ylabel("True positive rate", fontsize=16)
    fig.suptitle("Macro-averaged ROC curves", fontsize=22)
    fig.tight_layout()
    save_figure(fig, ROC_OUTPUT_FILE)
    plt.close(fig)
    return metric_rows


def draw_micro_pr(
    predictions: dict[str, dict[int, list[tuple[str, np.ndarray, np.ndarray]]]],
) -> list[dict]:
    fig, axes = plt.subplots(
        1,
        len(RANKING_METHODS),
        figsize=(7 * len(RANKING_METHODS), 6.5),
        sharex=True,
        sharey=True,
    )
    metric_rows = []

    for method_index, (ax, (method, config)) in enumerate(zip(axes, RANKING_METHODS.items())):
        curves = {}
        for dataset in DATASETS:
            drug_predictions = predictions[method][dataset]
            if not drug_predictions:
                continue
            recall, precision, pr_auc, baseline = micro_average_pr(drug_predictions)
            ci_low, ci_high = bootstrap_micro_auprc(
                drug_predictions,
                random_seed=RANDOM_SEED + method_index * 10 + dataset,
            )
            curves[dataset] = (recall, precision)
            ax.plot(
                recall,
                precision,
                color=DATASET_COLORS[dataset],
                linewidth=2.5,
                label=(
                    f"Dataset {dataset} (AUPRC={pr_auc:.3f}, "
                    f"95% CI {ci_low:.3f}–{ci_high:.3f}, prev={baseline:.3f})"
                ),
            )
            ax.axhline(
                baseline,
                color=DATASET_COLORS[dataset],
                linestyle=":",
                linewidth=1,
                alpha=0.7,
            )
            pooled_y_true = np.concatenate([prediction[1] for prediction in drug_predictions])
            metric_rows.append(
                {
                    "ranking_method": method,
                    "dataset": dataset,
                    "micro_auprc": pr_auc,
                    "micro_auprc_bootstrap_ci_95_low": ci_low,
                    "micro_auprc_bootstrap_ci_95_high": ci_high,
                    "micro_positive_prevalence": baseline,
                    "pooled_prediction_count": len(pooled_y_true),
                    "pooled_positive_count": int(pooled_y_true.sum()),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )

        ax.set_title(config["title"], fontsize=18)
        ax.set_xlabel("Recall", fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)

        zoom_ax = inset_axes(ax, width="48%", height="48%", loc="center right", borderpad=1.3)
        for dataset, (recall, precision) in curves.items():
            zoom_ax.plot(
                recall,
                precision,
                color=DATASET_COLORS[dataset],
                linewidth=1.8,
            )
        zoom_ax.set_xlim(0, 0.2)
        zoom_ax.set_ylim(0, 0.2)
        zoom_ax.set_xticks([0, 0.1, 0.2])
        zoom_ax.set_yticks([0, 0.1, 0.2])
        zoom_ax.tick_params(axis="both", labelsize=9)
        zoom_ax.grid(alpha=0.25)
        mark_inset(ax, zoom_ax, loc1=2, loc2=4, fc="none", ec="gray", linewidth=0.8)

    axes[0].set_ylabel("Precision", fontsize=16)
    fig.suptitle("Micro-averaged precision-recall curves", fontsize=22, y=0.97)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.82, wspace=0.08)
    save_figure(fig, PR_OUTPUT_FILE)
    plt.close(fig)
    return metric_rows


def draw_drug_auc_heatmap(
    predictions: (dict[str, dict[int, list[tuple[str, np.ndarray, np.ndarray]]]] | None) = None,
) -> pd.DataFrame:
    """Draw per-drug AUROCs for each ranking method and dataset."""
    binder_counts = {
        drug: len(read_binders(BINDER_DIR / f"{drug}_filtered_total.txt"))
        for drug in DRUGS
        if (BINDER_DIR / f"{drug}_filtered_total.txt").is_file()
    }
    if predictions is None:
        if not DRUG_AUC_OUTPUT_FILE.is_file():
            raise FileNotFoundError(DRUG_AUC_OUTPUT_FILE)
        auc_table = pd.read_csv(DRUG_AUC_OUTPUT_FILE)
    else:
        auc_rows = []
        for method in RANKING_METHODS:
            for dataset in DATASETS:
                for drug, y_true, y_score in predictions[method][dataset]:
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    auc_rows.append(
                        {
                            "drug": drug,
                            "dataset": dataset,
                            "ranking_method": method,
                            "auroc": auc(fpr, tpr),
                            "binder_count": binder_counts.get(drug, np.nan),
                        }
                    )
        auc_table = pd.DataFrame(auc_rows)
    method_labels = {method: config["title"] for method, config in RANKING_METHODS.items()}
    fig = plt.figure(figsize=(20, 12))
    grid = fig.add_gridspec(
        1,
        len(DATASETS) + 1,
        width_ratios=[1] * len(DATASETS) + [0.06],
        wspace=0.16,
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
        heatmap_data = (
            auc_table[auc_table["dataset"] == dataset]
            .pivot(index="drug", columns="ranking_method", values="auroc")
            .reindex(index=DRUGS, columns=RANKING_METHODS)
            .rename(columns=method_labels)
        )
        heatmap_data.index = [
            f"{drug} (n={binder_counts[drug]})" if drug in binder_counts else f"{drug} (n=NA)"
            for drug in heatmap_data.index
        ]
        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap="vlag",
            vmin=0,
            vmax=1,
            center=0.5,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 13, "fontweight": "bold"},
            linewidths=1.5,
            linecolor="white",
            square=False,
            cbar=index == len(DATASETS) - 1,
            cbar_ax=colorbar_axis if index == len(DATASETS) - 1 else None,
            cbar_kws={"label": "AUROC"} if index == len(DATASETS) - 1 else None,
        )
        ax.set_title(f"Dataset {dataset}", fontsize=20, fontweight="bold", pad=12)
        ax.set_xlabel("Sorting method", fontsize=16)
        ax.set_ylabel("Drug" if index == 0 else "", fontsize=16)
        ax.tick_params(axis="x", labelsize=12)
        plt.setp(
            ax.get_xticklabels(),
            rotation=30,
            ha="right",
            rotation_mode="anchor",
        )
        ax.tick_params(axis="y", labelrotation=0, labelsize=12)
        if index > 0:
            ax.tick_params(axis="y", labelleft=False, left=False)

    colorbar_axis.tick_params(labelsize=12)
    colorbar_axis.set_ylabel("AUROC", fontsize=15, labelpad=12)
    fig.suptitle(
        "Per-drug AUROC by sorting method",
        fontsize=25,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.14, right=0.95, bottom=0.14, top=0.90)
    save_figure(fig, HEATMAP_OUTPUT_FILE)
    plt.close(fig)
    return auc_table


def draw_zscore_roc_and_pr(
    zscore_predictions: list[tuple[str, np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Draw Dataset 3 ROC and PR figures for pocket-normalized CNNaffinity."""
    if not zscore_predictions:
        raise ValueError("No Dataset 3 z-score predictions are available")

    fpr_grid = np.linspace(0, 1, 1001)
    mean_tpr, mean_auroc, _, _ = macro_average_roc(zscore_predictions, fpr_grid)
    roc_fig, roc_ax = plt.subplots(figsize=(9, 7.5))
    roc_ax.plot(
        fpr_grid,
        mean_tpr,
        color="tab:purple",
        linewidth=3,
        label=f"Dataset 3 (average AUROC={mean_auroc:.3f})",
    )
    roc_ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5)
    roc_ax.set_title(
        "Pocket-normalized CNNaffinity: average ROC",
        fontsize=22,
        fontweight="bold",
        pad=16,
    )
    roc_ax.set_xlabel("False positive rate", fontsize=18)
    roc_ax.set_ylabel("True positive rate", fontsize=18)
    roc_ax.tick_params(axis="both", labelsize=15)
    roc_ax.grid(alpha=0.3)
    roc_ax.legend(loc="lower right", fontsize=14)
    roc_fig.tight_layout()
    save_figure(roc_fig, ZSCORE_ROC_OUTPUT_FILE)
    plt.close(roc_fig)

    recall, precision, auprc, baseline = micro_average_pr(zscore_predictions)
    pr_fig, pr_ax = plt.subplots(figsize=(9, 7.5))
    pr_ax.plot(
        recall,
        precision,
        color="tab:purple",
        linewidth=3,
        label=f"Dataset 3 (AUPRC={auprc:.3f})",
    )
    pr_ax.axhline(
        baseline,
        color="gray",
        linestyle=":",
        linewidth=1.8,
        label=f"Prevalence={baseline:.3f}",
    )
    pr_ax.set_title(
        "Pocket-normalized CNNaffinity: precision–recall",
        fontsize=22,
        fontweight="bold",
        pad=16,
    )
    pr_ax.set_xlabel("Recall", fontsize=18)
    pr_ax.set_ylabel("Precision", fontsize=18)
    pr_ax.tick_params(axis="both", labelsize=15)
    pr_ax.grid(alpha=0.3)
    pr_ax.legend(loc="upper right", fontsize=14)
    pr_fig.tight_layout()
    save_figure(pr_fig, ZSCORE_PR_OUTPUT_FILE)
    plt.close(pr_fig)

    return {
        "dataset": 3,
        "ranking_method": ZSCORE_METHOD,
        "drug_count": len(zscore_predictions),
        "macro_auroc": mean_auroc,
        "micro_auprc": auprc,
        "micro_positive_prevalence": baseline,
        "pooled_prediction_count": sum(len(y_true) for _, y_true, _ in zscore_predictions),
        "pooled_positive_count": sum(int(y_true.sum()) for _, y_true, _ in zscore_predictions),
    }


def per_drug_auroc_rows(
    predictions: list[tuple[str, np.ndarray, np.ndarray]],
    method: str,
) -> list[dict]:
    rows = []
    for drug, y_true, y_score in predictions:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        rows.append(
            {
                "drug": drug,
                "dataset": 3,
                "ranking_method": method,
                "auroc": auc(fpr, tpr),
            }
        )
    return rows


def draw_dataset3_heatmap_with_zscore(
    predictions: dict[str, dict[int, list[tuple[str, np.ndarray, np.ndarray]]]],
    zscore_predictions: list[tuple[str, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Draw Dataset 3 per-drug AUROCs for the original and z-score methods."""
    rows = []
    for method in RANKING_METHODS:
        rows.extend(per_drug_auroc_rows(predictions[method][3], method))
    rows.extend(per_drug_auroc_rows(zscore_predictions, ZSCORE_METHOD))
    auc_table = pd.DataFrame(rows)

    method_order = [*RANKING_METHODS, ZSCORE_METHOD]
    method_labels = {method: config["title"] for method, config in RANKING_METHODS.items()}
    method_labels[ZSCORE_METHOD] = ZSCORE_METHOD_LABEL
    heatmap_data = (
        auc_table.pivot(index="drug", columns="ranking_method", values="auroc")
        .reindex(index=DRUGS, columns=method_order)
        .rename(columns=method_labels)
    )
    binder_counts = {
        drug: len(read_binders(BINDER_DIR / f"{drug}_filtered_total.txt"))
        for drug in DRUGS
        if (BINDER_DIR / f"{drug}_filtered_total.txt").is_file()
    }
    heatmap_data.index = [
        f"{drug} (n={binder_counts[drug]})" if drug in binder_counts else f"{drug} (n=NA)"
        for drug in heatmap_data.index
    ]

    fig, ax = plt.subplots(figsize=(16, 11))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="vlag",
        vmin=0,
        vmax=1,
        center=0.5,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 12},
        linewidths=0.6,
        linecolor="white",
        square=True,
        mask=heatmap_data.isna(),
        cbar_kws={"label": "AUROC"},
    )
    ax.set_title(
        "Dataset 3 per-drug AUROC with pocket normalization",
        fontsize=24,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("Sorting method", fontsize=17)
    ax.set_ylabel("Drug", fontsize=17)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13, rotation=0)
    plt.setp(
        ax.get_xticklabels(),
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.24, top=0.90)
    save_figure(fig, ZSCORE_HEATMAP_OUTPUT_FILE)
    plt.close(fig)
    return auc_table


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    predictions, zscore_predictions = collect_predictions()
    roc_metrics = pd.DataFrame(draw_macro_roc(predictions))
    pr_metrics = pd.DataFrame(draw_micro_pr(predictions))
    drug_auc_table = draw_drug_auc_heatmap(predictions)
    zscore_metrics = draw_zscore_roc_and_pr(zscore_predictions)
    zscore_auc_table = draw_dataset3_heatmap_with_zscore(predictions, zscore_predictions)
    metrics = roc_metrics.merge(pr_metrics, on=["ranking_method", "dataset"], how="outer")
    metrics.to_csv(METRICS_OUTPUT_FILE, index=False)
    drug_auc_table.to_csv(DRUG_AUC_OUTPUT_FILE, index=False)
    pd.DataFrame([zscore_metrics]).to_csv(ZSCORE_METRICS_OUTPUT_FILE, index=False)
    zscore_auc_table[zscore_auc_table["ranking_method"] == ZSCORE_METHOD].to_csv(
        ZSCORE_DRUG_AUC_OUTPUT_FILE, index=False
    )
    print(f"ROC figure saved to: {ROC_OUTPUT_FILE}")
    print(f"ROC SVG saved to: {ROC_OUTPUT_FILE.with_suffix('.svg')}")
    print(f"PR figure saved to: {PR_OUTPUT_FILE}")
    print(f"PR SVG saved to: {PR_OUTPUT_FILE.with_suffix('.svg')}")
    print(f"AUROC heatmap saved to: {HEATMAP_OUTPUT_FILE}")
    print(f"AUROC heatmap SVG saved to: {HEATMAP_OUTPUT_FILE.with_suffix('.svg')}")
    print(f"Metrics saved to: {METRICS_OUTPUT_FILE}")
    print(f"Per-drug AUROCs saved to: {DRUG_AUC_OUTPUT_FILE}")
    print(f"Z-score ROC figure saved to: {ZSCORE_ROC_OUTPUT_FILE}")
    print(f"Z-score PR figure saved to: {ZSCORE_PR_OUTPUT_FILE}")
    print(f"Z-score metrics saved to: {ZSCORE_METRICS_OUTPUT_FILE}")
    print(f"Z-score per-drug AUROCs saved to: {ZSCORE_DRUG_AUC_OUTPUT_FILE}")
    print(f"Dataset 3 expanded heatmap saved to: {ZSCORE_HEATMAP_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
