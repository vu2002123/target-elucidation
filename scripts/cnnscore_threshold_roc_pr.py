#!/usr/bin/env python3

"""Evaluate CNNscore-filtered protein rankings across docking datasets."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

FILE_LOCATIONS_FILE = INTERIM_DIR / "file_locations.csv"
BINDER_DIR = RAW_DIR / "pubchem"
METRICS_OUTPUT_FILE = INTERIM_DIR / "cnnscore_threshold_total_metrics.csv"
DRUG_AUROC_OUTPUT_FILE = INTERIM_DIR / "cnnscore_threshold_drug_auroc.csv"
HEATMAP_OUTPUT_FILE = FIGURE_DIR / "cnnscore_threshold_drug_auroc_heatmap.png"

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
]
DATASETS = (1, 2, 3)
THRESHOLDS = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))
FIGURE_DPI = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate pooled AUROC/AUPRC after filtering pockets by CNNscore "
            "and collapsing each protein to one CNNaffinity score."
        )
    )
    parser.add_argument(
        "--file-locations",
        type=Path,
        default=FILE_LOCATIONS_FILE,
        help=f"Docking-file registry (default: {FILE_LOCATIONS_FILE}).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=FIGURE_DIR,
        help=f"Figure output directory (default: {FIGURE_DIR}).",
    )
    parser.add_argument(
        "--output-prefix",
        default="cnnscore_threshold",
        help="Prefix for output CSV and heatmap filenames.",
    )
    return parser.parse_args()


def read_binders(path: Path) -> set[str]:
    with path.open() as handle:
        return {
            binder_id
            for line in handle
            if (binder_id := line.strip().upper()) and binder_id != "NAN"
        }


def read_pocket_scores(path: Path, extension: str, drug: str, dataset: int) -> pd.DataFrame:
    """Read all valid pockets without collapsing duplicate protein IDs."""
    separator = "\t" if extension.lower() == "tsv" or path.suffix.lower() == ".tsv" else ","
    scores = pd.read_csv(path, sep=separator)
    scores = scores.rename(
        columns={
            "CNN_affinity": "CNNaffinity",
            "CNN_pose_score": "CNNscore",
            "CNN_score": "CNNscore",
            "affinity": "minimizedAffinity",
            "ID": "UNIPROT_ID",
        }
    )
    if "Compound" in scores.columns:
        scores = scores[scores["Compound"] == drug].copy()

    if "UNIPROT_ID" not in scores.columns:
        if "File_Name" not in scores.columns:
            raise ValueError("no UNIPROT_ID, ID, or File_Name column")
        if dataset == 1:
            scores["UNIPROT_ID"] = scores["File_Name"].str.split("-").str[1]
        else:
            scores["UNIPROT_ID"] = scores["File_Name"].str.split("_").str[0]

    required = {"CNNscore", "CNNaffinity"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"missing score columns: {sorted(missing)}")
    scores["UNIPROT_ID"] = scores["UNIPROT_ID"].astype("string").str.strip().str.upper()
    for column in required:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    if "minimizedAffinity" in scores.columns:
        scores["minimizedAffinity"] = pd.to_numeric(
            scores["minimizedAffinity"], errors="coerce"
        )
        scores = scores[scores["minimizedAffinity"] < 0]
    return scores.dropna(subset=["UNIPROT_ID", "CNNscore", "CNNaffinity"])


def collapse_protein_scores(pockets: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Filter pockets, apply the fallback, and retain one score per protein."""
    passing = (
        pockets[pockets["CNNscore"] >= threshold]
        .sort_values("CNNaffinity", ascending=False)
        .drop_duplicates("UNIPROT_ID", keep="first")
        .copy()
    )
    proteins_with_passing_pocket = set(passing["UNIPROT_ID"])
    fallback = (
        pockets[~pockets["UNIPROT_ID"].isin(proteins_with_passing_pocket)]
        .sort_values(["CNNscore", "CNNaffinity"], ascending=[False, False])
        .drop_duplicates("UNIPROT_ID", keep="first")
        .copy()
    )
    fallback["CNNaffinity"] = fallback["CNNaffinity"] * fallback["CNNscore"]
    fallback["used_cnnscore_fallback"] = True
    passing["used_cnnscore_fallback"] = False
    return (
        pd.concat([passing, fallback], ignore_index=True)
        .sort_values("CNNaffinity", ascending=False)
        .reset_index(drop=True)
    )


def load_all_inputs(file_locations_path: Path) -> dict[tuple[str, int], tuple[pd.DataFrame, set[str]]]:
    if not file_locations_path.is_file():
        raise FileNotFoundError(file_locations_path)
    file_locations = pd.read_csv(file_locations_path)
    loaded = {}
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
                (file_locations["Compound"] == drug)
                & (file_locations["Dataset"] == dataset)
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
            try:
                pockets = read_pocket_scores(dock_file, extension, drug, dataset)
            except (OSError, ValueError, pd.errors.ParserError) as error:
                print(f"Skipping {drug}, dataset {dataset}: {error}")
                continue
            if pockets.empty:
                print(f"Skipping {drug}, dataset {dataset}: no valid pocket scores")
                continue
            loaded[(drug, dataset)] = (pockets, binders)
    return loaded


def calculate_metrics(
    loaded: dict[tuple[str, int], tuple[pd.DataFrame, set[str]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_rows = []
    drug_rows = []
    for dataset in DATASETS:
        for threshold in THRESHOLDS:
            pooled_labels = []
            pooled_scores = []
            valid_drug_count = 0
            for drug in DRUGS:
                data = loaded.get((drug, dataset))
                if data is None:
                    continue
                pockets, binders = data
                proteins = collapse_protein_scores(pockets, threshold)
                y_true = proteins["UNIPROT_ID"].isin(binders).astype(int).to_numpy()
                y_score = proteins["CNNaffinity"].to_numpy(dtype=float)
                if np.unique(y_true).size < 2:
                    print(
                        f"Skipping metrics for {drug}, dataset {dataset}, "
                        f"threshold {threshold:.1f}: labels contain one class"
                    )
                    continue
                drug_precision, drug_recall, _ = precision_recall_curve(
                    y_true, y_score
                )
                drug_rows.append(
                    {
                        "drug": drug,
                        "dataset": dataset,
                        "cnnscore_threshold": threshold,
                        "auroc": roc_auc_score(y_true, y_score),
                        "auprc": auc(drug_recall, drug_precision),
                        "protein_count": len(y_true),
                        "positive_count": int(y_true.sum()),
                        "fallback_protein_count": int(
                            proteins["used_cnnscore_fallback"].sum()
                        ),
                    }
                )
                pooled_labels.append(y_true)
                pooled_scores.append(y_score)
                valid_drug_count += 1

            if not pooled_labels:
                continue
            y_true_total = np.concatenate(pooled_labels)
            y_score_total = np.concatenate(pooled_scores)
            precision, recall, _ = precision_recall_curve(y_true_total, y_score_total)
            total_rows.append(
                {
                    "dataset": dataset,
                    "cnnscore_threshold": threshold,
                    "drug_count": valid_drug_count,
                    "prediction_count": len(y_true_total),
                    "positive_count": int(y_true_total.sum()),
                    "positive_prevalence": float(y_true_total.mean()),
                    "total_auroc": roc_auc_score(y_true_total, y_score_total),
                    "total_auprc": auc(recall, precision),
                }
            )
    return pd.DataFrame(total_rows), pd.DataFrame(drug_rows)


def draw_drug_metric_heatmap(
    drug_metrics: pd.DataFrame,
    metric: str,
    output_file: Path,
) -> None:
    if drug_metrics.empty:
        raise ValueError(f"No per-drug {metric.upper()} values are available for the heatmap")
    figure, axes = plt.subplots(1, len(DATASETS), figsize=(25, 13), sharey=True)
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        matrix = (
            drug_metrics[drug_metrics["dataset"] == dataset]
            .pivot(index="drug", columns="cnnscore_threshold", values=metric)
            .reindex(index=DRUGS, columns=THRESHOLDS)
        )
        sns.heatmap(
            matrix,
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
            mask=matrix.isna(),
            cbar=index == len(DATASETS) - 1,
            cbar_kws={"label": metric.upper()} if index == len(DATASETS) - 1 else None,
        )
        ax.set_title(f"Dataset {dataset}", fontsize=22, fontweight="bold", pad=16)
        ax.set_xlabel("CNNscore threshold", fontsize=18, labelpad=10)
        ax.set_ylabel("Drug" if index == 0 else "", fontsize=18)
        ax.tick_params(axis="x", labelsize=14, rotation=0)
        ax.tick_params(axis="y", labelsize=14, rotation=0)
    figure.suptitle(
        f"Per-drug {metric.upper()} after CNNscore pocket filtering",
        fontsize=28,
        y=0.98,
    )
    figure.subplots_adjust(left=0.14, right=0.94, bottom=0.09, top=0.90, wspace=0.08)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    figure.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    loaded = load_all_inputs(args.file_locations)
    if not loaded:
        raise ValueError("No valid docking inputs were loaded")
    total_metrics, drug_metrics = calculate_metrics(loaded)

    total_output = INTERIM_DIR / f"{args.output_prefix}_total_metrics.csv"
    drug_output = INTERIM_DIR / f"{args.output_prefix}_drug_auroc.csv"
    auroc_heatmap_output = (
        args.figure_dir / f"{args.output_prefix}_drug_auroc_heatmap.png"
    )
    auprc_heatmap_output = (
        args.figure_dir / f"{args.output_prefix}_drug_auprc_heatmap.png"
    )
    total_metrics.to_csv(total_output, index=False)
    drug_metrics.to_csv(drug_output, index=False)
    draw_drug_metric_heatmap(drug_metrics, "auroc", auroc_heatmap_output)
    draw_drug_metric_heatmap(drug_metrics, "auprc", auprc_heatmap_output)
    print(f"Total AUROC/AUPRC metrics saved to: {total_output}")
    print(f"Per-drug AUROC/AUPRC values saved to: {drug_output}")
    print(f"AUROC heatmap saved to: {auroc_heatmap_output}")
    print(f"AUROC heatmap SVG saved to: {auroc_heatmap_output.with_suffix('.svg')}")
    print(f"AUPRC heatmap saved to: {auprc_heatmap_output}")
    print(f"AUPRC heatmap SVG saved to: {auprc_heatmap_output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
