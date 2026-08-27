#!/usr/bin/env python3

"""Plot cardiotoxicity-target ranks for positive and negative control drugs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_POSITIVE_DOCKING_FILES = [
    DATA_DIR / "interim" / "debby_cp_out.csv",
    DATA_DIR / "interim" / "debby_cp_2_out.csv",
]
DEFAULT_NEGATIVE_DOCKING_FILES = [DATA_DIR / "interim" / "debby_cp_n_out.csv"]
DEFAULT_TARGET_FILE = DATA_DIR / "raw" / "cardiotoxicity_targets.txt"
DEFAULT_UNIPROT_FILE = DATA_DIR / "raw" / "uniprot_ids.tsv"
DEFAULT_OUTPUT_DATA = DATA_DIR / "interim" / "cardiotoxicity_target_ranks.csv"
DEFAULT_OUTPUT_FIGURE = (
    PROJECT_DIR / "reports" / "figures" / "cardiotoxicity_target_ranks.png"
)
DEFAULT_AUROC_FIGURE = (
    PROJECT_DIR / "reports" / "figures" / "cardiotoxicity_KCNH2_auroc.png"
)
KCNH2_UNIPROT_ID = "Q12809"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-docking-files",
        type=Path,
        nargs="+",
        default=DEFAULT_POSITIVE_DOCKING_FILES,
        help="Docking CSV files containing cardiotoxicity-positive compounds",
    )
    parser.add_argument(
        "--negative-docking-files",
        type=Path,
        nargs="+",
        default=DEFAULT_NEGATIVE_DOCKING_FILES,
        help="Docking CSV files containing cardiotoxicity-negative compounds",
    )
    parser.add_argument("--target-file", type=Path, default=DEFAULT_TARGET_FILE)
    parser.add_argument("--uniprot-file", type=Path, default=DEFAULT_UNIPROT_FILE)
    parser.add_argument("--output-data", type=Path, default=DEFAULT_OUTPUT_DATA)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT_FIGURE)
    parser.add_argument(
        "--auroc-output",
        type=Path,
        default=DEFAULT_AUROC_FIGURE,
        help="Output path for the KCNH2 rank-percentile ROC curve",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_targets(path: Path) -> list[str]:
    with path.open() as handle:
        targets = list(
            dict.fromkeys(
                line.strip().upper()
                for line in handle
                if line.strip() and line.strip().upper() != "NAN"
            )
        )
    if not targets:
        raise ValueError(f"No target IDs found in {path}")
    return targets


def load_gene_names(path: Path) -> dict[str, str]:
    uniprot = pd.read_csv(path, sep="\t", usecols=["Entry", "Gene Names"])
    uniprot["Entry"] = uniprot["Entry"].astype("string").str.strip().str.upper()
    uniprot["gene"] = (
        uniprot["Gene Names"].astype("string").str.strip().str.split().str[0]
    )
    return (
        uniprot.dropna(subset=["Entry", "gene"])
        .drop_duplicates("Entry")
        .set_index("Entry")["gene"]
        .to_dict()
    )


def calculate_ranks(
    docking_files: list[Path],
    targets: list[str],
    gene_names: dict[str, str],
    compound_class: str,
) -> pd.DataFrame:
    docking_tables = []
    required = {"Compound", "CNN_VS", "File_Name"}
    for docking_file in docking_files:
        docking = pd.read_csv(docking_file)
        missing = required - set(docking.columns)
        if missing:
            raise ValueError(
                f"{docking_file} is missing docking columns: {sorted(missing)}"
            )
        docking["Source_file"] = docking_file.name
        docking_tables.append(docking)
    docking = pd.concat(docking_tables, ignore_index=True)

    docking["Compound"] = docking["Compound"].astype("string").str.strip()
    docking["UNIPROT_ID"] = (
        docking["File_Name"]
        .astype("string")
        .str.split("_")
        .str[0]
        .str.strip()
        .str.upper()
    )
    docking["CNN_VS"] = pd.to_numeric(docking["CNN_VS"], errors="coerce")

    rows = []
    for drug, drug_rows in docking.groupby("Compound", sort=False):
        ranked = (
            drug_rows.dropna(subset=["UNIPROT_ID", "CNN_VS"])
            .sort_values("CNN_VS", ascending=False, kind="stable")
            .drop_duplicates("UNIPROT_ID", keep="first")
            .reset_index(drop=True)
        )
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked_lookup = ranked.set_index("UNIPROT_ID")
        for target in targets:
            if target in ranked_lookup.index:
                target_row = ranked_lookup.loc[target]
                rank = int(target_row["rank"])
                score = float(target_row["CNN_VS"])
            else:
                rank = pd.NA
                score = np.nan
            rows.append(
                {
                    "Drug": drug,
                    "Cardiotoxicity_class": compound_class,
                    "UNIPROT_ID": target,
                    "Gene": gene_names.get(target, ""),
                    "Rank": rank,
                    "Rank_percentile": (
                        100 * (len(ranked) - rank + 1) / len(ranked)
                        if pd.notna(rank)
                        else np.nan
                    ),
                    "CNN_VS": score,
                    "Ranked_target_count": len(ranked),
                }
            )
    return pd.DataFrame(rows)


def draw_heatmap(ranks: pd.DataFrame, targets: list[str], output: Path, dpi: int) -> None:
    class_order = ["Positive", "Negative"]
    drugs = []
    class_by_drug = {}
    for compound_class in class_order:
        class_drugs = ranks.loc[
            ranks["Cardiotoxicity_class"].eq(compound_class), "Drug"
        ].drop_duplicates()
        for drug in class_drugs:
            if drug in class_by_drug:
                raise ValueError(
                    f"{drug} occurs in both positive and negative compound groups"
                )
            drugs.append(drug)
            class_by_drug[drug] = compound_class
    matrix = (
        ranks.pivot(index="Drug", columns="UNIPROT_ID", values="Rank")
        .reindex(index=drugs, columns=targets)
        .apply(pd.to_numeric, errors="coerce")
    )
    percentile_matrix = (
        ranks.pivot(
            index="Drug",
            columns="UNIPROT_ID",
            values="Rank_percentile",
        )
        .reindex(index=drugs, columns=targets)
        .apply(pd.to_numeric, errors="coerce")
    )
    genes = (
        ranks.drop_duplicates("UNIPROT_ID")
        .set_index("UNIPROT_ID")["Gene"]
        .to_dict()
    )
    target_labels = [
        f"{genes.get(target, '')}\n{target}".strip() for target in matrix.columns
    ]
    annotations = matrix.copy().astype(object)
    for drug in drugs:
        for target in targets:
            rank = matrix.loc[drug, target]
            percentile = percentile_matrix.loc[drug, target]
            annotations.loc[drug, target] = (
                ""
                if pd.isna(rank)
                else f"{percentile:.1f}%"
            )

    figure_height = max(10, 0.42 * len(drugs) + 3)
    figure, ax = plt.subplots(figsize=(15, figure_height))
    sns.heatmap(
        percentile_matrix,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=100,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 11, "fontweight": "bold"},
        linewidths=1.2,
        linecolor="white",
        mask=percentile_matrix.isna(),
        cbar_kws={"label": "Rank percentile (%; higher is better)"},
    )
    for row_index, column_index in zip(*np.where(matrix.isna().to_numpy())):
        ax.text(
            column_index + 0.5,
            row_index + 0.5,
            "Not found",
            ha="center",
            va="center",
            fontsize=9,
            color="dimgray",
            fontstyle="italic",
        )
    positive_count = sum(class_by_drug[drug] == "Positive" for drug in drugs)
    if 0 < positive_count < len(drugs):
        ax.axhline(positive_count, color="black", linewidth=3)
    ax.set_title(
        "Ranks of cardiotoxicity targets by drug",
        fontsize=22,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Cardiotoxicity target", fontsize=15, labelpad=10)
    ax.set_ylabel("Drug", fontsize=15)
    ax.set_xticklabels(target_labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    for label, drug in zip(ax.get_yticklabels(), drugs):
        if class_by_drug[drug] == "Negative":
            label.set_color("red")
    ax.tick_params(axis="both", labelsize=11)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=11)
    colorbar.set_label("Rank percentile (%; higher is better)", fontsize=13, labelpad=12)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def draw_kcnh2_auroc(ranks: pd.DataFrame, output: Path, dpi: int) -> tuple[float, int]:
    """Draw an ROC curve using KCNH2 rank percentile as the predictor."""
    kcnh2 = ranks[ranks["UNIPROT_ID"].eq(KCNH2_UNIPROT_ID)].copy()
    kcnh2["Rank_percentile"] = pd.to_numeric(
        kcnh2["Rank_percentile"], errors="coerce"
    )
    available = kcnh2.dropna(subset=["Rank_percentile"]).copy()
    labels = available["Cardiotoxicity_class"].eq("Positive").astype(int).to_numpy()
    scores = available["Rank_percentile"].to_numpy(dtype=float)
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "KCNH2 AUROC requires at least one positive and one negative compound"
        )

    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1]]
    true_positive_rates = []
    false_positive_rates = []
    for threshold in thresholds:
        predicted_positive = scores >= threshold
        true_positive_rates.append(
            np.sum(predicted_positive & (labels == 1)) / positive_count
        )
        false_positive_rates.append(
            np.sum(predicted_positive & (labels == 0)) / negative_count
        )
    true_positive_rates = np.asarray(true_positive_rates)
    false_positive_rates = np.asarray(false_positive_rates)
    auroc = float(np.trapezoid(true_positive_rates, false_positive_rates))

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot(
        false_positive_rates,
        true_positive_rates,
        color="tab:blue",
        linewidth=3,
        label=f"KCNH2 rank percentile (AUROC = {auroc:.3f})",
    )
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=2)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("False positive rate", fontsize=15)
    axis.set_ylabel("True positive rate", fontsize=15)
    axis.set_title("Cardiotoxicity classification using KCNH2 rank", fontsize=19, pad=14)
    axis.tick_params(labelsize=12)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=12)
    axis.text(
        0.03,
        0.97,
        f"Positive: {positive_count}\nNegative: {negative_count}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    return auroc, len(kcnh2) - len(available)


def main() -> None:
    args = parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be at least 1")
    targets = read_targets(args.target_file)
    gene_names = load_gene_names(args.uniprot_file)
    positive_ranks = calculate_ranks(
        args.positive_docking_files,
        targets,
        gene_names,
        "Positive",
    )
    negative_ranks = calculate_ranks(
        args.negative_docking_files,
        targets,
        gene_names,
        "Negative",
    )
    ranks = pd.concat([positive_ranks, negative_ranks], ignore_index=True)
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    ranks.to_csv(args.output_data, index=False)
    draw_heatmap(ranks, targets, args.output, args.dpi)
    auroc, missing_kcnh2 = draw_kcnh2_auroc(ranks, args.auroc_output, args.dpi)
    print(
        f"Calculated {len(targets)} target ranks for "
        f"{ranks['Drug'].nunique()} drugs "
        f"({positive_ranks['Drug'].nunique()} positive and "
        f"{negative_ranks['Drug'].nunique()} negative)."
    )
    print(f"Rank table saved to: {args.output_data}")
    print(f"Rank heatmap saved to: {args.output}")
    print(
        f"KCNH2 AUROC: {auroc:.3f} "
        f"({missing_kcnh2} compounds excluded because KCNH2 was not found)"
    )
    print(f"KCNH2 AUROC curve saved to: {args.auroc_output}")


if __name__ == "__main__":
    main()
