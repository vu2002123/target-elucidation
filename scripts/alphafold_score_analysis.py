#!/usr/bin/env python3

"""Draw cross-docking heatmaps from score_collection_alphafold.py output."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "interim" / "AF_cancer_drugs_out.csv"
CANCER_DRUG_FILE = DATA_DIR / "raw" / "cancer_drugs.csv"
DEFAULT_FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

SCORE_SETTINGS = {
    "CNNaffinity": {
        "title": "AlphaFold cross-docking scores: GNINA predicted affinity",
        "cmap": "viridis",
        "filename": "alphafold_cnn_affinity_heatmap.png",
    },
    "CNN_VS": {
        "title": "AlphaFold cross-docking scores: GNINA combined score",
        "cmap": "magma",
        "filename": "alphafold_cnn_vs_heatmap.png",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw AlphaFold cross-docking heatmaps and highlight true targets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Output CSV from score_collection_alphafold.py (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Figure output directory (default: {DEFAULT_FIGURE_DIR}).",
    )
    return parser.parse_args()


def load_inputs(score_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not score_file.is_file():
        raise FileNotFoundError(score_file)
    if not CANCER_DRUG_FILE.is_file():
        raise FileNotFoundError(CANCER_DRUG_FILE)

    scores = pd.read_csv(score_file)
    cancer_drugs = pd.read_csv(CANCER_DRUG_FILE, dtype="string")
    score_required = {"Compound", "UNIPROT_ID", "CNNaffinity"}
    reference_required = {"Compound", "Gene", "UNIPROT_ID"}
    missing_scores = score_required - set(scores.columns)
    missing_reference = reference_required - set(cancer_drugs.columns)
    if missing_scores:
        raise ValueError(f"Missing AlphaFold score columns: {sorted(missing_scores)}")
    if missing_reference:
        raise ValueError(f"Missing cancer-drug columns: {sorted(missing_reference)}")

    scores["UNIPROT_ID"] = scores["UNIPROT_ID"].astype("string").str.strip().str.upper()
    scores["CNNaffinity"] = pd.to_numeric(scores["CNNaffinity"], errors="coerce")
    if "CNN_VS" not in scores.columns:
        if "CNNscore" not in scores.columns:
            raise ValueError("CNN_VS is absent and cannot be calculated without CNNscore")
        scores["CNN_VS"] = (
            pd.to_numeric(scores["CNNscore"], errors="coerce")
            * scores["CNNaffinity"]
        )
    else:
        scores["CNN_VS"] = pd.to_numeric(scores["CNN_VS"], errors="coerce")

    for column in reference_required:
        cancer_drugs[column] = cancer_drugs[column].str.strip()
    cancer_drugs["UNIPROT_ID"] = cancer_drugs["UNIPROT_ID"].str.upper()
    cancer_drugs = cancer_drugs.dropna(subset=list(reference_required)).drop_duplicates(
        "UNIPROT_ID"
    )
    return scores, cancer_drugs


def ordered_axes(
    scores: pd.DataFrame, cancer_drugs: pd.DataFrame
) -> tuple[list[str], list[str]]:
    reference_ids = cancer_drugs["UNIPROT_ID"].tolist()
    true_compounds = dict(zip(cancer_drugs["UNIPROT_ID"], cancer_drugs["Compound"]))
    score_ids = set(scores["UNIPROT_ID"].dropna())
    uniprot_ids = [target_id for target_id in reference_ids if target_id in score_ids]
    uniprot_ids.extend(sorted(score_ids - set(uniprot_ids)))
    compounds = [true_compounds[target_id] for target_id in uniprot_ids if target_id in true_compounds]
    compounds.extend(sorted(set(scores["Compound"].dropna()) - set(compounds)))
    return uniprot_ids, compounds


def highlight_true_targets(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    true_compounds: dict[str, str],
) -> int:
    count = 0
    for column_index, uniprot_id in enumerate(matrix.columns):
        compound = true_compounds.get(uniprot_id)
        if compound not in matrix.index or pd.isna(matrix.loc[compound, uniprot_id]):
            continue
        row_index = matrix.index.get_loc(compound)
        ax.add_patch(
            Rectangle(
                (column_index, row_index),
                1,
                1,
                fill=False,
                edgecolor="red",
                linewidth=3.5,
                zorder=10,
            )
        )
        ax.text(
            column_index + 0.88,
            row_index + 0.16,
            "★",
            color="red",
            fontsize=22,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=11,
        )
        count += 1
    return count


def draw_heatmap(
    scores: pd.DataFrame,
    cancer_drugs: pd.DataFrame,
    score_column: str,
    output_dir: Path,
) -> Path:
    uniprot_ids, compounds = ordered_axes(scores, cancer_drugs)
    # Multiple AlphaFold models or pockets for a target are represented by their
    # best score, producing one compound-by-UniProt cell comparable to the PDB plot.
    matrix = scores.pivot_table(
        index="Compound",
        columns="UNIPROT_ID",
        values=score_column,
        aggfunc="max",
    ).reindex(index=compounds, columns=uniprot_ids)
    if matrix.empty or matrix.notna().sum().sum() == 0:
        raise ValueError(f"No usable {score_column} scores are available")

    genes = dict(zip(cancer_drugs["UNIPROT_ID"], cancer_drugs["Gene"]))
    true_compounds = dict(
        zip(cancer_drugs["UNIPROT_ID"], cancer_drugs["Compound"])
    )
    settings = SCORE_SETTINGS[score_column]
    figure, ax = plt.subplots(
        figsize=(max(14, 2.0 * len(matrix.columns) + 5), max(10, len(matrix.index) + 4))
    )
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=settings["cmap"],
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 20},
        linewidths=0.8,
        linecolor="white",
        mask=matrix.isna(),
        cbar_kws={"label": score_column, "shrink": 0.85},
    )
    highlighted = highlight_true_targets(ax, matrix, true_compounds)
    ax.set_title(settings["title"], fontsize=28, fontweight="bold", pad=22)
    ax.set_xlabel("AlphaFold target", fontsize=22, labelpad=10)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xticklabels(
        [f"{genes.get(target_id, 'Unknown')} ({target_id})" for target_id in matrix.columns],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=17)
    ax.collections[0].colorbar.set_label(score_column, fontsize=20, labelpad=12)
    if highlighted:
        ax.legend(
            handles=[
                Line2D(
                    [0], [0], marker="*", color="red", linestyle="none",
                    markersize=19, markerfacecolor="red",
                    label="True compound–target pair",
                )
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            frameon=False,
            fontsize=18,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / settings["filename"]
    figure.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"{score_column} heatmap saved to: {output_file}")
    return output_file


def main() -> None:
    args = parse_args()
    scores, cancer_drugs = load_inputs(args.input)
    for score_column in SCORE_SETTINGS:
        draw_heatmap(scores, cancer_drugs, score_column, args.figure_dir)


if __name__ == "__main__":
    main()
