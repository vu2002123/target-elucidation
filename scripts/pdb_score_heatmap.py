#!/usr/bin/env python3

"""Draw PDB cross-docking heatmaps without cognate-ligand information."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "interim" / "EGFR_PDB_out.csv"
DEFAULT_PDB_METADATA = DATA_DIR / "raw" / "PDB" / "PDB_IDs.csv"
DEFAULT_FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

# Optionally replace these placeholders with the compounds and structures to plot.
# An empty list includes every value found in the input file.
DRUGS = [
    "Erlotinib",
    "Gefitinib",
    "Afatinib",
]
PDB_IDS = [
    "1M17",
    "2ITY",
    "2ITZ",
    "4I22",
]

SCORE_SETTINGS = {
    "CNNaffinity": {
        "title": "EGFR cross-docking: GNINA predicted affinity",
        "color_map": "Blues",
        "filename": "pdb_cnn_affinity_heatmap_EGFR.png",
    },
    "CNN_VS": {
        "title": "EGFR cross-docking: GNINA combined score",
        "color_map": "Blues",
        "filename": "pdb_cnn_vs_heatmap_EGFR.png",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw PDB cross-docking heatmaps without cognate-ligand annotations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Collected PDB docking-score CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Figure output directory (default: {DEFAULT_FIGURE_DIR}).",
    )
    parser.add_argument(
        "--pdb-metadata",
        type=Path,
        default=DEFAULT_PDB_METADATA,
        help=f"PDB ID-to-target-name CSV (default: {DEFAULT_PDB_METADATA}).",
    )
    parser.add_argument(
        "--drugs",
        nargs="+",
        help="Drug names to plot, in display order (overrides the DRUGS list).",
    )
    parser.add_argument(
        "--pdb-ids",
        nargs="+",
        help="PDB IDs to plot, in display order (overrides the PDB_IDS list).",
    )
    return parser.parse_args()


def load_docking_scores(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    scores = pd.read_csv(path)
    required = {"Compound", "CNNaffinity", "File_Path"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing docking-score columns: {sorted(missing)}")

    scores["Compound"] = scores["Compound"].astype("string").str.strip()
    scores["PDB_ID"] = (
        scores["File_Path"]
        .astype("string")
        .map(lambda value: Path(value).parent.name.split("_")[0].upper())
    )
    scores["CNNaffinity"] = pd.to_numeric(scores["CNNaffinity"], errors="coerce")
    if "CNN_VS" in scores.columns:
        scores["CNN_VS"] = pd.to_numeric(scores["CNN_VS"], errors="coerce")
    else:
        if "CNNscore" not in scores.columns:
            raise ValueError("CNN_VS is absent and cannot be calculated without CNNscore")
        scores["CNN_VS"] = (
            pd.to_numeric(scores["CNNscore"], errors="coerce") * scores["CNNaffinity"]
        )
    return scores


def load_pdb_labels(path: Path) -> dict[str, str]:
    """Build PDB labels containing the target variant and cognate ligand."""
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = pd.read_csv(path)
    required = {"ID", "name", "cognate"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing PDB metadata columns: {sorted(missing)}")

    # The source CSV contains two columns named "name". Pandas renames the
    # second one (the cognate-ligand name) to "name.1".
    ligand_name_column = "name.1" if "name.1" in metadata.columns else None
    metadata = metadata.dropna(subset=["ID", "name"]).copy()
    metadata["ID"] = metadata["ID"].astype(str).str.strip().str.upper()
    metadata["target_name"] = (
        metadata["name"].astype(str).str.strip().str.replace("_", " ", regex=False)
    )
    metadata["cognate"] = metadata["cognate"].astype("string").str.strip()
    if ligand_name_column:
        metadata["ligand_name"] = (
            metadata[ligand_name_column].astype("string").str.strip()
        )
    else:
        metadata["ligand_name"] = metadata["cognate"]
    metadata = metadata.drop_duplicates("ID", keep="first")
    return {
        row.ID: (
            f"{row.ID}\n{row.target_name}\n"
            f"Cognate: {row.ligand_name} ({row.cognate})"
        )
        for row in metadata[
            ["ID", "target_name", "ligand_name", "cognate"]
        ].itertuples(index=False)
    }


def select_axis_values(
    scores: pd.DataFrame,
    requested_drugs: list[str],
    requested_pdb_ids: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    available_drugs = set(scores["Compound"].dropna())
    available_pdb_ids = set(scores["PDB_ID"].dropna())
    drugs = requested_drugs or sorted(available_drugs)
    pdb_ids = [pdb_id.upper() for pdb_id in requested_pdb_ids] or sorted(available_pdb_ids)

    missing_drugs = set(drugs) - available_drugs
    missing_pdb_ids = set(pdb_ids) - available_pdb_ids
    if missing_drugs:
        raise ValueError(f"Drugs not found in input: {sorted(missing_drugs)}")
    if missing_pdb_ids:
        raise ValueError(f"PDB IDs not found in input: {sorted(missing_pdb_ids)}")

    selected = scores[scores["Compound"].isin(drugs) & scores["PDB_ID"].isin(pdb_ids)].copy()
    if selected.empty:
        raise ValueError("No docking rows match the selected drugs and PDB IDs")
    return selected, drugs, pdb_ids


def draw_heatmap(
    scores: pd.DataFrame,
    drugs: list[str],
    pdb_ids: list[str],
    score_column: str,
    output_dir: Path,
    pdb_labels: dict[str, str],
) -> Path:
    matrix = scores.pivot_table(
        index="Compound",
        columns="PDB_ID",
        values=score_column,
        aggfunc="max",
    ).reindex(index=drugs, columns=pdb_ids)
    if matrix.notna().sum().sum() == 0:
        raise ValueError(f"No usable {score_column} values match the selections")

    settings = SCORE_SETTINGS[score_column]
    figure_width = max(14, 2.0 * len(matrix.columns) + 5)
    figure_height = max(10, len(matrix.index) + 4)
    figure, ax = plt.subplots(figsize=(figure_width, figure_height))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=settings["color_map"],
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 20},
        linewidths=0.8,
        linecolor="white",
        mask=matrix.isna(),
        cbar_kws={"label": score_column, "shrink": 0.85},
    )
    ax.set_title(settings["title"], fontsize=28, fontweight="bold", pad=22)
    ax.set_xlabel("PDB target structure", fontsize=22, labelpad=10)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=18)
    x_labels = [pdb_labels.get(pdb_id, pdb_id) for pdb_id in matrix.columns]
    ax.set_xticklabels(x_labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=17)
    colorbar.set_label(score_column, fontsize=20, labelpad=12)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / settings["filename"]
    figure.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"{score_column} heatmap saved to: {output_file}")
    return output_file


def main() -> None:
    args = parse_args()
    scores = load_docking_scores(args.input)
    pdb_labels = load_pdb_labels(args.pdb_metadata)
    requested_drugs = args.drugs if args.drugs is not None else DRUGS
    requested_pdb_ids = args.pdb_ids if args.pdb_ids is not None else PDB_IDS
    scores, drugs, pdb_ids = select_axis_values(scores, requested_drugs, requested_pdb_ids)
    for score_column in SCORE_SETTINGS:
        draw_heatmap(
            scores,
            drugs,
            pdb_ids,
            score_column,
            args.figure_dir,
            pdb_labels,
        )


if __name__ == "__main__":
    main()
