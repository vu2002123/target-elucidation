#!/usr/bin/env python3

"""Analyse cross-docking scores and highlight cognate compound-target pairs."""

import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_DOCKING_FILE = DATA_DIR / "interim" / "PDB_cancer_drugs_out.csv"
CANCER_DRUG_FILE = DATA_DIR / "raw" / "cancer_drugs.csv"
COGNATE_FILE = DATA_DIR / "raw" / "PDB" / "rec_files.csv"
PDB_DIR = DATA_DIR / "raw" / "PDB"
DEFAULT_FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

SCORE_SETTINGS = {
    "CNNaffinity": {
        "title": "Cross-docking scores: GNINA predicted affinity",
        "color_map": "Blues",
        "format": ".2f",
    },
    "CNN_VS": {
        "title": "Cross-docking scores: GNINA combined score",
        "color_map": "Blues",
        "format": ".2f",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw PDB cross-docking heatmaps and highlight true targets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DOCKING_FILE,
        help=f"Collected PDB docking-score CSV (default: {DEFAULT_DOCKING_FILE}).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Figure output directory (default: {DEFAULT_FIGURE_DIR}).",
    )
    parser.add_argument(
        "--skip-rmsd",
        action="store_true",
        help="Skip the original cognate-pose RMSD calculation.",
    )
    return parser.parse_args()


def load_cancer_drugs(path: Path = CANCER_DRUG_FILE) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    cancer_drugs = pd.read_csv(path, dtype="string")
    required = {"Compound", "Gene", "PDB_ID", "UNIPROT_ID"}
    missing = required - set(cancer_drugs.columns)
    if missing:
        raise ValueError(f"Missing cancer-drug columns: {sorted(missing)}")
    for column in required:
        cancer_drugs[column] = cancer_drugs[column].str.strip()
    return cancer_drugs.dropna(subset=list(required)).drop_duplicates("PDB_ID")


def load_docking_scores(path: Path, cancer_drugs: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    docking_df = pd.read_csv(path)
    required = {"Compound", "CNNaffinity", "File_Path"}
    missing = required - set(docking_df.columns)
    if missing:
        raise ValueError(f"Missing docking-score columns: {sorted(missing)}")

    docking_df["PDB_ID"] = (
        docking_df["File_Path"]
        .astype("string")
        .map(lambda value: Path(value).parent.name.split("_")[0])
    )
    docking_df["CNNaffinity"] = pd.to_numeric(
        docking_df["CNNaffinity"], errors="coerce"
    )
    if "CNN_VS" not in docking_df.columns:
        if "CNNscore" not in docking_df.columns:
            raise ValueError("CNN_VS is absent and cannot be calculated without CNNscore")
        docking_df["CNN_VS"] = (
            pd.to_numeric(docking_df["CNNscore"], errors="coerce")
            * docking_df["CNNaffinity"]
        )
    else:
        docking_df["CNN_VS"] = pd.to_numeric(docking_df["CNN_VS"], errors="coerce")

    true_ligands = dict(zip(cancer_drugs["PDB_ID"], cancer_drugs["Compound"]))
    docking_df["Cognate_ligand"] = docking_df["PDB_ID"].map(true_ligands)
    return docking_df


def ordered_labels(
    docking_df: pd.DataFrame, cancer_drugs: pd.DataFrame
) -> tuple[list[str], list[str]]:
    reference_pdb_ids = cancer_drugs["PDB_ID"].tolist()
    true_ligands = dict(zip(cancer_drugs["PDB_ID"], cancer_drugs["Compound"]))
    pdb_ids = [pdb_id for pdb_id in reference_pdb_ids if pdb_id in set(docking_df["PDB_ID"])]
    pdb_ids.extend(
        sorted(set(docking_df["PDB_ID"].dropna()) - set(pdb_ids))
    )
    compounds = [true_ligands[pdb_id] for pdb_id in pdb_ids]
    compounds.extend(
        sorted(set(docking_df["Compound"].dropna()) - set(compounds))
    )
    return pdb_ids, compounds


def highlight_true_targets(
    ax: plt.Axes, matrix: pd.DataFrame, true_ligands: dict[str, str]
) -> int:
    highlighted = 0
    for column_index, pdb_id in enumerate(matrix.columns):
        true_ligand = true_ligands.get(pdb_id)
        if true_ligand not in matrix.index or pd.isna(matrix.loc[true_ligand, pdb_id]):
            continue
        row_index = matrix.index.get_loc(true_ligand)
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
        highlighted += 1
    return highlighted


def draw_score_heatmaps(
    docking_df: pd.DataFrame,
    cancer_drugs: pd.DataFrame,
    output_dir: Path,
) -> Path:
    pdb_ids, compounds = ordered_labels(docking_df, cancer_drugs)
    true_ligands = dict(zip(cancer_drugs["PDB_ID"], cancer_drugs["Compound"]))
    genes = dict(zip(cancer_drugs["PDB_ID"], cancer_drugs["Gene"]))
    matrices = {}
    for score_column in SCORE_SETTINGS:
        matrix = docking_df.pivot_table(
            index="Compound",
            columns="PDB_ID",
            values=score_column,
            aggfunc="max",
        ).reindex(index=compounds, columns=pdb_ids)
        if matrix.empty or matrix.notna().sum().sum() == 0:
            raise ValueError(f"No usable {score_column} scores are available")
        matrices[score_column] = matrix

    figure_width = max(20, 3.1 * len(pdb_ids) + 7)
    figure_height = max(9, 1.05 * len(compounds) + 4)
    figure, axes = plt.subplots(
        1,
        len(SCORE_SETTINGS),
        figsize=(figure_width, figure_height),
    )
    any_highlighted = False

    for panel_index, (ax, (score_column, settings)) in enumerate(
        zip(axes, SCORE_SETTINGS.items())
    ):
        matrix = matrices[score_column]
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=settings["color_map"],
            annot=True,
            fmt=settings["format"],
            annot_kws={"fontsize": 14},
            linewidths=0.8,
            linecolor="white",
            mask=matrix.isna(),
            cbar_kws={"label": score_column, "shrink": 0.78},
        )
        any_highlighted |= bool(highlight_true_targets(ax, matrix, true_ligands))

        ax.set_title(settings["title"], fontsize=20, fontweight="bold", pad=16)
        ax.set_xlabel("PDB target structure", fontsize=17, labelpad=8)
        ax.set_ylabel("Compound" if panel_index == 0 else "", fontsize=17)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_xticklabels(
            [f"{genes.get(pdb_id, 'Unknown')} ({pdb_id})" for pdb_id in matrix.columns],
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        if panel_index == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.tick_params(axis="y", labelleft=False)
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=13)
        colorbar.set_label(score_column, fontsize=15, labelpad=10)

    if any_highlighted:
        legend_handle = Line2D(
            [0],
            [0],
            marker="*",
            color="red",
            linestyle="none",
            markersize=17,
            markerfacecolor="red",
            label="True compound–target pair",
        )
        figure.legend(
            handles=[legend_handle],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            frameon=False,
            fontsize=16,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pdb_score_heatmaps_side_by_side.png"
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.90, wspace=0.28)
    figure.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"Side-by-side score heatmaps saved to: {output_file}")
    return output_file


def calculate_cognate_rmsd(docking_df: pd.DataFrame) -> dict[str, float]:
    """Retain the script's original RMSD analysis for cognate docking pairs."""
    if not COGNATE_FILE.is_file():
        print(f"Skipping RMSD: missing cognate file {COGNATE_FILE}")
        return {}
    if shutil.which("obrms") is None:
        print("Skipping RMSD: obrms was not found on PATH")
        return {}

    cognate_df = pd.read_csv(COGNATE_FILE)
    cognate_df["PDB_ID"] = cognate_df["receptor"].str.split("_").str[0]
    cognate_files = dict(zip(cognate_df["PDB_ID"], cognate_df["cognate_ligand"]))

    rmsd_scores = {}
    cognate_rows = docking_df[docking_df["Compound"] == docking_df["Cognate_ligand"]]
    for row in cognate_rows.itertuples(index=False):
        reference_file = PDB_DIR / cognate_files.get(row.PDB_ID, "")
        if not reference_file.is_file() or not Path(row.File_Path).is_file():
            continue
        completed = subprocess.run(
            ["obrms", "-f", str(row.File_Path), str(reference_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            print(f"Could not calculate RMSD for {row.Compound} / {row.PDB_ID}")
            continue
        try:
            rmsd_scores[f"{row.Compound}_{row.PDB_ID}"] = float(
                completed.stdout.strip().split()[-1]
            )
        except ValueError:
            print(f"Could not parse RMSD for {row.Compound} / {row.PDB_ID}")
    return rmsd_scores


def main() -> None:
    args = parse_args()
    cancer_drugs = load_cancer_drugs()
    docking_df = load_docking_scores(args.input, cancer_drugs)
    draw_score_heatmaps(docking_df, cancer_drugs, args.figure_dir)

    if not args.skip_rmsd:
        rmsd_scores = calculate_cognate_rmsd(docking_df)
        if rmsd_scores:
            rmsd_file = DATA_DIR / "interim" / "PDB_cancer_drugs_cognate_rmsd.csv"
            pd.DataFrame(
                rmsd_scores.items(), columns=["Compound_PDB", "RMSD"]
            ).to_csv(rmsd_file, index=False)
            print(f"Cognate-pose RMSD scores saved to: {rmsd_file}")


if __name__ == "__main__":
    main()
