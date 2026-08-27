#!/usr/bin/env python3

"""Draw side-by-side affinity and combined-score heatmaps for NEN PDB docking."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "interim" / "NEN_g9a_out.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "reports" / "figures" / "NEN_PDB_score_g9a_heatmaps.png"

SCORE_SETTINGS = {
    "CNNaffinity": {
        "title": "GNINA predicted affinity",
        "colorbar": "CNN affinity",
    },
    "CNN_VS": {
        "title": "GNINA combined score",
        "colorbar": "CNN_VS",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_score_matrices(path: Path) -> dict[str, pd.DataFrame]:
    """Load docking scores and reshape them into compound-by-PDB matrices."""
    scores = pd.read_csv(path)
    required = {"Compound", "CNNaffinity", "CNN_VS", "File_Path"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Missing required column(s) in {path}: {', '.join(missing)}")

    scores["PDB_ID"] = (
        scores["File_Path"]
        .astype("string")
        .map(lambda value: Path(value).parent.name.split("_")[0])
    )
    for score_column in SCORE_SETTINGS:
        scores[score_column] = pd.to_numeric(scores[score_column], errors="coerce")

    compounds = scores["Compound"].dropna().astype(str).drop_duplicates().tolist()
    if "Niclosamide" in compounds:
        compounds = ["Niclosamide", *[name for name in compounds if name != "Niclosamide"]]
    pdb_ids = sorted(scores["PDB_ID"].dropna().unique())

    matrices = {}
    for score_column in SCORE_SETTINGS:
        matrix = scores.pivot_table(
            index="Compound",
            columns="PDB_ID",
            values=score_column,
            aggfunc="max",
        ).reindex(index=compounds, columns=pdb_ids)
        if matrix.empty or not matrix.notna().any().any():
            raise ValueError(f"No usable {score_column} values found in {path}")
        matrices[score_column] = matrix
    return matrices


def draw_heatmaps(matrices: dict[str, pd.DataFrame], output: Path, dpi: int) -> None:
    """Draw both score matrices in one figure using the same monochrome palette."""
    if dpi <= 0:
        raise ValueError("--dpi must be greater than zero")

    figure, axes = plt.subplots(1, 2, figsize=(14, 9))
    for panel_index, (ax, (score_column, settings)) in enumerate(
        zip(axes, SCORE_SETTINGS.items())
    ):
        matrix = matrices[score_column]
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="Blues",
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 13},
            linewidths=0.8,
            linecolor="white",
            mask=matrix.isna(),
            cbar_kws={"label": settings["colorbar"], "shrink": 0.78},
        )
        ax.set_title(settings["title"], fontsize=20, fontweight="bold", pad=14)
        ax.set_xlabel("PDB structure", fontsize=16)
        ax.set_ylabel("Compound" if panel_index == 0 else "", fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        if panel_index == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.tick_params(axis="y", labelleft=False)
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=12)
        colorbar.set_label(settings["colorbar"], fontsize=14, labelpad=9)

    figure.suptitle(
        "Niclosamide compound cross-docking scores",
        fontsize=23,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.88, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    matrices = load_score_matrices(args.input)
    draw_heatmaps(matrices, args.output, args.dpi)
    first_matrix = next(iter(matrices.values()))
    print(
        f"Plotted {len(first_matrix)} compounds across {len(first_matrix.columns)} PDB structures."
    )
    print(f"Heatmaps saved to {args.output}")


if __name__ == "__main__":
    main()
