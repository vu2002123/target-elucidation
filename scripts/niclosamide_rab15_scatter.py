"""Plot niclosamide log2(AUC) against RAB15 CRISPR gene effect in CRC cell lines."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from niclosamide_gene_effect_correlation import (
    DEFAULT_GENE_EFFECT_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_PRISM_FILE,
    FIGURE_DIR,
    find_gene_columns,
    load_crc_model_ids,
    load_niclosamide_log2_auc,
)


DEFAULT_OUTPUT = FIGURE_DIR / "niclosamide_rab15_gene_effect_scatter.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prism-file", type=Path, default=DEFAULT_PRISM_FILE)
    parser.add_argument("--gene-effect-file", type=Path, default=DEFAULT_GENE_EFFECT_FILE)
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_rab15_data(
    prism_file: Path,
    gene_effect_file: Path,
    model_file: Path,
) -> pd.DataFrame:
    """Align niclosamide log2(AUC) and RAB15 effects by CRC DepMap model."""
    crc_model_ids = load_crc_model_ids(model_file)
    log2_auc = load_niclosamide_log2_auc(prism_file, crc_model_ids)
    index_column, columns = find_gene_columns(gene_effect_file, ["RAB15"])
    effects = pd.read_csv(
        gene_effect_file,
        usecols=[index_column, columns["RAB15"]],
        index_col=index_column,
    )
    paired = pd.DataFrame(
        {
            "niclosamide_log2_auc": log2_auc,
            "RAB15_gene_effect": effects[columns["RAB15"]],
        }
    ).dropna()
    if len(paired) < 3:
        raise ValueError(f"Only {len(paired)} paired RAB15 observations are available")
    return paired


def draw_scatter(data: pd.DataFrame, output: Path, dpi: int) -> tuple[float, float]:
    """Draw the paired scatter plot and return Spearman rho and p-value."""
    result = spearmanr(data["niclosamide_log2_auc"], data["RAB15_gene_effect"])
    rho, p_value = float(result.statistic), float(result.pvalue)

    figure, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        data["niclosamide_log2_auc"],
        data["RAB15_gene_effect"],
        s=130,
        color="crimson",
        alpha=0.8,
        edgecolor="white",
        linewidth=1,
    )

    # Add a simple linear trend as a visual guide; inference remains Spearman-based.
    x = data["niclosamide_log2_auc"].to_numpy()
    y = data["RAB15_gene_effect"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(x.min(), x.max(), 100)
    ax.plot(line_x, slope * line_x + intercept, color="black", linewidth=2, alpha=0.75)

    ax.text(
        0.04,
        0.96,
        f"Spearman ρ = {rho:.3f}\np = {p_value:.4g}\nn = {len(data)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.9},
    )
    ax.set_xlabel("Niclosamide log2(AUC)", fontsize=17)
    ax.set_ylabel("RAB15 CRISPR gene effect", fontsize=17)
    ax.set_title("RAB15 dependency and niclosamide response in CRC", fontsize=19, pad=10)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(linestyle=":", alpha=0.35)
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return rho, p_value


def main() -> None:
    args = parse_args()
    data = load_rab15_data(args.prism_file, args.gene_effect_file, args.model_file)
    rho, p_value = draw_scatter(data, args.output, args.dpi)
    print(
        f"RAB15: Spearman rho={rho:.4f}, p={p_value:.4g}, "
        f"n={len(data)} CRC cell lines"
    )
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
