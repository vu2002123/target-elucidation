#!/usr/bin/env python3

"""Compare normal and tumor expression for one gene with a violin plot."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from gene_expression_boxplot import (
    FIGURE_DIR,
    expression_file,
    load_gene_expression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a normal-versus-tumor expression violin plot for one gene."
    )
    parser.add_argument(
        "cancer_type",
        help="Cancer type used in the input filenames, for example LUAD or CRC.",
    )
    parser.add_argument("gene", help="Gene symbol to compare, for example EGFR.")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output image path (default: "
            "reports/figures/<cancer_type>_<gene>_expression_violinplot.png)."
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Output resolution (default: 300)."
    )
    return parser.parse_args()


def draw_violinplot(
    normal: np.ndarray,
    tumor: np.ndarray,
    cancer_type: str,
    gene: str,
    output: Path,
    dpi: int,
) -> None:
    """Draw violins, medians, and a Mann–Whitney p-value annotation."""
    if dpi <= 0:
        raise ValueError("--dpi must be greater than zero")

    figure, ax = plt.subplots(figsize=(6.5, 6))
    violin = ax.violinplot(
        [normal, tumor],
        positions=[1, 2],
        widths=0.8,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        points=200,
    )
    for body, color in zip(violin["bodies"], ["#4C78A8", "#E45756"]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.8)
        body.set_linewidth(0.8)
    violin["cmedians"].set_color("black")
    violin["cmedians"].set_linewidth(2)
    for part in ["cbars", "cmins", "cmaxes"]:
        violin[part].set_color("black")
        violin[part].set_linewidth(1)

    _, p_value = mannwhitneyu(normal, tumor, alternative="two-sided")
    data_min = min(float(normal.min()), float(tumor.min()))
    data_max = max(float(normal.max()), float(tumor.max()))
    data_range = data_max - data_min
    if data_range == 0:
        data_range = max(abs(data_max), 1.0)
    bracket_y = data_max + 0.08 * data_range
    bracket_height = 0.035 * data_range
    ax.plot(
        [1, 1, 2, 2],
        [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
        color="black",
        linewidth=1.4,
    )
    p_text = f"p = {p_value:.2e}" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.text(
        1.5,
        bracket_y + bracket_height + 0.015 * data_range,
        p_text,
        ha="center",
        va="bottom",
        fontsize=14,
    )
    ax.set_ylim(data_min - 0.05 * data_range, data_max + 0.22 * data_range)
    ax.set_xticks(
        [1, 2],
        [f"Normal\n(n={len(normal)})", f"Tumor\n(n={len(tumor)})"],
    )
    ax.set_ylabel("Normalized gene expression", fontsize=17)
    ax.set_title(
        f"{gene.upper()} expression in {cancer_type.upper()}",
        fontsize=20,
        pad=14,
    )
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    tumor_file = expression_file(args.cancer_type, "Tumor")
    normal_file = expression_file(args.cancer_type, "Normal")
    tumor = load_gene_expression(tumor_file, args.gene)
    normal = load_gene_expression(normal_file, args.gene)

    output = args.output or (
        FIGURE_DIR
        / f"{args.cancer_type.upper()}_{args.gene.upper()}_expression_violinplot.png"
    )
    draw_violinplot(normal, tumor, args.cancer_type, args.gene, output, args.dpi)
    print(f"Saved violin plot to {output}")


if __name__ == "__main__":
    main()
