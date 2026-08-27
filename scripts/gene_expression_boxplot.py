#!/usr/bin/env python3

"""Compare normal and tumor expression for one gene with a boxplot."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a normal-versus-tumor expression boxplot for one gene."
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
            "reports/figures/<cancer_type>_<gene>_expression_boxplot.png)."
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Output resolution (default: 300)."
    )
    return parser.parse_args()


def expression_file(cancer_type: str, tissue: str) -> Path:
    """Resolve an expression file while allowing case-insensitive cancer input."""
    requested = INTERIM_DIR / f"{cancer_type}_{tissue}_normalized.csv"
    if requested.is_file():
        return requested

    expected_name = requested.name.casefold()
    matches = [
        path
        for path in INTERIM_DIR.glob(f"*_{tissue}_normalized.csv")
        if path.name.casefold() == expected_name
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Expression file not found: {requested}")


def load_gene_expression(path: Path, gene: str) -> np.ndarray:
    """Load all sample values for a gene from a wide expression matrix."""
    header = pd.read_csv(path, nrows=0)
    if "Gene_name" not in header.columns:
        raise ValueError(f"{path} does not contain a 'Gene_name' column")

    table = pd.read_csv(path)
    symbols = table["Gene_name"].astype("string").str.strip()
    selected = table.loc[symbols.str.casefold() == gene.strip().casefold()]
    if selected.empty:
        raise ValueError(f"Gene '{gene}' was not found in {path}")

    values = selected.drop(columns="Gene_name").stack()
    values = pd.to_numeric(values, errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if values.empty:
        raise ValueError(f"Gene '{gene}' has no numeric expression values in {path}")
    return values.to_numpy(dtype=float)


def draw_boxplot(
    normal: np.ndarray,
    tumor: np.ndarray,
    cancer_type: str,
    gene: str,
    output: Path,
    dpi: int,
) -> None:
    if dpi <= 0:
        raise ValueError("--dpi must be greater than zero")

    figure, ax = plt.subplots(figsize=(6.5, 6))
    box = ax.boxplot(
        [normal, tumor],
        tick_labels=[f"Normal\n(n={len(normal)})", f"Tumor\n(n={len(tumor)})"],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 1.6},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
    )
    for patch, color in zip(box["boxes"], ["#4C78A8", "#E45756"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

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
    ax.set_ylim(top=data_max + 0.22 * data_range)

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
        / f"{args.cancer_type.upper()}_{args.gene.upper()}_expression_boxplot.png"
    )
    draw_boxplot(normal, tumor, args.cancer_type, args.gene, output, args.dpi)
    print(f"Saved boxplot to {output}")


if __name__ == "__main__":
    main()
