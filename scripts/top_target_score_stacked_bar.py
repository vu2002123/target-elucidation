"""Draw stacked bars of component scores for high-scoring target genes."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "interim" / "top_target_scores_NEN_3.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "reports" / "figures" / "top_target_scores_NEN_3_stacked_bar.png"

MEMBER_SCORES = {
    "docking_score": "Docking",
    "analogue_agreement_score": "Analogue agreement",
    "expression_score": "Expression",
    "dependency_score": "Dependency",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Target-score CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--minimum-total-score",
        type=float,
        default=5,
        help="Minimum total score required for inclusion (default: 5).",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        help="Maximum number of highest-ranked proteins to display (default: no limit).",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Output resolution (default: 600).")
    return parser.parse_args()


def load_score_matrix(
    path: Path,
    minimum_total_score: float,
    max_proteins: int | None = None,
) -> pd.DataFrame:
    """Load, filter, and arrange the member-score matrix."""
    if max_proteins is not None and max_proteins < 1:
        raise ValueError("--max-proteins must be at least 1")

    required_columns = {"gene_name", "total_score", *MEMBER_SCORES}
    scores = pd.read_csv(path)
    missing_columns = sorted(required_columns.difference(scores.columns))
    if missing_columns:
        raise ValueError(f"Missing required column(s) in {path}: {', '.join(missing_columns)}")

    top_genes = scores.loc[
        scores["total_score"].ge(minimum_total_score) & scores["gene_name"].notna()
    ].copy()
    if top_genes.empty:
        raise ValueError(f"No genes have total_score >= {minimum_total_score:g} in {path}")

    # Keep the highest-ranked entry if multiple UniProt targets map to the same gene.
    sort_columns = [
        "total_score",
        "docking_score",
        "analogue_agreement_score",
        "expression_score",
        "dependency_score",
    ]
    top_genes = top_genes.sort_values(
        sort_columns, ascending=False, kind="stable"
    ).drop_duplicates("gene_name", keep="first")
    if max_proteins is not None:
        top_genes = top_genes.head(max_proteins)
    matrix = top_genes.set_index("gene_name")[list(MEMBER_SCORES)].rename(columns=MEMBER_SCORES)
    return matrix


def draw_stacked_bar(matrix: pd.DataFrame, output: Path, dpi: int) -> None:
    """Draw compact horizontal stacked bars using member scores only."""
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    figure_height = max(6.0, 0.45 * len(matrix) + 1.5)
    figure, ax = plt.subplots(figsize=(8, figure_height))
    left = pd.Series(0.0, index=matrix.index)

    for (score_name, values), color in zip(matrix.items(), colors):
        bars = ax.barh(
            matrix.index,
            values,
            left=left,
            label=score_name,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=11,
                    fontweight="bold",
                )
        left = left + values

    highest_score = float(left.max())
    highest_genes = set(left.index[left.eq(highest_score)])
    for position, gene in enumerate(matrix.index):
        if gene in highest_genes:
            ax.add_patch(
                Rectangle(
                    (0, position - 0.4),
                    highest_score,
                    0.8,
                    fill=False,
                    edgecolor="crimson",
                    linewidth=2.2,
                    zorder=4,
                )
            )

    ax.invert_yaxis()
    ax.set_xlabel("Cumulative member score")
    ax.set_ylabel("Gene")
    ax.set_title("Component scores for top target genes")
    ax.set_xlim(0, max(8, float(left.max()) + 0.5))
    ax.set_xticks(range(0, 9))
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    for tick_label in ax.get_yticklabels():
        if tick_label.get_text() in highest_genes:
            tick_label.set_color("crimson")
            tick_label.set_fontweight("bold")
    ax.legend(
        title="Score component",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    matrix = load_score_matrix(
        args.input,
        args.minimum_total_score,
        args.max_proteins,
    )
    draw_stacked_bar(matrix, args.output, args.dpi)
    print(f"Plotted {len(matrix)} genes using {len(matrix.columns)} member scores.")
    print(f"Saved stacked bar plot to {args.output}")


if __name__ == "__main__":
    main()
