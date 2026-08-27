"""Plot compound ranking percentiles for proteins tied at the highest total score.

For each compound, the ranking percentile is calculated using a fixed total
protein count of 3,257:

    100 * (3,257 - protein rank + 1) / 3,257

Higher percentiles indicate better docking ranks.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "interim" / "top_target_scores_PCP_3.csv"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "reports" / "figures" / "top_PCP_target_compound_rank_percentiles.png"
)
TOTAL_PROTEIN_COUNT = 3_257


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def compound_name(rank_column: str, parent_drug: str) -> str:
    """Convert a Dataset 3 rank-column name into a readable compound label."""
    if rank_column == "dataset_3_rank":
        return parent_drug
    prefix = rank_column.removesuffix("_dataset_3_rank")
    if prefix.startswith("ccl_"):
        return "CCL-" + prefix.removeprefix("ccl_")
    name = prefix.replace("_", " ").title()
    return name.replace("N Desmethyl", "N-desmethyl")


def load_percentiles(path: Path) -> tuple[pd.DataFrame, float]:
    """Return long-format percentiles for every highest-total-score protein."""
    scores = pd.read_csv(path)
    required = {"drug", "total_score", "uniprot_id", "gene_name", "dataset_3_rank"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Missing required column(s) in {path}: {', '.join(missing)}")

    rank_columns = [
        column
        for column in scores.columns
        if column == "dataset_3_rank" or column.endswith("_dataset_3_rank")
    ]
    if not rank_columns:
        raise ValueError(f"No Dataset 3 rank columns found in {path}")

    highest_score = float(scores["total_score"].max())
    top = scores.loc[scores["total_score"].eq(highest_score)].copy()
    if top.empty:
        raise ValueError(f"No proteins with a valid total_score found in {path}")
    top["Protein"] = top["gene_name"].fillna(top["uniprot_id"])
    parent_drug = str(scores["drug"].dropna().iloc[0])

    rows = []
    for rank_column in rank_columns:
        for row in top.itertuples(index=False):
            rank = pd.to_numeric(getattr(row, rank_column), errors="coerce")
            if pd.notna(rank) and rank > TOTAL_PROTEIN_COUNT:
                raise ValueError(
                    f"Rank {rank:g} in {rank_column} exceeds the configured "
                    f"total protein count ({TOTAL_PROTEIN_COUNT:,})"
                )
            percentile = (
                100 * (TOTAL_PROTEIN_COUNT - rank + 1) / TOTAL_PROTEIN_COUNT
                if pd.notna(rank)
                else np.nan
            )
            rows.append(
                {
                    "Protein": row.Protein,
                    "Compound": compound_name(rank_column, parent_drug),
                    "Rank": rank,
                    "Ranking percentile": percentile,
                }
            )
    if not rows:
        raise ValueError("No valid compound ranks are available for the top proteins")
    return pd.DataFrame(rows), highest_score


def draw_plot(data: pd.DataFrame, highest_score: float, output: Path, dpi: int) -> None:
    """Draw grouped bars of compound ranking percentiles."""
    proteins = data["Protein"].drop_duplicates().tolist()
    compounds = data["Compound"].drop_duplicates().tolist()
    colors = plt.get_cmap("Set2").colors
    x = np.arange(len(proteins))
    width = min(0.24, 0.8 / max(len(compounds), 1))

    figure_width = max(7.0, 1.3 * len(proteins) + 3)
    figure, ax = plt.subplots(figsize=(figure_width, 6))
    for index, compound in enumerate(compounds):
        values = (
            data.loc[data["Compound"].eq(compound)]
            .set_index("Protein")["Ranking percentile"]
            .reindex(proteins)
        )
        offset = (index - (len(compounds) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=compound,
            color=colors[index % len(colors)],
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.6,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    rotation=90,
                )

    ax.set_xticks(x, proteins)
    # Leave enough headroom for vertical value labels on near-100th-percentile bars.
    ax.set_ylim(0, 115)
    ax.set_ylabel("Ranking percentile (%)", fontsize=16)
    ax.set_xlabel("Protein", fontsize=16)
    ax.set_title(
        "Compound docking percentiles",
        fontsize=18,
        pad=10,
    )
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(title="Compound", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero")
    data, highest_score = load_percentiles(args.input)
    draw_plot(data, highest_score, args.output, args.dpi)
    print(
        f"Plotted {data['Protein'].nunique()} proteins and {data['Compound'].nunique()} compounds."
    )
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
