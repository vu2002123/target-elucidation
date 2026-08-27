"""Plot compound ranking percentiles for the highest-scoring NEN targets."""

import argparse
from pathlib import Path

from top_target_compound_rank_percentiles import draw_plot, load_percentiles


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "interim" / "top_target_scores_NEN_3.csv"
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "reports"
    / "figures"
    / "top_NEN_target_compound_rank_percentiles.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero")
    data, highest_score = load_percentiles(args.input)
    draw_plot(data, highest_score, args.output, args.dpi)
    print(
        f"Plotted {data['Protein'].nunique()} proteins and "
        f"{data['Compound'].nunique()} compounds."
    )
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
