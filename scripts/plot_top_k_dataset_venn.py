"""Draw a three-dataset top-K Venn diagram using the ``venn`` package."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from venn import venn

from export_known_target_ranks import DEFAULT_FILE_LOCATIONS, RANKING_METHODS
from export_top_k_dataset_intersection import (
    COMPOUND_ALIASES,
    DEFAULT_COMPOUND,
    DEFAULT_TOP_K,
    filename_slug,
    load_top_k_by_dataset,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = PROJECT_DIR / "reports" / "figures"
FIGURE_DPI = 600


def draw_venn_plot(
    dataset_sets: dict[str, set[str]],
    compound: str,
    top_k: int,
    ranking_label: str,
    output_file: Path,
) -> None:
    """Draw and save the top-K overlap without accompanying gene lists."""
    figure, axis = plt.subplots(figsize=(9, 8))
    venn(
        dataset_sets,
        ax=axis,
        cmap=["#1f77b4", "#ff7f0e", "#2ca02c"],
        alpha=0.5,
        fontsize=15,
        legend_loc="upper right",
    )
    # axis.set_title(
    #     f"{compound}: top-{top_k} protein overlap\n{ranking_label}",
    #     fontsize=20,
    #     pad=18,
    # )
    figure.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    figure.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compound", default=DEFAULT_COMPOUND)
    parser.add_argument("--compound-label")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--ranking-method",
        choices=RANKING_METHODS,
        default="CNN_VS",
    )
    parser.add_argument("--file-locations", type=Path, default=DEFAULT_FILE_LOCATIONS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    compound_label = args.compound_label or COMPOUND_ALIASES.get(args.compound, args.compound)
    output_file = args.output or (
        DEFAULT_FIGURE_DIR / f"{filename_slug(args.compound)}_top{args.top_k}_dataset_venn.png"
    )
    top_tables = load_top_k_by_dataset(
        args.compound,
        compound_label,
        args.top_k,
        args.ranking_method,
        args.file_locations,
    )
    dataset_sets = {
        f"Dataset {dataset}": set(table["UNIPROT_ID"]) for dataset, table in top_tables.items()
    }
    draw_venn_plot(
        dataset_sets,
        args.compound,
        args.top_k,
        RANKING_METHODS[args.ranking_method]["label"],
        output_file,
    )
    print(f"Venn diagram written to: {output_file}")
    print(f"SVG written to: {output_file.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
