"""Plot RAB15 expression in the SW480, RKO, and DLD1 active cell lines."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from top_target_extraction_2 import (
    EXPRESSION_THRESHOLD_TPM,
    PROJECT_DIR,
    load_active_cell_evidence,
)


DEFAULT_OUTPUT = PROJECT_DIR / "reports" / "figures" / "RAB15_active_cell_expression.png"
CELL_LINE_ORDER = ["SW480", "RKO", "DLD1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def draw_expression_plot(output: Path, dpi: int) -> dict[str, float]:
    """Load RAB15 expression with the target-extraction logic and draw the plot."""
    evidence = load_active_cell_evidence({"RAB15"})
    if evidence.empty:
        raise ValueError("No RAB15 expression evidence was found")
    row = evidence.iloc[0]
    expression = {
        cell_line: float(row[f"{cell_line}_expression_tpm"])
        for cell_line in CELL_LINE_ORDER
    }

    figure, ax = plt.subplots(figsize=(6, 5.5))
    bars = ax.bar(
        expression.keys(),
        expression.values(),
        color=["#4C78A8", "#F58518", "#54A24B"],
        edgecolor="black",
        linewidth=0.8,
        width=0.68,
    )
    for bar, value in zip(bars, expression.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(expression.values()) * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.axhline(
        EXPRESSION_THRESHOLD_TPM,
        color="crimson",
        linestyle="--",
        linewidth=1.6,
        label=f"Expression threshold ({EXPRESSION_THRESHOLD_TPM:g} TPM)",
    )
    ax.set_ylim(0, max(expression.values()) * 1.18)
    ax.set_xlabel("Cell line", fontsize=16)
    ax.set_ylabel("RAB15 expression (TPM)", fontsize=16)
    ax.set_title("RAB15 expression in active CRC cell lines", fontsize=18, pad=10)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc="upper left")
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return expression


def main() -> None:
    args = parse_args()
    expression = draw_expression_plot(args.output, args.dpi)
    print(
        "RAB15 expression (TPM): "
        + ", ".join(f"{cell}={value:.3f}" for cell, value in expression.items())
    )
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
