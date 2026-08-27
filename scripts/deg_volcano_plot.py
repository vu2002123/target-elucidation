#!/usr/bin/env python3

"""Draw a DEG volcano plot and highlight user-specified genes."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a volcano plot from data/interim/DEG_<cancer_type>_all.csv "
            "and highlight selected genes."
        )
    )
    parser.add_argument(
        "--cancer-type",
        required=True,
        help="Cancer type used in the DEG filename, for example CRC or LUAD.",
    )
    parser.add_argument(
        "--genes",
        nargs="*",
        default=[],
        help="Gene symbols to highlight, separated by spaces.",
    )
    parser.add_argument(
        "--gene-file",
        type=Path,
        help="Optional text file containing one gene symbol per line.",
    )
    parser.add_argument(
        "--log2fc-threshold",
        type=float,
        default=1.0,
        help="Absolute log2 fold-change significance threshold (default: 1).",
    )
    parser.add_argument(
        "--padj-threshold",
        type=float,
        default=0.05,
        help="Adjusted p-value significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--output-prefix",
        help="Output filename prefix (default: volcano_<cancer_type>).",
    )
    parser.add_argument(
        "--zoom",
        nargs=4,
        type=float,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help=(
            "Add a zoomed inset using the supplied log2FC and -log10(padj) "
            "bounds, for example: --zoom -2 2 0 20."
        ),
    )
    return parser.parse_args()


def resolve_deg_file(cancer_type: str) -> Path:
    """Support both underscore and hyphen variants of the documented filename."""
    candidates = (
        INTERIM_DIR / f"DEG_{cancer_type}_all.csv",
        INTERIM_DIR / f"DEG_{cancer_type}-all.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find a DEG table. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def read_highlight_genes(direct_genes: list[str], gene_file: Path | None) -> list[str]:
    values = list(direct_genes)
    if gene_file is not None:
        if not gene_file.is_file():
            raise FileNotFoundError(gene_file)
        with gene_file.open() as handle:
            values.extend(line.strip() for line in handle)

    genes = []
    seen = set()
    for value in values:
        gene = value.strip().upper()
        if not gene or gene == "NAN" or gene in seen:
            continue
        seen.add(gene)
        genes.append(gene)
    return genes


def load_deg_table(path: Path) -> pd.DataFrame:
    deg = pd.read_csv(path)
    required = {"Gene_name", "log2FoldChange", "padj"}
    missing = required - set(deg.columns)
    if missing:
        raise ValueError(f"Missing DEG columns: {sorted(missing)}")

    deg = deg.copy()
    deg["Gene_name"] = deg["Gene_name"].astype("string").str.strip().str.upper()
    deg["log2FoldChange"] = pd.to_numeric(deg["log2FoldChange"], errors="coerce")
    deg["padj"] = pd.to_numeric(deg["padj"], errors="coerce")
    deg = deg.dropna(subset=["Gene_name", "log2FoldChange", "padj"])
    deg = deg[(deg["Gene_name"] != "") & deg["padj"].between(0, 1)]
    if deg.empty:
        raise ValueError(f"No usable differential-expression rows in {path}")

    positive_padj = deg.loc[deg["padj"] > 0, "padj"]
    minimum_plot_padj = positive_padj.min() / 10 if not positive_padj.empty else 1e-300
    minimum_plot_padj = max(float(minimum_plot_padj), np.finfo(float).tiny)
    deg["minus_log10_padj"] = -np.log10(deg["padj"].clip(lower=minimum_plot_padj))
    return deg


def draw_volcano_plot(
    deg: pd.DataFrame,
    cancer_type: str,
    highlight_genes: list[str],
    log2fc_threshold: float,
    padj_threshold: float,
    output_prefix: str,
    zoom_bounds: list[float] | None = None,
) -> tuple[Path, Path]:
    if log2fc_threshold < 0:
        raise ValueError("--log2fc-threshold must be non-negative")
    if not 0 < padj_threshold < 1:
        raise ValueError("--padj-threshold must be between 0 and 1")
    if zoom_bounds is not None:
        x_min, x_max, y_min, y_max = zoom_bounds
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("--zoom requires X_MIN < X_MAX and Y_MIN < Y_MAX")

    significant = deg["padj"] < padj_threshold
    upregulated = significant & (deg["log2FoldChange"] >= log2fc_threshold)
    downregulated = significant & (deg["log2FoldChange"] <= -log2fc_threshold)

    figure, ax = plt.subplots(figsize=(11, 8))
    point_settings = (
        (~(upregulated | downregulated), "Not significant", "#bdbdbd", 12, 0.45),
        (downregulated, "Downregulated", "#377eb8", 16, 0.65),
        (upregulated, "Upregulated", "#e41a1c", 16, 0.65),
    )
    for mask, label, color, size, alpha in point_settings:
        ax.scatter(
            deg.loc[mask, "log2FoldChange"],
            deg.loc[mask, "minus_log10_padj"],
            s=size,
            c=color,
            alpha=alpha,
            edgecolors="none",
            label=label,
            rasterized=False,
        )

    highlight_set = set(highlight_genes)
    highlighted = deg[deg["Gene_name"].isin(highlight_set)].copy()
    # Label one representative row per gene if a table contains duplicate symbols.
    highlighted = (
        highlighted.sort_values("minus_log10_padj", ascending=False)
        .drop_duplicates("Gene_name")
    )
    if not highlighted.empty:
        ax.scatter(
            highlighted["log2FoldChange"],
            highlighted["minus_log10_padj"],
            s=100,
            marker="*",
            c="#ffd92f",
            edgecolors="black",
            linewidths=0.8,
            label="Highlighted genes",
            zorder=5,
        )
    ax.axvline(log2fc_threshold, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(-log2fc_threshold, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(-np.log10(padj_threshold), color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("log₂ fold change", fontsize=16)
    ax.set_ylabel("−log₁₀ adjusted p-value", fontsize=16)
    ax.set_title(f"Differential expression in {cancer_type}", fontsize=20, pad=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=12, loc="upper right")

    if zoom_bounds is not None:
        x_min, x_max, y_min, y_max = zoom_bounds
        zoom_ax = inset_axes(ax, width="42%", height="42%", loc="center right", borderpad=2)
        for mask, _, color, size, alpha in point_settings:
            zoom_ax.scatter(
                deg.loc[mask, "log2FoldChange"],
                deg.loc[mask, "minus_log10_padj"],
                s=size,
                c=color,
                alpha=alpha,
                edgecolors="none",
                rasterized=False,
            )
        if not highlighted.empty:
            zoom_ax.scatter(
                highlighted["log2FoldChange"],
                highlighted["minus_log10_padj"],
                s=100,
                marker="*",
                c="#ffd92f",
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
        zoom_ax.axvline(log2fc_threshold, color="black", linestyle="--", linewidth=0.9)
        zoom_ax.axvline(-log2fc_threshold, color="black", linestyle="--", linewidth=0.9)
        zoom_ax.axhline(
            -np.log10(padj_threshold), color="black", linestyle="--", linewidth=0.9
        )
        zoom_ax.set_xlim(x_min, x_max)
        zoom_ax.set_ylim(y_min, y_max)
        zoom_ax.tick_params(axis="both", labelsize=10)
        zoom_ax.grid(alpha=0.18)
        mark_inset(
            ax,
            zoom_ax,
            loc1=2,
            loc2=4,
            fc="none",
            ec="black",
            linewidth=0.8,
        )
    if zoom_bounds is None:
        figure.tight_layout()
    else:
        # inset_axes is incompatible with tight_layout; use explicit margins
        # when a zoom panel is requested.
        figure.subplots_adjust(left=0.11, right=0.97, bottom=0.12, top=0.90)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_file = FIGURE_DIR / f"{output_prefix}.png"
    svg_file = FIGURE_DIR / f"{output_prefix}.svg"
    figure.savefig(png_file, dpi=600, bbox_inches="tight")
    figure.savefig(svg_file, bbox_inches="tight")
    plt.close(figure)
    return png_file, svg_file


def main() -> None:
    args = parse_args()
    cancer_type = args.cancer_type.strip().upper()
    deg_file = resolve_deg_file(cancer_type)
    highlight_genes = read_highlight_genes(args.genes, args.gene_file)
    deg = load_deg_table(deg_file)
    output_prefix = args.output_prefix or f"volcano_{cancer_type.lower()}"
    png_file, svg_file = draw_volcano_plot(
        deg=deg,
        cancer_type=cancer_type,
        highlight_genes=highlight_genes,
        log2fc_threshold=args.log2fc_threshold,
        padj_threshold=args.padj_threshold,
        output_prefix=output_prefix,
        zoom_bounds=args.zoom,
    )

    available_genes = set(deg["Gene_name"])
    missing_genes = [gene for gene in highlight_genes if gene not in available_genes]
    print(f"Loaded {len(deg):,} DEG rows from: {deg_file}")
    print(f"Highlighted {len(highlight_genes) - len(missing_genes):,} requested genes")
    if missing_genes:
        print("Requested genes absent from the DEG table: " + ", ".join(missing_genes))
    print(f"PNG saved to: {png_file}")
    print(f"SVG saved to: {svg_file}")


if __name__ == "__main__":
    main()
