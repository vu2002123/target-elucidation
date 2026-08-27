"""Plot CRISPR gene-effect distributions for one DepMap model type.

Example:
    python scripts/depmap_model_type_plot.py CRC data/interim/NEN_targets.txt \
        --highlight-genes RAB15
"""

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
DEFAULT_MODEL_FILE = RAW_DIR / "Model.csv"
DEFAULT_GENE_EFFECT_FILE = RAW_DIR / "CRISPRGeneEffect.csv"
DEFAULT_FIGURE_DIR = PROJECT_DIR / "reports" / "figures"


class GeneNotFoundError(ValueError):
    """Raised when a requested gene cannot be plotted from the selected data."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the distribution of CRISPR gene effects across all cell lines "
            "whose Model.csv DepmapModelType matches the requested cancer type ID. "
            "CRC selects OncotreeCode COAD, READ, and COADREAD."
        )
    )
    parser.add_argument(
        "cancer_type_id",
        help="Exact DepmapModelType value to select, or CRC for colorectal models.",
    )
    parser.add_argument(
        "gene_file",
        type=Path,
        help="Text file containing gene symbols, normally one gene per line.",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help=f"Model metadata CSV (default: {DEFAULT_MODEL_FILE}).",
    )
    parser.add_argument(
        "--gene-effect-file",
        type=Path,
        default=DEFAULT_GENE_EFFECT_FILE,
        help=f"CRISPR gene-effect CSV (default: {DEFAULT_GENE_EFFECT_FILE}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path (default: reports/figures/depmap_<ID>_gene_effect.png).",
    )
    parser.add_argument(
        "--highlight-genes",
        nargs="+",
        default=[],
        metavar="GENE",
        help="Requested genes whose boxes and axis labels should be highlighted.",
    )
    parser.add_argument(
        "--highlight-color",
        default="crimson",
        help="Matplotlib color used for highlighted genes (default: crimson).",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Output resolution (default: 600).")
    return parser.parse_args()


def normalize_genes(values: list[str]) -> list[str]:
    """Normalize gene symbols while retaining the user's requested order."""
    genes = []
    seen = set()
    for value in values:
        for gene in value.split(","):
            gene = gene.strip().upper()
            if gene and gene not in seen:
                genes.append(gene)
                seen.add(gene)
    if not genes:
        raise ValueError("At least one non-empty gene symbol is required.")
    return genes


def load_gene_file(path: Path) -> list[str]:
    """Load genes from a text file, allowing whitespace or comma separation."""
    values = []
    with path.open() as handle:
        for line in handle:
            line = line.split("#", maxsplit=1)[0].replace(",", " ")
            values.extend(line.split())
    if not values:
        raise ValueError(f"No gene symbols found in {path}")
    return normalize_genes(values)


def find_gene_columns(gene_effect_file: Path, genes: list[str]) -> tuple[str, dict[str, str]]:
    """Map requested gene symbols to their full DepMap column names."""
    columns = pd.read_csv(gene_effect_file, nrows=0).columns.tolist()
    if not columns:
        raise ValueError(f"No columns found in {gene_effect_file}")

    requested = set(genes)
    gene_columns = {}
    for column in columns[1:]:
        symbol = column.split(" ", maxsplit=1)[0].upper()
        if symbol in requested and symbol not in gene_columns:
            gene_columns[symbol] = column

    missing = [gene for gene in genes if gene not in gene_columns]
    if missing:
        warnings.warn(
            "Skipping gene(s) not found in CRISPRGeneEffect.csv: " + ", ".join(missing),
            stacklevel=2,
        )
    return columns[0], gene_columns


def load_cohort_ids(model_file: Path, cancer_type_id: str) -> pd.Index:
    """Return matching ModelIDs, with CRC expanded to its component Oncotree codes."""
    model_info = pd.read_csv(
        model_file,
        usecols=["ModelID", "DepmapModelType", "OncotreeCode"],
    )
    if cancer_type_id.upper() == "CRC":
        crc_codes = ("COAD", "READ", "COADREAD")
        cohort_ids = pd.Index(
            model_info.loc[
                model_info["OncotreeCode"].isin(crc_codes),
                "ModelID",
            ].dropna()
        )
        if cohort_ids.empty:
            raise ValueError(
                "No models found with OncotreeCode COAD, READ, or COADREAD."
            )
        return cohort_ids

    cohort_ids = pd.Index(
        model_info.loc[
            model_info["DepmapModelType"].eq(cancer_type_id),
            "ModelID",
        ].dropna()
    )
    if cohort_ids.empty:
        available = sorted(model_info["DepmapModelType"].dropna().astype(str).unique())
        suggestions = [value for value in available if cancer_type_id.lower() in value.lower()]
        message = f"No models found with DepmapModelType == {cancer_type_id!r}."
        if suggestions:
            message += " Similar values: " + ", ".join(suggestions[:10])
        raise ValueError(message)
    return cohort_ids


def make_plot(
    cancer_type_id: str,
    genes: list[str],
    model_file: Path,
    gene_effect_file: Path,
    output: Path,
    dpi: int,
    highlight_genes: list[str] | None = None,
    highlight_color: str = "crimson",
) -> tuple[int, list[str]]:
    highlight_genes = highlight_genes or []
    unknown_highlights = [gene for gene in highlight_genes if gene not in genes]
    if unknown_highlights:
        raise ValueError(
            "Highlighted gene(s) must also be included in the plotted genes: "
            + ", ".join(unknown_highlights)
        )

    cohort_ids = load_cohort_ids(model_file, cancer_type_id)
    index_column, gene_columns = find_gene_columns(gene_effect_file, genes)
    available_genes = [gene for gene in genes if gene in gene_columns]
    if not available_genes:
        raise GeneNotFoundError(
            "None of the requested genes were found in CRISPRGeneEffect.csv."
        )
    gene_effect = pd.read_csv(
        gene_effect_file,
        usecols=[index_column, *(gene_columns[gene] for gene in available_genes)],
        index_col=index_column,
    )
    cohort = gene_effect.loc[gene_effect.index.intersection(cohort_ids)].copy()
    if cohort.empty:
        raise ValueError(
            f"{len(cohort_ids)} matching models were found in Model.csv, but none "
            "are present in CRISPRGeneEffect.csv."
        )

    cohort = cohort.rename(columns={column: gene for gene, column in gene_columns.items()})
    plot_data = cohort.melt(var_name="Gene", value_name="CRISPR Gene Effect").dropna()
    genes_without_values = [
        gene for gene in available_genes if not plot_data["Gene"].eq(gene).any()
    ]
    if genes_without_values:
        warnings.warn(
            f"Skipping gene(s) without gene-effect values in {cancer_type_id}: "
            + ", ".join(genes_without_values),
            stacklevel=2,
        )
        plot_data = plot_data[~plot_data["Gene"].isin(genes_without_values)]
    if plot_data.empty:
        raise GeneNotFoundError(
            f"None of the requested genes have gene-effect values in {cancer_type_id}."
        )

    order = (
        plot_data.groupby("Gene", sort=False)["CRISPR Gene Effect"]
        .median()
        .sort_values(kind="stable")
        .index.tolist()
    )

    sns.set_theme(style="whitegrid", context="talk")
    width = max(8.0, 0.65 * len(order) + 2.5)
    _, ax = plt.subplots(figsize=(width, 6))
    sns.boxplot(
        data=plot_data,
        x="Gene",
        y="CRISPR Gene Effect",
        hue="Gene",
        order=order,
        hue_order=order,
        palette=[
            highlight_color if gene in highlight_genes else "steelblue"
            for gene in order
        ],
        fliersize=2,
        legend=False,
        ax=ax,
    )
    ax.axhline(-0.5, color="red", linestyle="--", linewidth=1.2, label="Dependency threshold")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_title(f"{cancer_type_id} CRISPR gene effect (n={len(cohort)})")
    ax.set_xlabel("Gene (ordered by median gene effect)")
    ax.tick_params(axis="x", labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() in highlight_genes:
            tick_label.set_color(highlight_color)
            tick_label.set_fontweight("bold")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="best", fontsize="small")

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", dpi=dpi)
    plt.close()
    return len(cohort), order


def main() -> None:
    args = parse_args()
    genes = load_gene_file(args.gene_file)
    highlight_genes = normalize_genes(args.highlight_genes) if args.highlight_genes else []
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.cancer_type_id)
    output = args.output or DEFAULT_FIGURE_DIR / f"depmap_{safe_id}_gene_effect.png"
    cell_count, order = make_plot(
        cancer_type_id=args.cancer_type_id,
        genes=genes,
        model_file=args.model_file,
        gene_effect_file=args.gene_effect_file,
        output=output,
        dpi=args.dpi,
        highlight_genes=highlight_genes,
        highlight_color=args.highlight_color,
    )
    print(f"Plotted {len(order)} genes across {cell_count} cell lines.")
    print("Left-to-right order: " + ", ".join(order))
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
