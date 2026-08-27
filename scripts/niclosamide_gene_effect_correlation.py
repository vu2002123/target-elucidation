"""Correlate niclosamide log2(AUC) with CRISPR gene effects in CRC cell lines."""

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

DEFAULT_PRISM_FILE = (
    INTERIM_DIR
    / "prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv"
)
DEFAULT_TARGET_FILE = INTERIM_DIR / "NEN_targets.txt"
DEFAULT_GENE_EFFECT_FILE = RAW_DIR / "CRISPRGeneEffect.csv"
DEFAULT_MODEL_FILE = RAW_DIR / "Model.csv"
DEFAULT_RESULTS_FILE = INTERIM_DIR / "niclosamide_crc_gene_effect_correlations.csv"
DEFAULT_FIGURE_FILE = FIGURE_DIR / "niclosamide_crc_gene_effect_correlations.png"
CRC_ONCOTREE_CODES = ("COAD", "READ", "COADREAD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prism-file", type=Path, default=DEFAULT_PRISM_FILE)
    parser.add_argument("--target-file", type=Path, default=DEFAULT_TARGET_FILE)
    parser.add_argument("--gene-effect-file", type=Path, default=DEFAULT_GENE_EFFECT_FILE)
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_FIGURE_FILE)
    parser.add_argument(
        "--significance-threshold",
        type=float,
        default=0.05,
        help="Raw p-value threshold used for highlighting (default: 0.05).",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_target_genes(path: Path) -> list[str]:
    """Read unique gene symbols while retaining file order."""
    genes = []
    seen = set()
    with path.open() as handle:
        for line in handle:
            gene = line.strip().upper()
            if gene and gene not in seen:
                genes.append(gene)
                seen.add(gene)
    if not genes:
        raise ValueError(f"No gene symbols found in {path}")
    return genes


def find_gene_columns(path: Path, genes: list[str]) -> tuple[str, dict[str, str]]:
    """Map requested symbols to full DepMap gene-effect column names."""
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    requested = set(genes)
    gene_columns = {}
    for column in columns[1:]:
        symbol = column.split(" ", maxsplit=1)[0].upper()
        if symbol in requested and symbol not in gene_columns:
            gene_columns[symbol] = column
    missing = [gene for gene in genes if gene not in gene_columns]
    if missing:
        warnings.warn(
            "Skipping genes absent from CRISPRGeneEffect.csv: " + ", ".join(missing),
            stacklevel=2,
        )
    if not gene_columns:
        raise ValueError("None of the target genes occur in CRISPRGeneEffect.csv")
    return columns[0], gene_columns


def load_crc_model_ids(path: Path) -> set[str]:
    """Return models annotated as COAD, READ, or COADREAD."""
    models = pd.read_csv(path, usecols=["ModelID", "OncotreeCode"])
    model_ids = set(
        models.loc[models["OncotreeCode"].isin(CRC_ONCOTREE_CODES), "ModelID"].dropna()
    )
    if not model_ids:
        raise ValueError(
            "No models found with OncotreeCode " + ", ".join(CRC_ONCOTREE_CODES)
        )
    return model_ids


def load_niclosamide_log2_auc(path: Path, crc_model_ids: set[str]) -> pd.Series:
    """Load niclosamide AUC, convert to log2, and median-aggregate duplicate models."""
    prism = pd.read_csv(path, usecols=["depmap_id", "name", "auc"])
    niclosamide = prism[
        prism["name"].astype("string").str.strip().str.casefold().eq("niclosamide")
        & prism["depmap_id"].isin(crc_model_ids)
    ].copy()
    niclosamide["auc"] = pd.to_numeric(niclosamide["auc"], errors="coerce")
    invalid_auc = niclosamide["auc"].isna() | niclosamide["auc"].le(0)
    if invalid_auc.any():
        warnings.warn(
            f"Dropping {int(invalid_auc.sum())} niclosamide rows with missing/non-positive AUC.",
            stacklevel=2,
        )
        niclosamide = niclosamide.loc[~invalid_auc]
    if niclosamide.empty:
        raise ValueError("No valid niclosamide observations remain in CRC cell lines")
    niclosamide["log2_auc"] = np.log2(niclosamide["auc"])
    return niclosamide.groupby("depmap_id")["log2_auc"].median()


def calculate_correlations(
    genes: list[str],
    gene_effect_file: Path,
    log2_auc: pd.Series,
) -> pd.DataFrame:
    """Calculate per-gene Spearman rho and two-sided p-value."""
    index_column, gene_columns = find_gene_columns(gene_effect_file, genes)
    available_genes = [gene for gene in genes if gene in gene_columns]
    effects = pd.read_csv(
        gene_effect_file,
        usecols=[index_column, *(gene_columns[gene] for gene in available_genes)],
        index_col=index_column,
    )
    common_models = effects.index.intersection(log2_auc.index)
    if common_models.empty:
        raise ValueError("No overlapping CRC models between PRISM and CRISPR gene effects")

    rows = []
    for gene in available_genes:
        paired = pd.DataFrame(
            {
                "log2_auc": log2_auc.reindex(common_models),
                "gene_effect": effects.loc[common_models, gene_columns[gene]],
            }
        ).dropna()
        if len(paired) < 3 or paired["gene_effect"].nunique() < 2:
            rho, p_value = np.nan, np.nan
        else:
            result = spearmanr(paired["log2_auc"], paired["gene_effect"])
            rho, p_value = float(result.statistic), float(result.pvalue)
        rows.append(
            {
                "gene": gene,
                "spearman_rho": rho,
                "p_value": p_value,
                "cell_line_count": len(paired),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["p_value", "spearman_rho"], ascending=[True, False], na_position="last"
    )


def plot_correlations(
    correlations: pd.DataFrame,
    output: Path,
    significance_threshold: float,
    dpi: int,
) -> None:
    """Plot rho against -log10(p), highlighting significant genes."""
    plot_data = correlations.dropna(subset=["spearman_rho", "p_value"]).copy()
    if plot_data.empty:
        raise ValueError("No valid correlations are available to plot")
    plot_data["minus_log10_p"] = -np.log10(
        plot_data["p_value"].clip(lower=np.finfo(float).tiny)
    )
    plot_data["significant"] = plot_data["p_value"] < significance_threshold

    figure, ax = plt.subplots(figsize=(8, 6))
    colors = plot_data["significant"].map({True: "crimson", False: "steelblue"})
    ax.scatter(
        plot_data["spearman_rho"],
        plot_data["minus_log10_p"],
        c=colors,
        s=150,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.1,
    )
    for row in plot_data.loc[plot_data["significant"]].itertuples(index=False):
        ax.annotate(
            row.gene,
            (row.spearman_rho, row.minus_log10_p),
            xytext=(5, 5),
            textcoords="offset points",
            color="crimson",
            fontsize=15,
            fontweight="bold",
        )
    ax.axhline(
        -np.log10(significance_threshold),
        color="gray",
        linestyle="--",
        linewidth=1.6,
        label=f"p = {significance_threshold:g}",
    )
    ax.axvline(0, color="gray", linestyle=":", linewidth=1.6)
    rho_min = float(plot_data["spearman_rho"].min())
    rho_max = float(plot_data["spearman_rho"].max())
    rho_padding = max(0.12, 0.12 * (rho_max - rho_min))
    ax.set_xlim(max(-1.05, rho_min - rho_padding), min(1.05, rho_max + rho_padding))
    ax.set_ylim(0, float(plot_data["minus_log10_p"].max()) + 0.25)
    ax.set_xlabel("Spearman correlation (rho)", fontsize=18)
    ax.set_ylabel("-log10(p-value)", fontsize=18)
    ax.set_title(
        "Niclosamide log2(AUC) vs CRISPR gene effect in CRC cell lines",
        fontsize=19,
        pad=10,
    )
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(linestyle=":", alpha=0.35)
    ax.legend(loc="best", fontsize=14)
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 < args.significance_threshold < 1:
        raise ValueError("--significance-threshold must be between 0 and 1")
    genes = load_target_genes(args.target_file)
    crc_model_ids = load_crc_model_ids(args.model_file)
    log2_auc = load_niclosamide_log2_auc(args.prism_file, crc_model_ids)
    correlations = calculate_correlations(genes, args.gene_effect_file, log2_auc)
    correlations["significant"] = correlations["p_value"] < args.significance_threshold

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(args.results_file, index=False)
    plot_correlations(
        correlations,
        args.output,
        args.significance_threshold,
        args.dpi,
    )
    print(
        f"Calculated {len(correlations)} gene correlations across "
        f"{len(log2_auc)} CRC cell lines with niclosamide data."
    )
    print(f"Significant genes (p < {args.significance_threshold:g}): "
          f"{int(correlations['significant'].sum())}")
    print(f"Results saved to {args.results_file}")
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
