#!/usr/bin/env python3

"""Correlate PDB docking scores with user-provided compound IC50 values."""

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DOCKING_FILE = PROJECT_DIR / "data" / "interim" / "NEN_PDB_target_out.csv"
DEFAULT_RESULTS_FILE = (
    PROJECT_DIR / "data" / "interim" / "docking_ic50_correlations.csv"
)
DEFAULT_FIGURE_FILE = (
    PROJECT_DIR / "reports" / "figures" / "docking_ic50_correlations.png"
)
SCORE_COLUMNS = ("CNNaffinity", "CNN_VS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ic50_file",
        type=Path,
        help="CSV/TSV containing one compound-name column and one IC50 column.",
    )
    parser.add_argument(
        "--docking-file",
        type=Path,
        default=DEFAULT_DOCKING_FILE,
        help=f"Docking result CSV (default: {DEFAULT_DOCKING_FILE}).",
    )
    parser.add_argument(
        "--ic50-compound-column",
        default="Compound",
        help="Compound column in the IC50 file (default: Compound).",
    )
    parser.add_argument(
        "--ic50-column",
        default="IC50",
        help="IC50 value column (default: IC50).",
    )
    parser.add_argument(
        "--ic50-transform",
        choices=("log10", "none"),
        default="log10",
        help="IC50 transformation used for plotting (default: log10).",
    )
    parser.add_argument(
        "--significance-threshold",
        type=float,
        default=0.05,
        help="P-value threshold used to highlight panels (default: 0.05).",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help=f"Correlation result CSV (default: {DEFAULT_RESULTS_FILE}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_FIGURE_FILE,
        help=f"Output figure (default: {DEFAULT_FIGURE_FILE}).",
    )
    parser.add_argument(
        "--no-point-labels",
        action="store_true",
        help="Do not label individual compounds in scatter plots.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve_column(columns: list[str], requested: str) -> str:
    """Resolve a requested table column case-insensitively."""
    lookup = {column.casefold(): column for column in columns}
    resolved = lookup.get(requested.casefold())
    if resolved is None:
        raise ValueError(
            f"Column '{requested}' was not found. Available columns: {', '.join(columns)}"
        )
    return resolved


def read_delimited(path: Path) -> pd.DataFrame:
    """Read a CSV or TSV based on its extension."""
    if not path.is_file():
        raise FileNotFoundError(path)
    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=separator)


def load_ic50(args: argparse.Namespace) -> pd.DataFrame:
    """Load numeric IC50 values and median-aggregate duplicate compounds."""
    table = read_delimited(args.ic50_file)
    compound_column = resolve_column(table.columns.tolist(), args.ic50_compound_column)
    ic50_column = resolve_column(table.columns.tolist(), args.ic50_column)
    ic50 = table[[compound_column, ic50_column]].rename(
        columns={compound_column: "Compound", ic50_column: "IC50"}
    )
    ic50["Compound"] = ic50["Compound"].astype("string").str.strip()
    ic50["compound_key"] = ic50["Compound"].str.casefold()
    ic50["IC50"] = pd.to_numeric(ic50["IC50"], errors="coerce")
    invalid = ic50["compound_key"].isna() | ic50["IC50"].isna()
    if invalid.any():
        warnings.warn(f"Dropping {int(invalid.sum())} invalid IC50 rows.", stacklevel=2)
    ic50 = ic50.loc[~invalid]
    if args.ic50_transform == "log10":
        nonpositive = ic50["IC50"].le(0)
        if nonpositive.any():
            warnings.warn(
                f"Dropping {int(nonpositive.sum())} non-positive IC50 values before log10.",
                stacklevel=2,
            )
            ic50 = ic50.loc[~nonpositive]
    if ic50.empty:
        raise ValueError("No valid IC50 values remain")

    return (
        ic50.groupby("compound_key", as_index=False)
        .agg(Compound=("Compound", "first"), IC50=("IC50", "median"))
    )


def load_docking(path: Path) -> pd.DataFrame:
    """Load docking metrics and derive PDB IDs from File_Path when necessary."""
    docking = read_delimited(path)
    required = {"Compound", *SCORE_COLUMNS}
    missing = sorted(required.difference(docking.columns))
    if missing:
        raise ValueError(f"Missing docking column(s): {', '.join(missing)}")
    if "PDB_ID" not in docking.columns:
        if "File_Path" not in docking.columns:
            raise ValueError("Docking file needs either PDB_ID or File_Path")
        docking["PDB_ID"] = docking["File_Path"].astype("string").map(
            lambda value: Path(value).parent.name.split("_")[0]
        )
    docking["Compound"] = docking["Compound"].astype("string").str.strip()
    docking["compound_key"] = docking["Compound"].str.casefold()
    docking["PDB_ID"] = docking["PDB_ID"].astype("string").str.strip()
    for score_column in SCORE_COLUMNS:
        docking[score_column] = pd.to_numeric(docking[score_column], errors="coerce")

    # Match the heatmap scripts by retaining the best score for duplicate pairs.
    return (
        docking.groupby(["compound_key", "PDB_ID"], as_index=False)
        .agg(
            Compound=("Compound", "first"),
            **{score: (score, "max") for score in SCORE_COLUMNS},
        )
    )


def prepare_data(args: argparse.Namespace) -> pd.DataFrame:
    """Join docking scores to IC50 values by normalized compound name."""
    docking = load_docking(args.docking_file)
    ic50 = load_ic50(args)
    merged = docking.merge(
        ic50[["compound_key", "IC50"]],
        on="compound_key",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No compound names overlap between docking and IC50 files")
    merged["IC50_plot"] = (
        np.log10(merged["IC50"])
        if args.ic50_transform == "log10"
        else merged["IC50"]
    )
    return merged


def calculate_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-PDB Spearman correlation for each docking score."""
    rows = []
    for score_column in SCORE_COLUMNS:
        for pdb_id, group in data.groupby("PDB_ID", sort=False):
            paired = group[[score_column, "IC50_plot"]].dropna()
            if (
                len(paired) < 3
                or paired[score_column].nunique() < 2
                or paired["IC50_plot"].nunique() < 2
            ):
                rho, p_value = np.nan, np.nan
            else:
                result = spearmanr(paired[score_column], paired["IC50_plot"])
                rho, p_value = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "score": score_column,
                    "PDB_ID": pdb_id,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "compound_count": len(paired),
                }
            )
    return pd.DataFrame(rows)


def draw_correlations(
    data: pd.DataFrame,
    correlations: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Draw one annotated scatter panel for every score/PDB combination."""
    pdb_ids = data["PDB_ID"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(SCORE_COLUMNS),
        len(pdb_ids),
        figsize=(5 * len(pdb_ids), 9),
        squeeze=False,
    )
    y_label = "log10(IC50)" if args.ic50_transform == "log10" else "IC50"

    for row_index, score_column in enumerate(SCORE_COLUMNS):
        for column_index, pdb_id in enumerate(pdb_ids):
            ax = axes[row_index, column_index]
            panel = data.loc[data["PDB_ID"].eq(pdb_id)].dropna(
                subset=[score_column, "IC50_plot"]
            )
            stats = correlations[
                correlations["score"].eq(score_column)
                & correlations["PDB_ID"].eq(pdb_id)
            ].iloc[0]
            significant = pd.notna(stats["p_value"]) and (
                stats["p_value"] < args.significance_threshold
            )
            color = "crimson" if significant else "steelblue"
            ax.scatter(
                panel[score_column],
                panel["IC50_plot"],
                s=85,
                color=color,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.8,
            )
            if len(panel) >= 2 and panel[score_column].nunique() >= 2:
                slope, intercept = np.polyfit(panel[score_column], panel["IC50_plot"], 1)
                line_x = np.linspace(panel[score_column].min(), panel[score_column].max(), 100)
                ax.plot(line_x, slope * line_x + intercept, color="black", linewidth=1.5)
            if not args.no_point_labels:
                for point in panel.itertuples(index=False):
                    ax.annotate(
                        point.Compound,
                        (getattr(point, score_column), point.IC50_plot),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )
            p_text = (
                f"{stats['p_value']:.3g}" if pd.notna(stats["p_value"]) else "NA"
            )
            rho_text = (
                f"{stats['spearman_rho']:.3f}"
                if pd.notna(stats["spearman_rho"])
                else "NA"
            )
            ax.text(
                0.04,
                0.96,
                f"ρ = {rho_text}\np = {p_text}\nn = {int(stats['compound_count'])}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": color},
            )
            ax.set_title(
                pdb_id,
                color=color if significant else "black",
                fontweight="bold" if significant else "normal",
            )
            ax.set_xlabel(score_column)
            ax.set_ylabel(y_label if column_index == 0 else "")
            ax.grid(linestyle=":", alpha=0.3)

    figure.suptitle(
        "Docking-score correlation with compound IC50",
        fontsize=20,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 < args.significance_threshold < 1:
        raise ValueError("--significance-threshold must be between 0 and 1")
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero")
    data = prepare_data(args)
    correlations = calculate_correlations(data)
    correlations["significant"] = (
        correlations["p_value"] < args.significance_threshold
    )
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(args.results_file, index=False)
    draw_correlations(data, correlations, args)
    print(
        f"Matched {data['Compound'].nunique()} compounds across "
        f"{data['PDB_ID'].nunique()} PDB structures."
    )
    print(f"Correlation results saved to {args.results_file}")
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
