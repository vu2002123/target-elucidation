"""Summarize the protein, pocket, and domain coverage of the three datasets.

Running this script writes one sorted UniProt ID list per dataset, a CSV summary,
and two bar plots: one for protein/pocket counts and one for domain counts.
Dataset 1 is omitted from the domain plot because it has no domain annotation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "interim"
DEFAULT_COVERAGE_FIGURE = (
    PROJECT_DIR / "reports" / "figures" / "dataset_protein_pocket_comparison.png"
)
DEFAULT_DOMAIN_FIGURE = PROJECT_DIR / "reports" / "figures" / "dataset_domain_comparison.png"
DEFAULT_SUMMARY = DEFAULT_DATA_DIR / "dataset_info_summary.csv"

DATASETS = {
    "Dataset 1": {
        "filename": "PCP_PCP_D1_new_all_pocket_score.csv",
        "domain_pattern": None,
    },
    "Dataset 2": {
        "filename": "PCP_PCP_D2_new_all_pocket_score.csv",
        "domain_pattern": r"_(PF\d+)_",
    },
    "Dataset 3": {
        "filename": "PCP_DS_PCP_all_pocket_score.csv",
        "domain_pattern": r"_(IPR\d+)_",
    },
}


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a score table and validate the columns needed for this analysis."""
    data = pd.read_csv(path)
    required_columns = {"UNIPROT_ID", "File_Name"}
    missing = required_columns.difference(data.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required column(s): {missing_text}")
    return data


def extract_uniprot_ids(data: pd.DataFrame) -> list[str]:
    """Return sorted, unique, non-empty UniProt accessions."""
    ids = data["UNIPROT_ID"].dropna().astype(str).str.strip()
    return sorted(ids[ids.ne("")].unique())


def extract_domain_ids(data: pd.DataFrame, pattern: str) -> list[str]:
    """Extract sorted unique Pfam/InterPro IDs from pocket filenames."""
    filenames = data["File_Name"].dropna().astype(str)
    domain_ids = filenames.str.extract(f"({pattern})", expand=False)

    # The supplied pattern contains the domain ID as an inner capture group.
    if isinstance(domain_ids, pd.DataFrame):
        domain_ids = domain_ids.iloc[:, -1]

    unmatched = filenames[domain_ids.isna()]
    if not unmatched.empty:
        examples = ", ".join(unmatched.head(3))
        raise ValueError(
            f"Could not extract a domain ID from {len(unmatched)} pocket filename(s), "
            f"including: {examples}"
        )
    return sorted(domain_ids.dropna().unique())


def write_id_list(ids: list[str], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("".join(f"{accession}\n" for accession in ids))


def summarize_datasets(data_dir: Path) -> pd.DataFrame:
    """Extract ID lists and calculate counts for all three datasets."""
    records = []

    for dataset_name, config in DATASETS.items():
        data = read_dataset(data_dir / config["filename"])
        uniprot_ids = extract_uniprot_ids(data)
        list_name = dataset_name.lower().replace(" ", "_") + "_uniprot_ids.txt"
        write_id_list(uniprot_ids, data_dir / list_name)

        domain_pattern = config["domain_pattern"]
        domain_count = pd.NA
        if domain_pattern is not None:
            domain_ids = extract_domain_ids(data, domain_pattern)
            domain_count = len(domain_ids)

        records.append(
            {
                "dataset": dataset_name,
                "number_of_proteins": len(uniprot_ids),
                # Each all-pocket-score row represents one unique docked pocket.
                "number_of_pockets": data["File_Name"].dropna().astype(str).nunique(),
                "number_of_domains": domain_count,
            }
        )

    return pd.DataFrame.from_records(records)


def annotate_bars(axis: plt.Axes, bars, values: pd.Series) -> None:
    """Add comma-formatted counts above a collection of bars."""
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{int(value):,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=15,
        )


def style_axis(axis: plt.Axes) -> None:
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", labelsize=15)
    axis.spines[["top", "right", "bottom", "left"]].set_visible(True)
    axis.grid(axis="y", linestyle=":", alpha=0.4)
    axis.set_axisbelow(True)
    axis.margins(y=0.15)


def save_figure(figure: plt.Figure, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_protein_pocket_counts(summary: pd.DataFrame, output_file: Path) -> None:
    """Draw grouped protein and pocket counts for all three datasets."""
    metrics = [
        ("number_of_proteins", "Proteins", "#4C78A8"),
        ("number_of_pockets", "Pockets", "#F58518"),
    ]
    x_positions = range(len(summary))
    width = 0.34

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for offset, (column, label, color) in zip((-width / 2, width / 2), metrics):
        values = summary[column]
        bars = axis.bar(
            [x + offset for x in x_positions],
            values,
            width,
            label=label,
            color=color,
        )
        annotate_bars(axis, bars, values)

    axis.set_xticks(list(x_positions), summary["dataset"])
    axis.set_title("Protein and pocket coverage by dataset", fontsize=20)
    axis.legend(frameon=False, fontsize=15)
    style_axis(axis)
    save_figure(figure, output_file)


def plot_domain_counts(summary: pd.DataFrame, output_file: Path) -> None:
    """Draw domain counts for Datasets 2 and 3; Dataset 1 is not applicable."""
    domain_summary = summary.dropna(subset=["number_of_domains"]).copy()
    values = pd.to_numeric(domain_summary["number_of_domains"])

    figure, axis = plt.subplots(figsize=(7, 5.5))
    bars = axis.bar(domain_summary["dataset"], values, width=0.55, color="#54A24B")
    annotate_bars(axis, bars, values)
    axis.set_title("Domain number by dataset", fontsize=15)
    style_axis(axis)
    save_figure(figure, output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing the score CSVs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"Output summary CSV (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--coverage-figure",
        type=Path,
        default=DEFAULT_COVERAGE_FIGURE,
        help=f"Protein/pocket plot (default: {DEFAULT_COVERAGE_FIGURE})",
    )
    parser.add_argument(
        "--domain-figure",
        type=Path,
        default=DEFAULT_DOMAIN_FIGURE,
        help=f"Domain plot (default: {DEFAULT_DOMAIN_FIGURE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_datasets(args.data_dir)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    plot_protein_pocket_counts(summary, args.coverage_figure)
    plot_domain_counts(summary, args.domain_figure)

    print(summary.to_string(index=False))
    print(f"\nUniProt ID lists written to: {args.data_dir}")
    print(f"Summary written to: {args.summary}")
    print(f"Protein/pocket figure written to: {args.coverage_figure}")
    print(f"Domain figure written to: {args.domain_figure}")


if __name__ == "__main__":
    main()
