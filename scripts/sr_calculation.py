#!/usr/bin/env python3

"""Compare paper and Dataset 3 Top-100 target-retrieval success rates."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

DEFAULT_DOCKING_FILE = INTERIM_DIR / "D1_validation_90cp_out.csv"
TARGET_FILE = RAW_DIR / "90cp_targets.csv"
OUTPUT_FILE = INTERIM_DIR / "dataset3_top100_success_rate_comparison.csv"
FIGURE_FILE = FIGURE_DIR / "dataset3_top100_success_rate_comparison.png"
TOP_K = 100
ORIGINAL_PAPER_TOP100_SUCCESS_RATE = 15.6

SORTING_METHODS = {
    "minimizedAffinity": {
        "label": "Dataset 3\nsmina",
        "ascending": True,
    },
    "CNNaffinity": {
        "label": "Dataset 3\nCNNaffinity",
        "ascending": False,
    },
    "CNN_VS": {
        "label": "Dataset 3\nCNN_VS",
        "ascending": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the original-paper Top-100 success rate with Dataset 3 "
            "rankings based on smina, CNNaffinity, and CNN_VS."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DOCKING_FILE,
        help=f"Dataset 3 docking-score CSV (default: {DEFAULT_DOCKING_FILE}).",
    )
    parser.add_argument(
        "--paper-success-rate",
        type=float,
        default=ORIGINAL_PAPER_TOP100_SUCCESS_RATE,
        help=(
            "Original-paper Top-100 success percentage "
            f"(default: {ORIGINAL_PAPER_TOP100_SUCCESS_RATE})."
        ),
    )
    return parser.parse_args()


def load_targets(path: Path = TARGET_FILE) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    targets = pd.read_csv(path, dtype="string")
    required = {"Compound", "Target"}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")
    targets["Compound"] = targets["Compound"].str.strip()
    targets["Target"] = targets["Target"].str.strip().str.upper()
    targets = targets.dropna(subset=["Compound", "Target"]).drop_duplicates("Compound")
    if targets.empty:
        raise ValueError(f"No compound-target pairs found in {path}")
    return targets


def load_docking_scores(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    scores = pd.read_csv(path)
    scores = scores.rename(
        columns={
            "CNN_affinity": "CNNaffinity",
            "CNN_pose_score": "CNNscore",
            "CNN_score": "CNNscore",
            "affinity": "minimizedAffinity",
            "ID": "UNIPROT_ID",
        }
    )
    required = {"Compound", "minimizedAffinity", "CNNaffinity", "File_Name"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing docking-score columns: {sorted(missing)}")

    if "UNIPROT_ID" not in scores.columns:
        scores["UNIPROT_ID"] = scores["File_Name"].str.split("_").str[0]
    scores["Compound"] = scores["Compound"].astype("string").str.strip()
    scores["UNIPROT_ID"] = (
        scores["UNIPROT_ID"].astype("string").str.strip().str.upper()
    )
    for column in ("minimizedAffinity", "CNNaffinity"):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    if "CNN_VS" in scores.columns:
        scores["CNN_VS"] = pd.to_numeric(scores["CNN_VS"], errors="coerce")
    else:
        if "CNNscore" not in scores.columns:
            raise ValueError("CNN_VS is absent and cannot be calculated without CNNscore")
        scores["CNNscore"] = pd.to_numeric(scores["CNNscore"], errors="coerce")
        scores["CNN_VS"] = scores["CNNscore"] * scores["CNNaffinity"]
    return scores.dropna(subset=["Compound", "UNIPROT_ID"])


def evaluate_sorting_method(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    score_column: str,
    ascending: bool,
) -> dict:
    """Calculate whether each compound retrieves its target within Top 100."""
    successes = 0
    compounds_with_scores = 0
    targets_with_scores = 0

    for row in targets.itertuples(index=False):
        compound_scores = scores[scores["Compound"] == row.Compound].dropna(
            subset=[score_column]
        )
        if compound_scores.empty:
            continue
        compounds_with_scores += 1
        ranked_proteins = (
            compound_scores.sort_values(score_column, ascending=ascending)
            .drop_duplicates("UNIPROT_ID", keep="first")
            .reset_index(drop=True)
        )
        if row.Target in set(ranked_proteins["UNIPROT_ID"]):
            targets_with_scores += 1
        top_ids = set(ranked_proteins.head(TOP_K)["UNIPROT_ID"])
        if row.Target in top_ids:
            successes += 1

    all_drug_count = len(targets)
    return {
        "method": score_column,
        "successful_drug_count": successes,
        "all_drug_count": all_drug_count,
        "compounds_with_scores": compounds_with_scores,
        "targets_present_in_scores": targets_with_scores,
        "success_rate_percent": (
            100 * successes / targets_with_scores if targets_with_scores else pd.NA
        ),
    }


def draw_comparison(result_table: pd.DataFrame, output_file: Path) -> None:
    plot_table = result_table.copy()
    palette = {
        "Original paper": "#8c8c8c",
        "Dataset 3\nsmina": "#4C78A8",
        "Dataset 3\nCNNaffinity": "#F58518",
        "Dataset 3\nCNN_VS": "#54A24B",
    }
    figure, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(
        data=plot_table,
        x="label",
        y="success_rate_percent",
        hue="label",
        palette=palette,
        dodge=False,
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{bar.get_height():.1f}%" for bar in container],
            padding=5,
            fontsize=15,
            fontweight="bold",
        )
    ax.set_title(
        "Top-100 target-retrieval success rate",
        fontsize=24,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Top-100 success rate (%)", fontsize=18)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelsize=15, rotation=0)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=600, bbox_inches="tight")
    figure.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 <= args.paper_success_rate <= 100:
        raise ValueError("--paper-success-rate must be between 0 and 100")
    targets = load_targets()
    scores = load_docking_scores(args.input)

    rows = [
        {
            "method": "original_paper",
            "label": "Original paper",
            "successful_drug_count": pd.NA,
            "all_drug_count": len(targets),
            "compounds_with_scores": pd.NA,
            "targets_present_in_scores": pd.NA,
            "success_rate_percent": args.paper_success_rate,
        }
    ]
    for score_column, settings in SORTING_METHODS.items():
        row = evaluate_sorting_method(
            scores,
            targets,
            score_column,
            settings["ascending"],
        )
        row["label"] = settings["label"]
        rows.append(row)

    result_table = pd.DataFrame(rows)
    result_table.to_csv(OUTPUT_FILE, index=False)
    draw_comparison(result_table, FIGURE_FILE)
    print(result_table.to_string(index=False))
    print(f"Comparison values saved to: {OUTPUT_FILE}")
    print(f"Comparison figure saved to: {FIGURE_FILE}")
    print(f"Comparison SVG saved to: {FIGURE_FILE.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
