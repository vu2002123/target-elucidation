import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"

FILE_LOCATIONS_FILE = INTERIM_DIR / "file_locations.csv"
UNIPROT_FILE = RAW_DIR / "uniprot_ids.tsv"
GENE_EFFECT_FILE = RAW_DIR / "CRISPRGeneEffect.csv"
GENE_DEPENDENCY_FILE = RAW_DIR / "CRISPRGeneDependency.csv"
MODEL_FILE = RAW_DIR / "Model.csv"
COMMON_ESSENTIAL_FILE = INTERIM_DIR / "CommonEssential_25q3p.csv"

# DRUG = "Niclosamide"
# ACTIVE_ANALOGUES = ["CCL-7293q", "CCL-7411k", "CCL-7284h", "CCL-7291o"]
# INACTIVE_ANALOGUES = ["CCL-7286j", "CCL-7414n", "CCL-7415o"]
#
# # Names used in the Compound column of the docking result files.
# ALTERNATIVE_NAMES = {
#     "Niclosamide": "Niclosamide",
#     "CCL-7293q": "7293q",
#     "CCL-7411k": "7411k",
#     "CCL-7284h": "7284h",
#     "CCL-7286j": "7286j",
#     "CCL-7291o": "7291o",
#     "CCL-7414n": "7414n",
#     "CCL-7415o": "7415o",
# }
#
# CANCER_TYPE = "CRC"

DRUG = "Prochlorperazine"
ACTIVE_ANALOGUES = ["N-desmethyl Prochlorperazine"]
INACTIVE_ANALOGUES = ["Prochlorperazine Sulfoxide"]

# Names used in the Compound column of the docking result files.
ALTERNATIVE_NAMES = {
    "Prochlorperazine": "PCP",
    "N-desmethyl Prochlorperazine": "NPCP",
    "Prochlorperazine Sulfoxide": "PCPS",
}

CANCER_TYPE = "LUAD"
TOP_N_VALUES = (100, 500)
OUTPUT_FILE = INTERIM_DIR / "top_target_scores_PCP_LUAD.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analogue-auroc",
        action="store_true",
        help=(
            "Score analogue agreement by active/inactive docking-percentile AUROC "
            "for the selected parent-drug dataset's top-500 proteins. "
            "By default, use rank agreement."
        ),
    )
    parser.add_argument(
        "--scoring-dataset",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help=(
            "Docking dataset used for docking and analogue-agreement scoring "
            "(default: 3). Ranks from all parent-drug datasets are still reported."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output target-score CSV (default: {OUTPUT_FILE}).",
    )
    return parser.parse_args()


def read_dock_scores(path: Path, extension: str, compound_name: str, dataset: int) -> pd.DataFrame:
    """Read and rank one docking result by the GNINA combined CNN_VS score."""
    separator = "\t" if extension.lower() == "tsv" or path.suffix.lower() == ".tsv" else ","
    dock_scores = pd.read_csv(path, sep=separator)
    dock_scores = dock_scores.rename(
        columns={
            "CNN_pose_score": "CNNscore",
            "CNN_affinity": "CNNaffinity",
            "affinity": "minimizedAffinity",
            "ID": "UNIPROT_ID",
        }
    )

    if "Compound" in dock_scores.columns:
        dock_scores = dock_scores[dock_scores["Compound"] == compound_name].copy()

    if "UNIPROT_ID" not in dock_scores.columns:
        if "File_Name" not in dock_scores.columns:
            raise ValueError(f"No UniProt ID or File_Name column in {path}")
        if dataset == 1:
            dock_scores["UNIPROT_ID"] = dock_scores["File_Name"].str.split("-").str[1]
        else:
            dock_scores["UNIPROT_ID"] = dock_scores["File_Name"].str.split("_").str[0]
    if "CNNaffinity" in dock_scores.columns:
        dock_scores["CNNaffinity"] = pd.to_numeric(dock_scores["CNNaffinity"], errors="coerce")
    if "CNN_VS" not in dock_scores.columns:
        if not {"CNNscore", "CNNaffinity"}.issubset(dock_scores.columns):
            raise ValueError(
                f"No CNN_VS column in {path}, and CNNscore/CNNaffinity are not "
                "both available to calculate it"
            )
        dock_scores["CNNscore"] = pd.to_numeric(dock_scores["CNNscore"], errors="coerce")
        dock_scores["CNN_VS"] = dock_scores["CNNscore"] * dock_scores["CNNaffinity"]
    else:
        dock_scores["CNN_VS"] = pd.to_numeric(dock_scores["CNN_VS"], errors="coerce")

    dock_scores["UNIPROT_ID"] = dock_scores["UNIPROT_ID"].astype("string").str.strip().str.upper()
    if "minimizedAffinity" in dock_scores.columns:
        dock_scores["minimizedAffinity"] = pd.to_numeric(
            dock_scores["minimizedAffinity"], errors="coerce"
        )
        dock_scores = dock_scores[dock_scores["minimizedAffinity"] < 0]

    dock_scores = (
        dock_scores.dropna(subset=["UNIPROT_ID", "CNN_VS"])
        .sort_values("CNN_VS", ascending=False)
        .drop_duplicates(subset="UNIPROT_ID")
        .reset_index(drop=True)
    )
    dock_scores["rank"] = np.arange(1, len(dock_scores) + 1)
    return dock_scores


def load_docking_results(
    drug: str, datasets: tuple[int, ...] = (1, 2, 3)
) -> dict[int, pd.DataFrame]:
    """Load the requested datasets for one compound using the file registry."""
    file_locations = pd.read_csv(FILE_LOCATIONS_FILE)
    docking_results = {}
    compound_name = ALTERNATIVE_NAMES.get(drug, drug)

    for dataset in datasets:
        locations = file_locations[
            (file_locations["Compound"] == drug) & (file_locations["Dataset"] == dataset)
        ]
        if locations.empty:
            raise FileNotFoundError(f"No docking-file entry for {drug}, dataset {dataset}")

        location = locations.iloc[0]
        dock_file = INTERIM_DIR / str(location["File_location"])
        if not dock_file.is_file():
            raise FileNotFoundError(dock_file)

        docking_results[dataset] = read_dock_scores(
            dock_file,
            str(location.get("Extension", dock_file.suffix.lstrip("."))),
            compound_name,
            dataset,
        )
        print(f"{drug}, dataset {dataset}: {len(docking_results[dataset]):,} ranked targets")

    return docking_results


def load_uniprot_gene_map() -> dict[str, str]:
    """Map each UniProt entry to the first gene name listed in uniprot_ids.tsv."""
    uniprot = pd.read_csv(UNIPROT_FILE, sep="\t", usecols=["Entry", "Gene Names"])
    uniprot["Entry"] = uniprot["Entry"].astype("string").str.strip().str.upper()
    uniprot["gene_name"] = (
        uniprot["Gene Names"]
        .astype("string")
        .str.strip()
        .str.split()
        .str[0]
        .str.replace(";", "", regex=False)
    )
    return (
        uniprot.dropna(subset=["Entry", "gene_name"])
        .drop_duplicates(subset="Entry", keep="first")
        .set_index("Entry")["gene_name"]
        .to_dict()
    )


def load_common_essential_genes() -> set[str]:
    """Return genes flagged TRUE in the DepMap common-essential gene file."""
    common_essential = pd.read_csv(COMMON_ESSENTIAL_FILE)
    if "symbol" not in common_essential.columns or len(common_essential.columns) < 2:
        raise ValueError(
            f"{COMMON_ESSENTIAL_FILE} must contain a 'symbol' column and a flag column"
        )

    flag_column = next(column for column in common_essential.columns if column != "symbol")
    flags = common_essential[flag_column]
    if not pd.api.types.is_bool_dtype(flags):
        flags = flags.astype("string").str.strip().str.upper().eq("TRUE")

    return set(
        common_essential.loc[flags.fillna(False), "symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )


def load_expression_evidence() -> pd.DataFrame:
    """Load differential expression and KM/COX significance for the cancer type."""
    deg = pd.read_csv(INTERIM_DIR / f"DEG_{CANCER_TYPE}_all.csv")
    deg = deg.dropna(subset=["Gene_name"]).copy()
    deg["Gene_name"] = deg["Gene_name"].astype(str).str.strip()
    deg["significantly_upregulated"] = (deg["log2FoldChange"] >= 1) & (deg["padj"] < 0.05)
    deg = deg.sort_values(
        ["significantly_upregulated", "padj", "log2FoldChange"],
        ascending=[False, True, False],
    ).drop_duplicates("Gene_name")[
        ["Gene_name", "log2FoldChange", "padj", "significantly_upregulated"]
    ]

    survival = pd.read_csv(INTERIM_DIR / f"survival_TCGA-{CANCER_TYPE}-all.csv")
    survival = survival.dropna(subset=["Gene_name"]).copy()
    survival["Gene_name"] = survival["Gene_name"].astype(str).str.strip()
    survival = survival.groupby("Gene_name", as_index=False).agg(
        km_p_value=("km_p_value", "min"),
        cox_p_value=("cox_p_value", "min"),
    )
    survival["km_significant"] = survival["km_p_value"] < 0.05
    survival["cox_significant"] = survival["cox_p_value"] < 0.05

    return deg.merge(survival, on="Gene_name", how="outer")


def select_depmap_columns(path: Path, genes: set[str]) -> tuple[str, dict[str, str]]:
    """Find the first DepMap column corresponding to each requested gene."""
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    index_column = columns[0]
    gene_columns = {}
    for column in columns[1:]:
        gene_name = column.split(" ", maxsplit=1)[0]
        if gene_name in genes and gene_name not in gene_columns:
            gene_columns[gene_name] = column
    return index_column, gene_columns


def load_dependency_evidence(genes: set[str]) -> pd.DataFrame:
    """Calculate dependency metrics across DepMap models matching CANCER_TYPE."""
    model_info = pd.read_csv(MODEL_FILE, usecols=["ModelID", "OncotreeCode"])
    depmap_cancer_types = (
        ("COAD", "READ", "COADREAD") if CANCER_TYPE.upper() == "CRC" else (CANCER_TYPE,)
    )
    model_ids = set(
        model_info.loc[
            model_info["OncotreeCode"].isin(depmap_cancer_types),
            "ModelID",
        ]
    )
    if not model_ids:
        raise ValueError(
            "No DepMap models found for OncotreeCode(s) " + ", ".join(depmap_cancer_types)
        )

    effect_index, effect_columns = select_depmap_columns(GENE_EFFECT_FILE, genes)
    dependency_index, dependency_columns = select_depmap_columns(GENE_DEPENDENCY_FILE, genes)

    effect = pd.read_csv(
        GENE_EFFECT_FILE,
        usecols=[effect_index, *effect_columns.values()],
        index_col=0,
    )
    dependency = pd.read_csv(
        GENE_DEPENDENCY_FILE,
        usecols=[dependency_index, *dependency_columns.values()],
        index_col=0,
    )
    cohort_ids = effect.index.intersection(dependency.index).intersection(list(model_ids))
    if cohort_ids.empty:
        raise ValueError(f"No overlapping DepMap rows for {CANCER_TYPE}")

    rows = []
    for gene in sorted(genes):
        effect_column = effect_columns.get(gene)
        dependency_column = dependency_columns.get(gene)
        if effect_column is None or dependency_column is None:
            rows.append({"Gene_name": gene})
            continue

        gene_effect = effect.loc[cohort_ids, effect_column]
        gene_dependency = dependency.loc[cohort_ids, dependency_column]
        valid_gene_dependency = gene_dependency.dropna()
        rows.append(
            {
                "Gene_name": gene,
                "median_gene_dependency": gene_dependency.median(skipna=True),
                "median_gene_effect": gene_effect.median(skipna=True),
                "dependent_cell_percent": (
                    100 * (valid_gene_dependency > 0.5).mean()
                    if not valid_gene_dependency.empty
                    else np.nan
                ),
                "depmap_model_count": len(valid_gene_dependency),
            }
        )

    return pd.DataFrame(rows)


def expression_score(row: pd.Series) -> int:
    if not bool(row.get("significantly_upregulated", False)):
        return 0
    km_significant = bool(row.get("km_significant", False))
    cox_significant = bool(row.get("cox_significant", False))
    if km_significant or cox_significant:
        return 2
    return 1


def dependency_score(row: pd.Series) -> int:
    median_effect = row.get("median_gene_effect")

    if pd.isna(median_effect):
        return 0
    if median_effect <= -0.5:
        return 2
    if -0.5 < median_effect < 0:
        return 1
    return 0


def compound_column_name(compound: str) -> str:
    """Create a readable, CSV-safe prefix for compound-specific columns."""
    return compound.lower().replace("-", "_").replace(" ", "_")


def calculate_analogue_agreement_score(
    targets: pd.DataFrame,
    active_rank_columns: list[str],
    inactive_rank_columns: list[str],
    parent_rank_column: str = "dataset_3_rank",
) -> pd.Series:
    """Score whether active analogues rank ahead of inactive analogues.

    A smaller numeric rank is better. Unranked active analogues therefore fail
    the top-500 requirement, while unranked inactive analogues are treated as
    lower-ranked than any ranked active analogue or parent compound.
    """
    if not active_rank_columns:
        raise ValueError("At least one active analogue is required for agreement scoring")

    active_ranks = targets[active_rank_columns]
    every_active_top_500 = active_ranks.notna().all(axis=1) & active_ranks.le(500).all(axis=1)

    parent_rank = targets[parent_rank_column]
    if inactive_rank_columns:
        worst_comparator_rank = pd.concat(
            [active_ranks, parent_rank.rename(parent_rank_column)],
            axis=1,
        ).max(axis=1)
        every_inactive_ranks_lower = (
            targets[inactive_rank_columns]
            .fillna(np.inf)
            .gt(worst_comparator_rank, axis=0)
            .all(axis=1)
        )
    else:
        every_inactive_ranks_lower = pd.Series(True, index=targets.index)

    score_two = every_active_top_500 & parent_rank.notna() & every_inactive_ranks_lower
    return pd.Series(
        np.select(
            [score_two, every_active_top_500],
            [2, 1],
            default=0,
        ),
        index=targets.index,
        dtype=int,
    )


def binary_auroc(labels: np.ndarray, predictors: np.ndarray) -> float:
    """Calculate AUROC without requiring scikit-learn; higher predicts active."""
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return np.nan

    ranks = pd.Series(predictors).rank(method="average").to_numpy()
    positive_rank_sum = ranks[labels.astype(bool)].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def calculate_analogue_auroc_score(
    targets: pd.DataFrame,
    active_results: dict[str, pd.DataFrame],
    inactive_results: dict[str, pd.DataFrame],
    scoring_dataset: int = 3,
) -> pd.Series:
    """Add analogue percentiles/AUROC and return AUROC-based agreement scores."""
    percentile_columns = []
    labels = []
    for is_active, results in ((True, active_results), (False, inactive_results)):
        for analogue, scores in results.items():
            prefix = compound_column_name(analogue)
            percentile_column = f"{prefix}_docking_percentile"
            percentiles = 1 - (scores["rank"] - 1) / len(scores)
            percentile_map = pd.Series(
                percentiles.to_numpy(),
                index=scores["UNIPROT_ID"],
            )
            targets[percentile_column] = targets["uniprot_id"].map(percentile_map).fillna(0.0)
            percentile_columns.append(percentile_column)
            labels.append(is_active)

    label_values = np.asarray(labels, dtype=int)
    parent_top_500 = targets[f"dataset_{scoring_dataset}_top_500"]
    targets["analogue_auroc"] = np.nan
    targets.loc[parent_top_500, "analogue_auroc"] = targets.loc[
        parent_top_500, percentile_columns
    ].apply(
        lambda row: binary_auroc(label_values, row.to_numpy(dtype=float)),
        axis=1,
    )
    return pd.Series(
        np.select(
            [
                parent_top_500 & targets["analogue_auroc"].gt(0.75),
                parent_top_500 & targets["analogue_auroc"].between(0.58, 0.75, inclusive="both"),
            ],
            [2, 1],
            default=0,
        ),
        index=targets.index,
        dtype=int,
    )


def main(
    use_analogue_auroc: bool = False,
    scoring_dataset: int = 3,
    output_file: Path = OUTPUT_FILE,
) -> None:
    docking_results = load_docking_results(DRUG)
    active_analogue_results = {
        analogue: load_docking_results(analogue, datasets=(scoring_dataset,))[
            scoring_dataset
        ]
        for analogue in ACTIVE_ANALOGUES
    }
    inactive_analogue_results = {
        analogue: load_docking_results(analogue, datasets=(scoring_dataset,))[
            scoring_dataset
        ]
        for analogue in INACTIVE_ANALOGUES
    }
    uniprot_gene_map = load_uniprot_gene_map()

    top_500_ids = {
        dataset: set(scores.head(500)["UNIPROT_ID"]) for dataset, scores in docking_results.items()
    }
    candidate_ids = set().union(*top_500_ids.values())

    targets = pd.DataFrame({"uniprot_id": sorted(candidate_ids)})
    targets["gene_name"] = targets["uniprot_id"].map(uniprot_gene_map)
    common_essential_genes = load_common_essential_genes()
    targets["common_essential"] = (
        targets["gene_name"].astype("string").str.strip().str.upper().isin(common_essential_genes)
    )

    for dataset, scores in docking_results.items():
        rank_map = scores.set_index("UNIPROT_ID")["rank"]
        targets[f"dataset_{dataset}_rank"] = targets["uniprot_id"].map(rank_map)

        if dataset == scoring_dataset:
            score_index = scores.set_index("UNIPROT_ID")
            targets[f"dataset_{dataset}_CNN_VS"] = targets["uniprot_id"].map(
                score_index["CNN_VS"]
            )
            if "CNNaffinity" in scores.columns:
                targets[f"dataset_{dataset}_CNNaffinity"] = targets["uniprot_id"].map(
                    score_index["CNNaffinity"]
                )
            for top_n in TOP_N_VALUES:
                targets[f"dataset_{dataset}_top_{top_n}"] = (
                    targets[f"dataset_{dataset}_rank"].le(top_n).fillna(False)
                )

    targets["docking_score"] = np.select(
        [
            targets[f"dataset_{scoring_dataset}_top_100"],
            targets[f"dataset_{scoring_dataset}_top_500"],
        ],
        [2, 1],
        default=0,
    )

    active_hit_columns = {}
    active_rank_columns = []
    for analogue, scores in active_analogue_results.items():
        prefix = compound_column_name(analogue)
        rank_column = f"{prefix}_dataset_{scoring_dataset}_rank"
        hit_column = f"{prefix}_dataset_{scoring_dataset}_top_500"
        targets[rank_column] = targets["uniprot_id"].map(scores.set_index("UNIPROT_ID")["rank"])
        targets[hit_column] = targets[rank_column].le(500).fillna(False)
        active_hit_columns[hit_column] = analogue
        active_rank_columns.append(rank_column)

    inactive_hit_columns = {}
    inactive_rank_columns = []
    for analogue, scores in inactive_analogue_results.items():
        prefix = compound_column_name(analogue)
        rank_column = f"{prefix}_dataset_{scoring_dataset}_rank"
        hit_column = f"{prefix}_dataset_{scoring_dataset}_top_500"
        targets[rank_column] = targets["uniprot_id"].map(scores.set_index("UNIPROT_ID")["rank"])
        targets[hit_column] = targets[rank_column].le(500).fillna(False)
        inactive_hit_columns[hit_column] = analogue
        inactive_rank_columns.append(rank_column)

    targets["active_analogue_hits"] = targets[list(active_hit_columns)].apply(
        lambda row: "; ".join(
            active_hit_columns[column] for column, is_hit in row.items() if is_hit
        ),
        axis=1,
    )
    targets["inactive_analogue_hits"] = targets[list(inactive_hit_columns)].apply(
        lambda row: "; ".join(
            inactive_hit_columns[column] for column, is_hit in row.items() if is_hit
        ),
        axis=1,
    )
    if use_analogue_auroc:
        targets["analogue_agreement_score"] = calculate_analogue_auroc_score(
            targets,
            active_analogue_results,
            inactive_analogue_results,
            scoring_dataset=scoring_dataset,
        )
    else:
        targets["analogue_agreement_score"] = calculate_analogue_agreement_score(
            targets,
            active_rank_columns,
            inactive_rank_columns,
            parent_rank_column=f"dataset_{scoring_dataset}_rank",
        )

    expression = load_expression_evidence().rename(columns={"Gene_name": "gene_name"})
    targets = targets.merge(expression, on="gene_name", how="left")
    for column in ["significantly_upregulated", "km_significant", "cox_significant"]:
        targets[column] = targets[column].astype("boolean").fillna(False).astype(bool)
    targets["expression_score"] = targets.apply(expression_score, axis=1)

    mapped_genes = set(targets["gene_name"].dropna())
    dependency = load_dependency_evidence(mapped_genes).rename(columns={"Gene_name": "gene_name"})
    targets = targets.merge(dependency, on="gene_name", how="left")
    targets["dependency_score"] = targets.apply(dependency_score, axis=1)

    targets["total_score"] = (
        targets["docking_score"]
        + targets["analogue_agreement_score"]
        + targets["expression_score"]
        + targets["dependency_score"]
    )
    targets.insert(0, "drug", DRUG)
    targets.insert(1, "cancer_type", CANCER_TYPE)
    targets.insert(2, "scoring_dataset", scoring_dataset)
    leading_columns = [
        "drug",
        "cancer_type",
        "scoring_dataset",
        "total_score",
        "docking_score",
        "analogue_agreement_score",
        *(["analogue_auroc"] if use_analogue_auroc else []),
        "expression_score",
        "dependency_score",
        "common_essential",
    ]
    targets = targets[
        leading_columns + [column for column in targets.columns if column not in leading_columns]
    ]
    targets = targets.sort_values(
        [
            "total_score",
            "docking_score",
            "analogue_agreement_score",
            f"dataset_{scoring_dataset}_rank",
        ],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_file, index=False)
    print(f"Scored {len(targets):,} unique targets")
    print(f"Targets without a mapped gene name: {targets['gene_name'].isna().sum():,}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(
        use_analogue_auroc=args.analogue_auroc,
        scoring_dataset=args.scoring_dataset,
        output_file=args.output,
    )
