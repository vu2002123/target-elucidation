"""Rank parent-drug targets using docking, analogue AUROC, and active-cell expression.

The analogue AUROC is calculated separately for every parent-drug top-500
target.  Active analogues are the positive class, inactive analogues are the
negative class, and each analogue's docking-rank percentile is the predictor.
"""

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
MODEL_FILE = RAW_DIR / "Model.csv"
EXPRESSION_FILE = RAW_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
OUTPUT_FILE = INTERIM_DIR / "top_target_scores_NEN_2.csv"

DRUG = "Niclosamide"
ACTIVE_ANALOGUES = ["CCL-7293q", "CCL-7411k", "CCL-7284h", "CCL-7291o"]
INACTIVE_ANALOGUES = ["CCL-7286j", "CCL-7414n", "CCL-7415o"]

# Names used in the Compound column of the docking result files.
ALTERNATIVE_NAMES = {
    "Niclosamide": "Niclosamide",
    "CCL-7293q": "7293q",
    "CCL-7411k": "7411k",
    "CCL-7284h": "7284h",
    "CCL-7286j": "7286j",
    "CCL-7291o": "7291o",
    "CCL-7414n": "7414n",
    "CCL-7415o": "7415o",
}

CANCER_TYPE = "CRC"
ACTIVE_CELL_LINES = ["SW480", "DLD1", "RKO"]
TOP_N_VALUES = (100, 500)
EXPRESSION_THRESHOLD_TPM = 1.0


def read_dock_scores(path: Path, extension: str, compound_name: str, dataset: int) -> pd.DataFrame:
    """Read and rank one docking result by the GNINA combined CNN_VS score."""
    separator = "\t" if extension.lower() == "tsv" or path.suffix.lower() == ".tsv" else ","
    dock_scores = pd.read_csv(path, sep=separator).rename(
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
        separator_index = 1 if dataset == 1 else 0
        delimiter = "-" if dataset == 1 else "_"
        dock_scores["UNIPROT_ID"] = (
            dock_scores["File_Name"].str.split(delimiter).str[separator_index]
        )

    if "CNNaffinity" in dock_scores.columns:
        dock_scores["CNNaffinity"] = pd.to_numeric(dock_scores["CNNaffinity"], errors="coerce")
    if "CNN_VS" not in dock_scores.columns:
        if not {"CNNscore", "CNNaffinity"}.issubset(dock_scores.columns):
            raise ValueError(f"Cannot calculate CNN_VS for {path}")
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
    dock_scores["docking_percentile"] = 1 - (dock_scores["rank"] - 1) / len(dock_scores)
    return dock_scores


def load_docking_results(
    drug: str, datasets: tuple[int, ...] = (1, 2, 3)
) -> dict[int, pd.DataFrame]:
    """Load the requested docking datasets for one compound."""
    file_locations = pd.read_csv(FILE_LOCATIONS_FILE)
    results = {}
    compound_name = ALTERNATIVE_NAMES.get(drug, drug)
    for dataset in datasets:
        locations = file_locations[
            (file_locations["Compound"] == drug) & (file_locations["Dataset"] == dataset)
        ]
        if locations.empty:
            raise FileNotFoundError(f"No docking-file entry for {drug}, dataset {dataset}")
        location = locations.iloc[0]
        path = INTERIM_DIR / str(location["File_location"])
        if not path.is_file():
            raise FileNotFoundError(path)
        results[dataset] = read_dock_scores(
            path, str(location.get("Extension", path.suffix.lstrip("."))), compound_name, dataset
        )
        print(f"{drug}, dataset {dataset}: {len(results[dataset]):,} ranked targets")
    return results


def load_uniprot_gene_map() -> dict[str, str]:
    """Map each UniProt accession to its first listed gene symbol."""
    uniprot = pd.read_csv(UNIPROT_FILE, sep="\t", usecols=["Entry", "Gene Names"])
    uniprot["Entry"] = uniprot["Entry"].astype("string").str.strip().str.upper()
    uniprot["gene_name"] = (
        uniprot["Gene Names"].astype("string").str.strip().str.split().str[0].str.replace(";", "")
    )
    return (
        uniprot.dropna(subset=["Entry", "gene_name"])
        .drop_duplicates(subset="Entry", keep="first")
        .set_index("Entry")["gene_name"]
        .to_dict()
    )


def compound_column_name(compound: str) -> str:
    return compound.lower().replace("-", "_").replace(" ", "_")


def binary_auroc(labels: np.ndarray, predictors: np.ndarray) -> float:
    """Calculate AUROC without a scikit-learn dependency; higher predicts active."""
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


def add_analogue_auroc(
    targets: pd.DataFrame,
    active_results: dict[str, pd.DataFrame],
    inactive_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Add per-target analogue docking percentiles, AUROCs, and AUROC scores."""
    percentile_columns = []
    labels = []
    for is_active, results in ((True, active_results), (False, inactive_results)):
        for analogue, scores in results.items():
            prefix = compound_column_name(analogue)
            rank_column = f"{prefix}_dataset_3_rank"
            percentile_column = f"{prefix}_docking_percentile"
            score_index = scores.set_index("UNIPROT_ID")
            targets[rank_column] = targets["uniprot_id"].map(score_index["rank"])
            # A target omitted from a ranked result is assigned percentile zero.
            targets[percentile_column] = (
                targets["uniprot_id"].map(score_index["docking_percentile"]).fillna(0.0)
            )
            percentile_columns.append(percentile_column)
            labels.append(is_active)

    label_values = np.asarray(labels, dtype=int)
    targets["analogue_auroc"] = targets[percentile_columns].apply(
        lambda row: binary_auroc(label_values, row.to_numpy(dtype=float)), axis=1
    )
    targets["analogue_agreement_score"] = np.select(
        [
            targets["analogue_auroc"] > 0.75,
            targets["analogue_auroc"].between(0.58, 0.75, inclusive="both"),
        ],
        [2, 1],
        default=0,
    )
    return targets


def select_gene_columns(path: Path, genes: set[str]) -> tuple[str, dict[str, str]]:
    """Find one full data-column name for every requested gene symbol."""
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    gene_columns = {}
    for column in columns:
        gene_name = column.split(" ", maxsplit=1)[0]
        if gene_name in genes and gene_name not in gene_columns:
            gene_columns[gene_name] = column
    return columns[0], gene_columns


def resolve_active_cell_lines() -> pd.DataFrame:
    """Resolve configured active cell-line labels to unique DepMap ModelIDs."""
    model_info = pd.read_csv(
        MODEL_FILE, usecols=["ModelID", "CellLineName", "StrippedCellLineName", "CCLEName"]
    )
    resolved = []
    for requested_name in ACTIVE_CELL_LINES:
        normalized_name = "".join(char for char in requested_name.upper() if char.isalnum())
        matches = pd.Series(False, index=model_info.index)
        for column in ["CellLineName", "StrippedCellLineName", "CCLEName"]:
            normalized_column = (
                model_info[column]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.replace(r"[^A-Z0-9]", "", regex=True)
            )
            matches |= normalized_column.eq(normalized_name)
        matching_models = model_info.loc[matches, ["ModelID", "CellLineName"]]
        if len(matching_models) != 1:
            raise ValueError(
                f"Expected one ModelID for {requested_name}; found {len(matching_models)}."
            )
        resolved.append(
            {
                "requested_name": requested_name,
                "model_id": matching_models.iloc[0]["ModelID"],
                "cell_line_name": matching_models.iloc[0]["CellLineName"],
            }
        )
    return pd.DataFrame(resolved)


def load_active_cell_evidence(genes: set[str]) -> pd.DataFrame:
    """Return TPM and gene-effect values for every target gene in active cell lines."""
    active_cells = resolve_active_cell_lines()
    expression_index, expression_columns = select_gene_columns(EXPRESSION_FILE, genes)
    effect_index, effect_columns = select_gene_columns(GENE_EFFECT_FILE, genes)
    expression = pd.read_csv(
        EXPRESSION_FILE,
        usecols=["ModelID", "IsDefaultEntryForModel", *expression_columns.values()],
    )
    expression = (
        expression[expression["IsDefaultEntryForModel"].eq("Yes")]
        .drop(columns="IsDefaultEntryForModel")
        .set_index("ModelID")
    )
    gene_effect = pd.read_csv(
        GENE_EFFECT_FILE,
        usecols=[effect_index, *effect_columns.values()],
        index_col=effect_index,
    )

    rows = []
    for gene in sorted(genes):
        row = {"gene_name": gene}
        expressed_count = 0
        for cell in active_cells.itertuples(index=False):
            expression_column = expression_columns.get(gene)
            effect_column = effect_columns.get(gene)
            log_tpm = (
                expression.at[cell.model_id, expression_column]
                if expression_column and cell.model_id in expression.index
                else np.nan
            )
            # The source is log2(TPM + 1), so report the requested TPM scale.
            tpm = 2**log_tpm - 1 if pd.notna(log_tpm) else np.nan
            effect = (
                gene_effect.at[cell.model_id, effect_column]
                if effect_column and cell.model_id in gene_effect.index
                else np.nan
            )
            row[f"{cell.requested_name}_expression_tpm"] = tpm
            row[f"{cell.requested_name}_expressed"] = bool(tpm >= EXPRESSION_THRESHOLD_TPM)
            row[f"{cell.requested_name}_gene_effect"] = effect
            expressed_count += int(tpm >= EXPRESSION_THRESHOLD_TPM) if pd.notna(tpm) else 0
        row["expressed_active_cell_line_count"] = expressed_count
        row["expression_score"] = (
            2
            if expressed_count == len(active_cells)
            else 1
            if expressed_count == len(active_cells) - 1
            else 0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    docking_results = load_docking_results(DRUG)
    active_results = {
        analogue: load_docking_results(analogue, datasets=(3,))[3] for analogue in ACTIVE_ANALOGUES
    }
    inactive_results = {
        analogue: load_docking_results(analogue, datasets=(3,))[3]
        for analogue in INACTIVE_ANALOGUES
    }

    # Both docking score and analogue AUROC are defined for parent dataset-3 top-500 targets.
    parent_scores = docking_results[3]
    targets = parent_scores.head(500)[["UNIPROT_ID", "rank", "CNN_VS"]].rename(
        columns={
            "UNIPROT_ID": "uniprot_id",
            "rank": "dataset_3_rank",
            "CNN_VS": "dataset_3_CNN_VS",
        }
    )
    if "CNNaffinity" in parent_scores.columns:
        targets["dataset_3_CNNaffinity"] = parent_scores.head(500)["CNNaffinity"].to_numpy()
    targets["gene_name"] = targets["uniprot_id"].map(load_uniprot_gene_map())
    for dataset, scores in docking_results.items():
        if dataset != 3:
            targets[f"dataset_{dataset}_rank"] = targets["uniprot_id"].map(
                scores.set_index("UNIPROT_ID")["rank"]
            )
    for top_n in TOP_N_VALUES:
        targets[f"dataset_3_top_{top_n}"] = targets["dataset_3_rank"].le(top_n)
    targets["docking_score"] = np.select(
        [targets["dataset_3_top_100"], targets["dataset_3_top_500"]], [2, 1], default=0
    )

    targets = add_analogue_auroc(targets, active_results, inactive_results)
    mapped_genes = set(targets["gene_name"].dropna())
    expression = load_active_cell_evidence(mapped_genes)
    targets = targets.merge(expression, on="gene_name", how="left")
    targets["expression_score"] = targets["expression_score"].fillna(0).astype(int)
    targets["total_score"] = (
        targets["docking_score"]
        + targets["analogue_agreement_score"]
        + targets["expression_score"]
    )
    targets.insert(0, "drug", DRUG)
    targets.insert(1, "cancer_type", CANCER_TYPE)
    leading_columns = [
        "drug",
        "cancer_type",
        "total_score",
        "docking_score",
        "analogue_agreement_score",
        "analogue_auroc",
        "expression_score",
        "expressed_active_cell_line_count",
    ]
    targets = targets[
        leading_columns + [column for column in targets.columns if column not in leading_columns]
    ]
    targets = targets.sort_values(
        ["total_score", "docking_score", "analogue_agreement_score", "dataset_3_rank"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    targets.to_csv(OUTPUT_FILE, index=False)
    print(f"Scored {len(targets):,} parent-drug top-500 targets")
    print(f"Targets without a mapped gene name: {targets['gene_name'].isna().sum():,}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
