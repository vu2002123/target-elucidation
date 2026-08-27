from pathlib import Path

import polars as pl


# =============================================================================
# Configuration
# =============================================================================

TARGET_DIR = Path("/home/vu2002123/target-elucidation/data/external/target")

OUTPUT_FILE = Path(
    "/home/vu2002123/target-elucidation/data/external/"
    "chembl_target_classes_uniprot_tractability.csv"
)

LEVEL_1_CLASSES = [
    "Enzyme",
    "Membrane receptor",
    "Ion channel",
    "Transcription factor",
    "Transporter",
]

ALLOWED_PROTEIN_SOURCES = [
    "uniprot_swissprot",
    "uniprot",
]


# Lower bucket number means stronger small-molecule tractability evidence.
SM_TRACTABILITY_LEVELS = [
    (1, "Approved Drug"),
    (2, "Advanced Clinical"),
    (3, "Phase 1 Clinical"),
    (4, "Structure with Ligand"),
    (5, "High-Quality Ligand"),
    (6, "High-Quality Pocket"),
    (7, "Med-Quality Pocket"),
    (8, "Druggable Family"),
]


# =============================================================================
# Locate and read Open Targets Parquet files
# =============================================================================

parquet_files = sorted(TARGET_DIR.rglob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError(f"No Parquet files were found under: {TARGET_DIR}")

print(f"Found {len(parquet_files):,} Parquet files.")

targets = pl.scan_parquet([str(path) for path in parquet_files])


# =============================================================================
# Validate required top-level columns
# =============================================================================

required_columns = {
    "id",
    "approvedSymbol",
    "approvedName",
    "targetClass",
    "proteinIds",
    "tractability",
}

available_columns = set(targets.collect_schema().names())

missing_columns = required_columns - available_columns

if missing_columns:
    raise ValueError(
        f"The target Parquet dataset is missing required columns: {sorted(missing_columns)}"
    )


print("\nTractability column schema:")
print(targets.collect_schema()["tractability"])


# =============================================================================
# Convert targetClass into a long table
#
# Expected records:
#
# {"label": "Enzyme", "level": "l1"}
# {"label": "Kinase", "level": "l2"}
# {"label": "Protein kinase", "level": "l3"}
# =============================================================================

target_classes = (
    targets.select(
        "id",
        "targetClass",
    )
    .explode(
        "targetClass",
        empty_as_null=False,
        keep_nulls=False,
    )
    .with_columns(
        pl.col("targetClass")
        .struct.field("label")
        .cast(pl.String)
        .str.strip_chars()
        .alias("chembl_class"),
        pl.col("targetClass")
        .struct.field("level")
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .alias("chembl_class_level"),
    )
    .select(
        "id",
        "chembl_class",
        "chembl_class_level",
    )
    .drop_nulls(
        [
            "id",
            "chembl_class",
            "chembl_class_level",
        ]
    )
    .filter(pl.col("chembl_class") != "")
)


# =============================================================================
# Extract selected level-1 classes
#
# A target with more than one selected level-1 class will produce one output
# row per target/level-1-class combination.
# =============================================================================

level_1_table = (
    target_classes.filter(
        (pl.col("chembl_class_level") == "l1") & pl.col("chembl_class").is_in(LEVEL_1_CLASSES)
    )
    .select(
        "id",
        pl.col("chembl_class").alias("level_1_class"),
    )
    .unique()
)


# =============================================================================
# Extract level-2 classes
#
# Multiple level-2 labels for the same target are combined with semicolons.
# =============================================================================

level_2_table = (
    target_classes.filter(pl.col("chembl_class_level") == "l2")
    .group_by("id")
    .agg(pl.col("chembl_class").drop_nulls().unique().sort().str.join("; ").alias("level_2_class"))
)


# =============================================================================
# Extract level-3 classes
# =============================================================================

level_3_table = (
    target_classes.filter(pl.col("chembl_class_level") == "l3")
    .group_by("id")
    .agg(pl.col("chembl_class").drop_nulls().unique().sort().str.join("; ").alias("level_3_class"))
)


# =============================================================================
# Combine ChEMBL class levels
# =============================================================================

class_summary = level_1_table.join(
    level_2_table,
    on="id",
    how="left",
).join(
    level_3_table,
    on="id",
    how="left",
)


# =============================================================================
# Build the small-molecule tractability lookup
# =============================================================================

sm_tractability_lookup = pl.DataFrame(
    {
        "highest_small_molecule_tractability": [label for _, label in SM_TRACTABILITY_LEVELS],
        "small_molecule_tractability_bucket": [bucket for bucket, _ in SM_TRACTABILITY_LEVELS],
    }
).lazy()


# =============================================================================
# Convert the nested tractability column into a long table
#
# Actual structure observed:
#
# {
#     "modality": "SM",
#     "id": "Druggable Family",
#     "value": True,
# }
#
# The "id" field is the readable tractability label.
# =============================================================================

tractability_long = (
    targets.select(
        "id",
        "tractability",
    )
    .explode(
        "tractability",
        empty_as_null=False,
        keep_nulls=False,
    )
    .with_columns(
        pl.col("tractability")
        .struct.field("modality")
        .cast(pl.String)
        .str.strip_chars()
        .str.to_uppercase()
        .alias("tractability_modality"),
        pl.col("tractability")
        .struct.field("id")
        .cast(pl.String)
        .str.strip_chars()
        .alias("tractability_label"),
        pl.col("tractability")
        .struct.field("value")
        .cast(pl.Boolean, strict=False)
        .fill_null(False)
        .alias("tractability_value"),
    )
    .select(
        "id",
        "tractability_modality",
        "tractability_label",
        "tractability_value",
    )
    .drop_nulls(
        [
            "id",
            "tractability_modality",
            "tractability_label",
        ]
    )
)


# =============================================================================
# Display available small-molecule tractability labels
# =============================================================================

available_sm_tractability = (
    tractability_long.filter(pl.col("tractability_modality") == "SM")
    .select(
        "tractability_label",
        "tractability_value",
    )
    .unique()
    .sort(
        [
            "tractability_label",
            "tractability_value",
        ]
    )
    .collect()
)

print("\nAvailable small-molecule tractability values:")
print(available_sm_tractability)


# =============================================================================
# Keep positive SM tractability assessments and assign bucket numbers
# =============================================================================

positive_sm_tractability = (
    tractability_long.filter(
        (pl.col("tractability_modality") == "SM") & pl.col("tractability_value")
    )
    .join(
        sm_tractability_lookup,
        left_on="tractability_label",
        right_on="highest_small_molecule_tractability",
        how="inner",
    )
    .select(
        "id",
        "tractability_label",
        "small_molecule_tractability_bucket",
    )
    .unique()
)


# =============================================================================
# Select the strongest positive SM tractability level per target
#
# Example:
#
# Approved Drug=True and Druggable Family=True
#     -> choose Approved Drug because bucket 1 is stronger than bucket 8.
# =============================================================================

highest_sm_bucket = positive_sm_tractability.group_by("id").agg(
    pl.col("small_molecule_tractability_bucket").min().alias("small_molecule_tractability_bucket")
)


highest_sm_tractability = (
    highest_sm_bucket.join(
        sm_tractability_lookup,
        on="small_molecule_tractability_bucket",
        how="left",
    )
    .with_columns(
        pl.concat_str(
            [
                pl.lit("SM"),
                pl.col("small_molecule_tractability_bucket").cast(pl.String),
            ]
        ).alias("highest_small_molecule_tractability_id")
    )
    .select(
        "id",
        "highest_small_molecule_tractability_id",
        "small_molecule_tractability_bucket",
        "highest_small_molecule_tractability",
    )
)


# =============================================================================
# Extract accepted UniProt identifiers
#
# Both of these source values are allowed:
#
#   uniprot_swissprot
#   uniprot
#
# If the same target/accession occurs under both source names,
# uniprot_swissprot is preferred.
# =============================================================================

uniprot_proteins = (
    targets.select(
        "id",
        "approvedSymbol",
        "approvedName",
        "proteinIds",
    )
    .explode(
        "proteinIds",
        empty_as_null=False,
        keep_nulls=False,
    )
    .with_columns(
        pl.col("proteinIds")
        .struct.field("id")
        .cast(pl.String)
        .str.strip_chars()
        .alias("uniprot_id"),
        pl.col("proteinIds")
        .struct.field("source")
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .alias("protein_id_source"),
    )
    .filter(pl.col("protein_id_source").is_in(ALLOWED_PROTEIN_SOURCES))
    .with_columns(
        pl.when(pl.col("protein_id_source") == "uniprot_swissprot")
        .then(pl.lit(1))
        .otherwise(pl.lit(2))
        .alias("protein_source_priority")
    )
    .select(
        "id",
        "approvedSymbol",
        "approvedName",
        "uniprot_id",
        "protein_id_source",
        "protein_source_priority",
    )
    .drop_nulls(
        [
            "id",
            "uniprot_id",
        ]
    )
    .filter(pl.col("uniprot_id") != "")
    .sort(
        [
            "id",
            "uniprot_id",
            "protein_source_priority",
        ]
    )
    .unique(
        subset=[
            "id",
            "uniprot_id",
        ],
        keep="first",
        maintain_order=True,
    )
    .drop("protein_source_priority")
)


# =============================================================================
# Diagnostics before building final table
# =============================================================================

level_1_count = level_1_table.select(pl.len()).collect().item()

level_2_count = level_2_table.select(pl.len()).collect().item()

level_3_count = level_3_table.select(pl.len()).collect().item()

uniprot_count = uniprot_proteins.select(pl.len()).collect().item()

positive_sm_count = highest_sm_tractability.select(pl.len()).collect().item()


print(f"\nSelected level-1 classification rows: {level_1_count:,}")
print(f"Targets with level-2 classifications: {level_2_count:,}")
print(f"Targets with level-3 classifications: {level_3_count:,}")
print(f"Accepted UniProt identifier rows: {uniprot_count:,}")

print(f"Targets with at least one positive SM tractability assessment: {positive_sm_count:,}")


# =============================================================================
# Stop early if an important filter produced no rows
# =============================================================================

if level_1_count == 0:
    available_level_1_classes = (
        target_classes.filter(pl.col("chembl_class_level") == "l1")
        .select("chembl_class")
        .unique()
        .sort("chembl_class")
        .collect()
    )

    print("\nAvailable ChEMBL level-1 classes:")
    print(available_level_1_classes)

    raise RuntimeError("None of the requested LEVEL_1_CLASSES were found.")


if uniprot_count == 0:
    available_sources = (
        targets.select("proteinIds")
        .explode(
            "proteinIds",
            empty_as_null=False,
            keep_nulls=False,
        )
        .select(
            pl.col("proteinIds")
            .struct.field("source")
            .cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase()
            .alias("source")
        )
        .drop_nulls()
        .unique()
        .sort("source")
        .collect()
    )

    print("\nAvailable protein-ID sources:")
    print(available_sources)

    raise RuntimeError("No protein IDs matched the allowed UniProt sources.")


if positive_sm_count == 0:
    positive_labels = (
        tractability_long.filter(
            (pl.col("tractability_modality") == "SM") & pl.col("tractability_value")
        )
        .select("tractability_label")
        .unique()
        .sort("tractability_label")
        .collect()
    )

    print("\nPositive SM labels found in the Parquet dataset:")
    print(positive_labels)

    print("\nExpected SM labels:")
    print(pl.DataFrame({"expected_label": [label for _, label in SM_TRACTABILITY_LEVELS]}))

    raise RuntimeError(
        "Positive SM assessments were present, but none matched "
        "the configured tractability labels."
    )


# =============================================================================
# Build the final lazy query
# =============================================================================

protein_families_query = (
    uniprot_proteins.join(
        class_summary,
        on="id",
        how="inner",
    )
    .join(
        highest_sm_tractability,
        on="id",
        how="left",
    )
    .with_columns(
        pl.col("highest_small_molecule_tractability_id").fill_null("None"),
        pl.col("highest_small_molecule_tractability").fill_null("No positive SM assessment"),
        pl.col("small_molecule_tractability_bucket").cast(pl.Int8),
    )
    .select(
        "uniprot_id",
        "approvedSymbol",
        "approvedName",
        "level_1_class",
        "level_2_class",
        "level_3_class",
        "highest_small_molecule_tractability_id",
        "small_molecule_tractability_bucket",
        "highest_small_molecule_tractability",
        pl.col("id").alias("ensembl_gene_id"),
        "protein_id_source",
    )
    .unique(
        subset=[
            "uniprot_id",
            "level_1_class",
            "level_2_class",
            "level_3_class",
        ],
        keep="first",
    )
    .sort(
        [
            "small_molecule_tractability_bucket",
            "level_1_class",
            "level_2_class",
            "level_3_class",
            "approvedSymbol",
            "uniprot_id",
        ],
        nulls_last=True,
    )
)


# =============================================================================
# Collect safely
#
# Assign to protein_families only after collect() succeeds. This prevents an
# older IPython variable from being accidentally written after a failed query.
# =============================================================================

new_protein_families = protein_families_query.collect()


if new_protein_families.is_empty():
    raise RuntimeError("The final query produced zero rows.")


required_output_columns = {
    "uniprot_id",
    "level_1_class",
    "level_2_class",
    "level_3_class",
    "highest_small_molecule_tractability_id",
    "small_molecule_tractability_bucket",
    "highest_small_molecule_tractability",
}

missing_output_columns = required_output_columns - set(new_protein_families.columns)

if missing_output_columns:
    raise RuntimeError(
        f"The final output is missing expected columns: {sorted(missing_output_columns)}"
    )


protein_families = new_protein_families


# =============================================================================
# Save output
# =============================================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

protein_families.write_csv(OUTPUT_FILE)


# =============================================================================
# Print summaries
# =============================================================================

print(f"\nSaved {protein_families.height:,} rows to:")
print(OUTPUT_FILE)


print("\nOutput columns:")
print(protein_families.columns)


print("\nUnique UniProt IDs by level-1 class:")

level_1_summary = (
    protein_families.group_by("level_1_class")
    .agg(pl.col("uniprot_id").n_unique().alias("n_unique_uniprot_ids"))
    .sort(
        "n_unique_uniprot_ids",
        descending=True,
    )
)

print(level_1_summary)


print("\nHighest small-molecule tractability levels:")

tractability_summary = (
    protein_families.group_by(
        [
            "small_molecule_tractability_bucket",
            "highest_small_molecule_tractability",
        ]
    )
    .agg(pl.col("uniprot_id").n_unique().alias("n_unique_uniprot_ids"))
    .sort(
        "small_molecule_tractability_bucket",
        nulls_last=True,
    )
)

print(tractability_summary)


print("\nUnique UniProt IDs by level-1 class and tractability:")

class_tractability_summary = (
    protein_families.group_by(
        [
            "level_1_class",
            "small_molecule_tractability_bucket",
            "highest_small_molecule_tractability",
        ]
    )
    .agg(pl.col("uniprot_id").n_unique().alias("n_unique_uniprot_ids"))
    .sort(
        [
            "level_1_class",
            "small_molecule_tractability_bucket",
        ],
        nulls_last=True,
    )
)

print(class_tractability_summary)


print("\nProtein-ID source summary:")

source_summary = (
    protein_families.group_by("protein_id_source")
    .agg(
        pl.len().alias("n_rows"),
        pl.col("uniprot_id").n_unique().alias("n_unique_uniprot_ids"),
    )
    .sort(
        "n_unique_uniprot_ids",
        descending=True,
    )
)

print(source_summary)


print("\nFirst 20 rows:")

print(protein_families.head(20))
