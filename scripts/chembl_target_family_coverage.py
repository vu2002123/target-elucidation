from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# File paths
# ============================================================================

OPENTARGETS_FILE = Path(
    "/home/vu2002123/target-elucidation/data/external/"
    "chembl_target_classes_uniprot_tractability.csv"
)

DATASET_FILES = {
    "Dataset 1": Path("/home/vu2002123/target-elucidation/data/HPC_input/D1_all_IDs.txt"),
    "Dataset 2": Path("/home/vu2002123/target-elucidation/data/HPC_input/D2_combined_all_IDs.txt"),
    "Dataset 3": Path("/home/vu2002123/target-elucidation/data/HPC_input/DS_Vu_all_ids.txt"),
}

INTERIM_DIR = Path("/home/vu2002123/target-elucidation/data/interim/")
FIG_DIR = Path("/home/vu2002123/target-elucidation/reports/figures/")

LEVEL_1_CLASSES = [
    "Enzyme",
    "Membrane receptor",
    "Ion channel",
    "Transcription factor",
    "Transporter",
]


# ============================================================================
# Read one UniProt ID per line
# ============================================================================


def read_uniprot_ids(path: Path) -> set[str]:
    with path.open() as file:
        return {line.strip().upper() for line in file if line.strip()}


dataset_ids = {name: read_uniprot_ids(path) for name, path in DATASET_FILES.items()}

for name, ids in dataset_ids.items():
    print(f"{name}: {len(ids):,} unique UniProt IDs")


# ============================================================================
# Read and clean the Open Targets table
# ============================================================================

targets = pd.read_csv(OPENTARGETS_FILE)
targets = targets.query("small_molecule_tractability_bucket <= 3")

required_columns = {
    "uniprot_id",
    "level_1_class",
    "level_2_class",
}

missing_columns = required_columns - set(targets.columns)

if missing_columns:
    raise ValueError(f"Missing columns in Open Targets table: {sorted(missing_columns)}")

targets["uniprot_id"] = targets["uniprot_id"].astype(str).str.strip().str.upper()

targets["level_1_class"] = targets["level_1_class"].astype("string").str.strip()

targets["level_2_class"] = targets["level_2_class"].astype("string").str.strip()

# Keep only the five requested level-1 classes.
targets = targets[targets["level_1_class"].isin(LEVEL_1_CLASSES)].copy()


# ============================================================================
# Level-1 coverage
# ============================================================================

# One protein should count only once within each level-1 class.
level_1_targets = targets[["uniprot_id", "level_1_class"]].dropna().drop_duplicates()

level_1_results = []

for target_class in LEVEL_1_CLASSES:
    class_ids = set(
        level_1_targets.loc[
            level_1_targets["level_1_class"] == target_class,
            "uniprot_id",
        ]
    )

    row = {
        "level_1_class": target_class,
        "total_targets": len(class_ids),
    }

    for dataset_name, ids in dataset_ids.items():
        overlap = len(class_ids & ids)

        row[f"{dataset_name}_count"] = overlap
        row[f"{dataset_name}_percent"] = 100 * overlap / len(class_ids) if class_ids else 0

    level_1_results.append(row)

level_1_coverage = pd.DataFrame(level_1_results)

print("\nLevel-1 coverage:")
print(level_1_coverage)


# ============================================================================
# Level-2 coverage
# ============================================================================

# Put targets without a level-2 class into an "Other <level-1 class>" group.
level_2_targets = targets[
    [
        "uniprot_id",
        "level_1_class",
        "level_2_class",
    ]
].copy()

level_2_values = level_2_targets["level_2_class"].fillna("").str.strip()
missing_level_2 = (level_2_values == "") | (level_2_values.str.lower() == "null")

level_2_targets.loc[missing_level_2, "level_2_class"] = (
    "Other " + level_2_targets.loc[missing_level_2, "level_1_class"]
)

# Some non-empty level-2 cells may contain multiple classes separated by semicolons.
level_2_targets["level_2_class"] = level_2_targets["level_2_class"].str.split(";")
level_2_targets = level_2_targets.explode("level_2_class")

level_2_targets["level_2_class"] = level_2_targets["level_2_class"].str.strip()

level_2_targets = level_2_targets[
    level_2_targets["level_2_class"].notna()
    & (level_2_targets["level_2_class"] != "")
    & (level_2_targets["level_2_class"].str.lower() != "null")
].drop_duplicates()

level_2_results = []

for (level_1_class, level_2_class), group in level_2_targets.groupby(
    ["level_1_class", "level_2_class"]
):
    subclass_ids = set(group["uniprot_id"])

    row = {
        "level_1_class": level_1_class,
        "level_2_class": level_2_class,
        "total_targets": len(subclass_ids),
    }

    for dataset_name, ids in dataset_ids.items():
        overlap = len(subclass_ids & ids)

        row[f"{dataset_name}_count"] = overlap
        row[f"{dataset_name}_percent"] = 100 * overlap / len(subclass_ids) if subclass_ids else 0

    level_2_results.append(row)

level_2_coverage = (
    pd.DataFrame(level_2_results)
    .sort_values(
        ["level_1_class", "total_targets"],
        ascending=[True, False],
    )
    .reset_index(drop=True)
)

print("\nLevel-2 coverage:")
print(level_2_coverage)


# ============================================================================
# Save result tables
# ============================================================================

level_1_coverage.to_csv(
    INTERIM_DIR / "level_1_coverage.csv",
    index=False,
)

level_2_coverage.to_csv(
    INTERIM_DIR / "level_2_coverage.csv",
    index=False,
)


# ============================================================================
# Plot level-1 coverage
# ============================================================================

dataset_names = list(DATASET_FILES)

x = np.arange(len(LEVEL_1_CLASSES))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(9.75, 6))

for index, dataset_name in enumerate(dataset_names):
    percentages = level_1_coverage[f"{dataset_name}_percent"]

    positions = x + (index - 1) * bar_width

    bars = ax.bar(
        positions,
        percentages,
        width=bar_width,
        label=dataset_name,
    )

    ax.bar_label(
        bars,
        fmt="%.1f%%",
        padding=2,
        fontsize=8,
    )

ax.set_xticks(x)
ax.set_xticklabels(
    LEVEL_1_CLASSES,
    rotation=20,
    ha="right",
    fontsize=13,
)

ax.set_ylabel("Coverage of total targets (%)", fontsize=14)
ax.set_xlabel("Level-1 target class", fontsize=14)
ax.set_title("Coverage of major target classes", fontsize=16)
ax.set_ylim(0, 105)
ax.tick_params(axis="y", labelsize=13)
ax.legend(fontsize=12)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "level_1_coverage.png",
    dpi=300,
)
fig.savefig(
    FIG_DIR / "level_1_coverage.svg",
)


# ============================================================================
# Plot level-2 coverage
# ============================================================================

bar_height = 0.30

# Draw all Level 2 subclasses in one figure. Each Level 1 family gets a
# separate panel whose height is proportional to its number of subclasses.
level_2_groups = [
    level_2_coverage[level_2_coverage["level_1_class"] == level_1_class]
    .reset_index(drop=True)
    for level_1_class in LEVEL_1_CLASSES
]
height_ratios = [max(len(group), 1) for group in level_2_groups]
figure_height = 4 + 0.52 * sum(height_ratios)

fig, axes = plt.subplots(
    nrows=len(LEVEL_1_CLASSES),
    ncols=1,
    figsize=(14, figure_height),
    sharex=True,
    gridspec_kw={
        "height_ratios": height_ratios,
        "hspace": 0.42,
    },
)

legend_handles = None
legend_labels = None

for ax, level_1_class, class_coverage in zip(
    axes,
    LEVEL_1_CLASSES,
    level_2_groups,
):
    labels = class_coverage["level_2_class"].tolist()
    y = np.arange(len(labels))

    for index, dataset_name in enumerate(dataset_names):
        percentages = class_coverage[f"{dataset_name}_percent"]
        positions = y + (index - 1) * bar_height
        ax.barh(
            positions,
            percentages,
            height=bar_height,
            label=dataset_name,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(
        level_1_class,
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=6,
    )

    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

axes[-1].set_xlabel(
    "Coverage of total targets in Level 2 subclass (%)",
    fontsize=14,
)
fig.supylabel("Level 2 target class", fontsize=14, x=0.015)
fig.suptitle(
    "Coverage of Level 2 target subclasses by Level 1 family",
    fontsize=18,
    y=0.995,
)
fig.legend(
    legend_handles,
    legend_labels,
    title="Dataset",
    fontsize=12,
    title_fontsize=13,
    loc="upper left",
    bbox_to_anchor=(0.82, 0.96),
    frameon=True,
)
fig.subplots_adjust(
    left=0.38,
    right=0.80,
    bottom=0.06,
    top=0.95,
)
fig.savefig(
    FIG_DIR / "level_2_coverage_all_families.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    FIG_DIR / "level_2_coverage_all_families.svg",
    bbox_inches="tight",
)
plt.close(fig)

print(f"\nResults saved to: {FIG_DIR}")
