from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

WORK_DIR = Path.home() / "target-elucidation" / "data"
RAW_DIR = WORK_DIR / "raw"
INTERIM_DIR = WORK_DIR / "interim"

gene_effect = pd.read_csv(RAW_DIR / "CRISPRGeneEffect.csv", index_col=0)
gene_dependency = pd.read_csv(RAW_DIR / "CRISPRGeneDependency.csv", index_col=0)
model_info = pd.read_csv(RAW_DIR / "Model.csv", index_col=0)

lineage = "Lung"
disease = "Non-Small Cell Lung Cancer"
nsclc_ids = model_info.query(
    "(OncotreeLineage == @lineage) & (OncotreePrimaryDisease == @disease)"
).index
subset_data = gene_effect.loc[gene_effect.index.intersection(nsclc_ids), :]

gene_file = INTERIM_DIR / "PCP_top500_proto.txt"
with open(gene_file, "r") as file:
    gene_list = file.read().splitlines()

plot_data = subset_data[[col for col in subset_data.columns if col.split(" ")[0] in gene_list]]

# 4. Reshape for Seaborn (Long-format)
df_melted = plot_data.melt(var_name="Gene", value_name="CRISPR Gene Effect")
df_melted["Gene"] = df_melted["Gene"].str.split(" ").str[0]

# 5. Determine Order and Coloring
order = df_melted.groupby("Gene")["CRISPR Gene Effect"].median().sort_values().index
order = order[:41]
# colors = ["red" if gene in ["SLC7A5"] else "steelblue" for gene in order]

# 6. Visualization
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_melted, x="Gene", y="CRISPR Gene Effect", order=order, palette=colors, fliersize=2
)

plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.xticks(rotation=45, ha="right")
# plt.title('Gene dependency enrichment in 98 NSCLC cell lines')
plt.grid(axis="y", linestyle=":", alpha=0.7)
# plt.tight_layout()

outfile = Path.home() / "target-elucidation" / "reports" / "figures" / "depmap_luad_newds.png"
plt.savefig(outfile, bbox_inches="tight", dpi=600)
plt.close()

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

WORK_DIR = Path.home() / "target-elucidation" / "data"
RAW_DIR = WORK_DIR / "raw"
INTERIM_DIR = WORK_DIR / "interim"

# 1. Load Data
gene_effect = pd.read_csv(RAW_DIR / "CRISPRGeneEffect.csv", index_col=0)
model_info = pd.read_csv(RAW_DIR / "Model.csv", index_col=0)

# 2. Extract IDs for both cohorts
nsclc_ids = model_info.query(
    "(OncotreeLineage == 'Lung') & (OncotreePrimaryDisease == 'Non-Small Cell Lung Cancer')"
).index

pdac_ids = model_info.query(
    "(OncotreeLineage == 'Pancreas') & (OncotreePrimaryDisease == 'Pancreatic Adenocarcinoma')"
).index

# 3. Subset data and assign labels
subset_nsclc = gene_effect.loc[gene_effect.index.intersection(nsclc_ids), :].copy()
subset_nsclc["Cancer Type"] = "NSCLC"

subset_pdac = gene_effect.loc[gene_effect.index.intersection(pdac_ids), :].copy()
subset_pdac["Cancer Type"] = "PDAC"

# Combine both cohorts into one DataFrame
combined_data = pd.concat([subset_nsclc, subset_pdac])

# 4. Filter by Gene List
gene_file = INTERIM_DIR / "PCP_top500_proto.txt"
with open(gene_file, "r") as file:
    gene_list = file.read().splitlines()

# Keep the gene columns AND the 'Cancer Type' column
target_cols = [col for col in gene_effect.columns if col.split(" ")[0] in gene_list]
plot_data = combined_data[target_cols + ["Cancer Type"]]

# 5. Reshape for Seaborn (Long-format)
# Note: id_vars="Cancer Type" ensures this label stays attached to the values
df_melted = plot_data.melt(id_vars="Cancer Type", var_name="Gene", value_name="CRISPR Gene Effect")
df_melted["Gene"] = df_melted["Gene"].str.split(" ").str[0]

# 6. Determine Order
# Sorting by the combined median across both cancer types
order = df_melted.groupby("Gene")["CRISPR Gene Effect"].median().sort_values().index[:30]

# Filter melted df to only include the top 41 genes
df_melted = df_melted[df_melted["Gene"].isin(order)]

# 7. Visualization
plt.figure(figsize=(14, 6))  # Made slightly wider to accommodate split boxes

# Use 'hue' to separate NSCLC and PDAC
sns.set_context("talk")
ax = sns.boxplot(
    data=df_melted,
    x="Gene",
    y="CRISPR Gene Effect",
    hue="Cancer Type",
    order=order,
    palette={"NSCLC": "steelblue", "PDAC": "darkorange"},  # Define colors for cohorts
    fliersize=2,
)
# Move the threshold line to -0.5 (the dependency cutoff for Chronos)
plt.axhline(-0.5, color="red", linestyle="--", linewidth=1.5, label="Dependency Threshold (-0.5)")

# Optional: Add a line at 0 to show the "no effect" baseline
plt.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
plt.xticks(rotation=45, ha="right")

# Highlight "SLC7A5" text label in red instead of the box
for tick_label in ax.get_xticklabels():
    if tick_label.get_text() in ["SLC7A5", "SLC2A1"]:
        tick_label.set_color("red")
        tick_label.set_fontweight("bold")

plt.grid(axis="y", linestyle=":", alpha=0.7)
plt.legend(title="Cancer Type", loc="lower right")

# 8. Save output
outfile = (
    Path.home()
    / "target-elucidation"
    / "reports"
    / "figures"
    / "depmap_CY001_only_nsclc_vs_pdac_newds.png"
)
plt.savefig(outfile, bbox_inches="tight", dpi=600)
plt.close()

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setup Paths
WORK_DIR = Path.home() / "target-elucidation" / "data"
RAW_DIR = WORK_DIR / "raw"
INTERIM_DIR = WORK_DIR / "interim"
FIG_DIR = Path.home() / "target-elucidation" / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load Data
gene_effect = pd.read_csv(RAW_DIR / "CRISPRGeneEffect.csv", index_col=0)
gene_dependency = pd.read_csv(RAW_DIR / "CRISPRGeneDependency.csv", index_col=0)
model_info = pd.read_csv(RAW_DIR / "Model.csv", index_col=0)

# 3. Load Gene List & Filter
gene_file = INTERIM_DIR / "PCP_top500_proto.txt"
with open(gene_file, "r") as file:
    gene_list = [line.strip() for line in file.readlines()]

target_cols = [col for col in gene_effect.columns if col.split(" ")[0] in gene_list]
gene_effect = gene_effect[target_cols]
gene_dependency = gene_dependency[target_cols]

# 4. Highlight Configuration
highlight_genes = ["SLC2A1", "SLC7A5"]
cohorts = {"NSCLC": "Non-Small Cell Lung Cancer", "PAAD": "Pancreatic Adenocarcinoma"}

# 5. Iterative Processing
sns.set_context("talk")

for code, disease in cohorts.items():
    ids = model_info.query("OncotreePrimaryDisease == @disease").index
    ids = ids.intersection(gene_effect.index)

    # Calculate Metrics
    perc_dep = (gene_dependency.loc[ids] >= 0.5).mean() * 100
    median_eff = gene_effect.loc[ids].median()

    plot_df = pd.DataFrame(
        {
            "Percent Dependent": perc_dep,
            "Median Gene Effect": median_eff,
            "Gene": [c.split(" ")[0] for c in target_cols],
        }
    )
    # Define your threshold conditions
    # Example: More than 50% cells dependent, and Median Gene Effect strong (score <= -0.5)
    dep_threshold = 50.0
    eff_threshold = -0.5

    # Filter the DataFrame based on conditions
    threshold_hits = plot_df[
        (plot_df["Percent Dependent"] >= dep_threshold)
        & (plot_df["Median Gene Effect"] <= eff_threshold)
    ]

    # Extract the unique gene names as a clean list
    hit_genes = threshold_hits["Gene"].tolist()

    # Output the results for the current cohort
    print(f"\n--- {code} Threshold Hits ({len(hit_genes)} genes) ---")

    # Add Color Map for Highlighting
    plot_df["Color"] = plot_df["Gene"].apply(
        lambda x: "red" if x in highlight_genes else "steelblue"
    )

    # Visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=plot_df,
        x="Percent Dependent",
        y="Median Gene Effect",
        hue="Color",
        palette={"red": "red", "steelblue": "steelblue"},
        alpha=0.7,
        s=180,
        legend=False,
    )

    # Persistent Labeling for Highlighted Genes
    for _, row in plot_df.iterrows():
        if row["Gene"] in highlight_genes:
            plt.text(
                row["Percent Dependent"] + 1.5,
                row["Median Gene Effect"],
                row["Gene"],
                fontsize=18,
                fontweight="bold",
                color="red",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            )

    # Reference Lines & Formatting
    plt.axhline(-0.5, color="black", linestyle="--", alpha=0.5, label="Essentiality Cutoff")
    plt.axvline(50, color="gray", linestyle=":", alpha=0.5)

    plt.title(f"{code}: Metabolic Dependency Profile (n={len(ids)})")
    plt.xlabel("% Cell Lines Dependent (Prob ≥ 0.5)")
    plt.ylabel("Median Gene Score (Chronos)")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle=":", alpha=0.4)

    outfile = FIG_DIR / f"highlight_analysis_{code.lower()}_PCP_newds.png"
    plt.savefig(outfile, bbox_inches="tight", dpi=600)
    plt.close()

print(f"Highlighted plots saved to {FIG_DIR}")
