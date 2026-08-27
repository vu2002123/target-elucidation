from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path.home() / "target-elucidation" / "data" / "raw"
HPC_DIR = Path.home() / "target-elucidation" / "data" / "HPC_input"
FIG_DIR = Path.home() / "target-elucidation" / "reports" / "figures"
annotation_file = DATA_DIR / "uniprot_human_count_total.tsv"
id_file = DATA_DIR / "uniprot_ids.txt"

domain_name_df = pd.read_csv(DATA_DIR / "names.dat", sep="\t", header=None, names=["Key", "Value"])
domain_name_dict = dict(zip(domain_name_df["Key"], domain_name_df["Value"]))

df = (
    pd.read_csv(annotation_file, sep="\t", header=None)
    .rename(
        columns={
            0: "Accession",
            1: "Name",
            2: "Source_database",
            3: "Type",
            4: "Integrated",
            5: "Member_databases",
            6: "GO_terms",
            7: "Protein_accession",
            8: "Protein_length",
            9: "Entry_protein_locations",
        }
    )
    .drop(columns=[10])
)
df = df.dropna(subset="Protein_accession")

with open(id_file, "r") as file:
    ids = [line.strip().lower() for line in file]

df_grouped = (
    df.query("Type == 'family'")
    .groupby("Protein_accession")
    .agg({"Accession": list})
    .reset_index()
)

family_dict = {}
current_root = None
family_tree_file = DATA_DIR / "interpro_tree.txt"
with open(family_tree_file, "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        # Split the line by '::' and filter out empty strings
        parts = [part for part in line.split("::") if part]
        if len(parts) < 2:
            continue
        if line.startswith("--"):
            # It is a child domain
            family_id = parts[0].lstrip("-")
            family_name = parts[1]
            # Append to the current active root node
            if current_root and current_root in family_dict:
                family_dict[current_root]["children"][family_id] = family_name
        else:
            # It is a root domain
            family_id = parts[0]
            family_name = parts[1]
            current_root = family_id
            # Initialize the root in the dictionary
            family_dict[current_root] = {"name": family_name, "children": {}}

child_to_parent = {}
for parent_id, data in family_dict.items():
    for child_id in data["children"].keys():
        child_to_parent[child_id] = parent_id

filtered_family = {}
for _, row in df_grouped.iterrows():
    protein = row.Protein_accession
    families = list(row.Accession)
    kept_families = []
    for family in families:
        parent_id = child_to_parent.get(family)
        if parent_id in families:
            continue
        else:
            kept_families.append(family)
    filtered_family[protein] = kept_families

df_grouped["Accession_filtered"] = df_grouped["Protein_accession"].map(filtered_family)

present_id = set(df_grouped["Protein_accession"])
missing_id = list(set(ids) - present_id)

df_missing = pd.DataFrame(
    {
        "Protein_accession": missing_id,
        "Accession": [[] for _ in range(len(missing_id))],
        "Accession_filtered": [[] for _ in range(len(missing_id))],
    }
)
df_merged = pd.concat([df_grouped, df_missing], ignore_index=True)

df_merged["Count"] = df_merged["Accession_filtered"].str.len()

df_merged["Accession_filtered"].explode().value_counts().head(10)
df_merged["Protein_accession"] = df_merged["Protein_accession"].str.upper()

with open(HPC_DIR / "DS_Vu_all_ids.txt", "r") as file:
    kept_ids = [line.strip() for line in file]

with open(HPC_DIR / "D1_all_IDs.txt", "r") as file:
    kept_ids_D1 = [line.strip() for line in file]

with open(HPC_DIR / "D2_combined_all_IDs.txt", "r") as file:
    kept_ids_D2 = [line.strip() for line in file]

# --- Query target datasets ---
df_merged_filtered = df_merged.query("Protein_accession in @kept_ids")
df_merged_filtered_D1 = df_merged.query("Protein_accession in @kept_ids_D1")
df_merged_filtered_D2 = df_merged.query("Protein_accession in @kept_ids_D2")

# --- Extract raw InterPro IDs of the top 10 families from D3 (DS_Vu) ---
top_10_interpro_ids = (
    df_merged_filtered["Accession_filtered"].explode().value_counts().head(10).index
)

# --- Calculate counts for all three datasets using the same InterPro IDs ---
d3_counts = (
    df_merged_filtered["Accession_filtered"]
    .explode()
    .value_counts()
    .reindex(top_10_interpro_ids, fill_value=0)
)
d1_counts = (
    df_merged_filtered_D1["Accession_filtered"]
    .explode()
    .value_counts()
    .reindex(top_10_interpro_ids, fill_value=0)
)
d2_counts = (
    df_merged_filtered_D2["Accession_filtered"]
    .explode()
    .value_counts()
    .reindex(top_10_interpro_ids, fill_value=0)
)

# --- Construct a Long-Format DataFrame for Seaborn Grouped Barplot ---
plot_data = pd.DataFrame(
    {
        "Family_ID": list(top_10_interpro_ids) * 3,
        "Family_Name": list(top_10_interpro_ids.map(domain_name_dict)) * 3,
        "Count": list(d1_counts.values) + list(d2_counts.values) + list(d3_counts.values),
        "Dataset": ["D1"] * 10 + ["D2"] * 10 + ["D3"] * 10,
    }
)

# Denominators for percentage calculation
total_D3 = len(kept_ids)
total_D1 = len(kept_ids_D1)
total_D2 = len(kept_ids_D2)

totals = {"D1": total_D1, "D2": total_D2, "D3": total_D3}

# Calculate relative percentages using mapped dictionary to avoid label mismatch errors
plot_data["Percentage"] = plot_data.apply(
    lambda row: (row["Count"] / totals[row["Dataset"]] * 100) if totals[row["Dataset"]] > 0 else 0,
    axis=1,
)

# --- Plotting Grouped Bar Chart ---
plt.figure(figsize=(16, 8))  # Wider canvas to avoid overlapping labels on 3-bar groups
ax = sns.barplot(
    data=plot_data,
    x="Family_Name",
    y="Count",
    hue="Dataset",
    hue_order=["D1", "D2", "D3"],
    palette="muted",
)

ax.set_ylim(0, plot_data["Count"].max() * 1.15)
plt.xticks(rotation=45, ha="right")

# Dynamically add count and percentage labels above each bar
for container in ax.containers:
    labels = []
    dataset_label = container.get_label()
    tot = totals.get(dataset_label, 0)

    ax.bar_label(container, labels=labels, padding=4, fontsize=12)

plt.title("Family Frequency Comparison: D1 vs D2 vs D3", fontsize=14, fontweight="bold")
plt.xlabel("Protein Family", fontsize=12)
plt.ylabel("Protein Count", fontsize=12)
plt.legend(title="Dataset", loc="upper right")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "family_frequency_comparison.png", bbox_inches="tight", dpi=300)
plt.close("all")

df_merged_filtered.value_counts(subset="Count")

total = len(kept_ids)
domain_freq_top = df_merged_filtered["Accession_filtered"].explode().value_counts().head(10)
domain_freq_top.index = domain_freq_top.index.map(domain_name_dict)

with open(HPC_DIR / "D1_all_IDs.txt", "r") as file:
    kept_ids_D1 = [line.strip() for line in file]

plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x=domain_freq_top.index,
    y=domain_freq_top.values,
    hue=domain_freq_top.index,
    palette="tab10",
    legend="brief",
)
for container in ax.containers:
    percentages = [f"{(bar.get_height() / total) * 100:.1f}%" for bar in container]
    ax.bar_label(container, labels=percentages, padding=3)
ax.set_xticklabels([])
sns.move_legend(ax, "upper right", title="Families")
plt.title("Family Frequency")
plt.xlabel("Family")
plt.ylabel("Count")
plt.savefig(FIG_DIR / "family_frequency_DS1.png", bbox_inches="tight")
plt.close("all")

# out_file = DATA_DIR / "all_family.csv"
# df_merged.to_csv(out_file, index=False)
