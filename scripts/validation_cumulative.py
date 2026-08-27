import pandas as pd
import matplotlib.pyplot as plt


# 1. Define helper functions
def get_binders(drug_name):
    bind_file = f"/home/vu2002123/target-elucidation/data/raw/bindingDB/{drug_name}_bindingDB.tsv"
    df_bind = pd.read_csv(bind_file, sep="\t")
    df_bind_filtered = df_bind.dropna(subset=["Ki (nM)", "Kd (nM)"], how="all")
    df_bind_filtered = df_bind_filtered.replace(r">.*", None, regex=True)

    for col in ["Ki (nM)", "Kd (nM)"]:
        df_bind_filtered[col] = (
            df_bind_filtered[col].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
        )

    df_final = df_bind_filtered[
        (df_bind_filtered["Ki (nM)"] < 1000) | (df_bind_filtered["Kd (nM)"] < 1000)
    ]
    return set(df_final.iloc[:, 44])


def calculate_enrichment(df, binders_set, score_column="CNN_VS"):
    df_sorted = df.sort_values(by=score_column, ascending=False).reset_index(drop=True)
    df_sorted["is_target"] = df_sorted["UNIPROT_ID"].isin(binders_set).astype(int)

    total_proteins = len(df_sorted)
    total_targets = df_sorted["is_target"].sum()

    if total_targets == 0:
        df_sorted["percent_screened"] = ((df_sorted.index + 1) / total_proteins) * 100
        df_sorted["recall"] = 0.0
        return df_sorted

    df_sorted["percent_screened"] = ((df_sorted.index + 1) / total_proteins) * 100
    df_sorted["cumulative_targets"] = df_sorted["is_target"].cumsum()
    df_sorted["recall"] = (df_sorted["cumulative_targets"] / total_targets) * 100

    return df_sorted


def load_and_combine_docking_scores(drug_name):
    dock_file_d1 = (
        f"/home/vu2002123/target-elucidation/data/interim/{drug_name}_LUAD_D1_full_score.csv"
    )
    dock_file_d2 = (
        f"/home/vu2002123/target-elucidation/data/interim/{drug_name}_LUAD_D2_full_score.csv"
    )

    dock_score_d1 = pd.read_csv(dock_file_d1)
    dock_score_d2 = pd.read_csv(dock_file_d2)

    dock_score_combined = (
        pd.concat([dock_score_d1, dock_score_d2])
        .sort_values(by="CNN_VS", ascending=False)
        .drop_duplicates(subset="UNIPROT_ID")
        .reset_index(drop=True)
    )
    return dock_score_combined


# 2. Main execution loop
drug_list = ["Afatinib", "Erlotinib", "Crizotinib", "Gefitinib", "Ruxolitinib"]

# Increased figure size slightly to accommodate larger fonts
plt.figure(figsize=(10, 8))

for drug in drug_list:
    binders = get_binders(drug)
    df_dock = load_and_combine_docking_scores(drug)
    df_processed = calculate_enrichment(df_dock, binders)

    # Increased line width for better visibility
    plt.plot(df_processed["percent_screened"], df_processed["recall"], label=drug, linewidth=2.5)

# 3. Add baselines and highlights
plt.plot([0, 100], [0, 100], label="Random Selection", linestyle="--", color="gray", linewidth=2.5)

plt.axvline(x=20, color="red", linestyle="--", linewidth=2, label="20% Cutoff")
plt.axvspan(0, 20, color="red", alpha=0.05)

# 4. Format the graph with increased text sizes
plt.xlim(0, 100)
plt.ylim(0, 100)

# Scaled up axis labels and added bold weight
plt.xlabel("Database Screened (%)", fontsize=16, fontweight="bold")
plt.ylabel("True Targets Found / Recall (%)", fontsize=16, fontweight="bold")

# Scaled up title
plt.title("Cumulative Enrichment Comparison Across Drugs", fontsize=18, fontweight="bold", pad=15)

# Scaled up tick marks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Combined, enlarged legend with an opaque frame
plt.legend(loc="lower right", fontsize=14, framealpha=0.95, edgecolor="black")

plt.grid(True, linestyle=":", alpha=0.6)

# Save output
save_path = "/home/vu2002123/target-elucidation/reports/figures/multi_drug_enrichment.png"
plt.savefig(save_path, bbox_inches="tight", dpi=600)
