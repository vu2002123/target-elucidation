from gc import get_debug
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from rdkit.Chem import CleanupStereoGroups
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
)
from scipy import stats

DATA_DIR = Path.home() / "target-elucidation" / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
HPC_DIR = DATA_DIR / "HPC_mount"
FIG_DIR = Path.home() / "target-elucidation" / "reports" / "figures"

file_location = pd.read_csv(INTERIM_DIR / "file_locations.csv")

drug_name = "Erlotinib"
drug_name_2 = drug_name

binder_file = f"/home/vu2002123/target-elucidation/data/raw/pubchem/{drug_name}_filtered_total.txt"
with open(binder_file, "r") as file:
    binders = [line.strip() for line in file]

dock_file_1 = file_location.query("Compound == @drug_name and Dataset == 1")["File_location"].iloc[
    0
]

dock_score_1 = pd.read_csv(INTERIM_DIR / dock_file_1)
dock_score_1["UNIPROT_ID"] = dock_score_1["File_Name"].str.split("-").str[1]
dock_score_1 = dock_score_1.query("Compound == @drug_name_2")
dock_score_1 = dock_score_1[dock_score_1["minimizedAffinity"] < 0]
dock_score_1 = (
    dock_score_1.sort_values(by="CNNaffinity", ascending=False)
    .drop_duplicates(subset="UNIPROT_ID")
    .reset_index(drop=True)
)

# dock_score_1 = pd.read_csv(INTERIM_DIR / dock_file_1, sep="\t")
# dock_score_1 = dock_score_1[dock_score_1["affinity"] < 0]

dock_score_1["is_target"] = dock_score_1["UNIPROT_ID"].isin(binders).astype(int)
y_true1 = dock_score_1["is_target"]
y_scores1 = dock_score_1["CNNaffinity"]
fpr1, tpr1, _ = roc_curve(y_true1, y_scores1)
roc_auc1 = auc(fpr1, tpr1)
precision1, recall1, _ = precision_recall_curve(y_true1, y_scores1)
pr_auc1 = auc(recall1, precision1)
avg_precision1 = average_precision_score(y_true1, y_scores1)
baseline1 = y_true1.sum() / len(y_true1)

dock_file_2 = file_location.query("Compound == @drug_name and Dataset == 2")["File_location"].iloc[
    0
]

dock_score_2 = pd.read_csv(INTERIM_DIR / dock_file_2)
dock_score_2["UNIPROT_ID"] = dock_score_2["File_Name"].str.split("_").str[0]
dock_score_2 = dock_score_2.query("Compound == @drug_name_2")
dock_score_2 = dock_score_2[dock_score_2["minimizedAffinity"] < 0]
dock_score_2 = (
    dock_score_2.sort_values(by="CNNaffinity", ascending=False)
    .drop_duplicates(subset="UNIPROT_ID")
    .reset_index(drop=True)
)

# dock_score_2 = pd.read_csv(INTERIM_DIR / dock_file_2, sep="\t")
# dock_score_2["UNIPROT_ID"] = dock_score_2["File_Name"].str.split("_").str[0]
# dock_score_2 = dock_score_2[dock_score_2["affinity"] < 0]
# dock_score_2 = (
#     dock_score_2.sort_values(by="CNNaffinity", ascending=False)
#     .drop_duplicates(subset="UNIPROT_ID")
#     .reset_index(drop=True)
# )
dock_score_2["is_target"] = dock_score_2["UNIPROT_ID"].isin(binders).astype(int)
y_true2 = dock_score_2["is_target"]
y_scores2 = dock_score_2["CNNaffinity"]
fpr2, tpr2, _ = roc_curve(y_true2, y_scores2)
roc_auc2 = auc(fpr2, tpr2)
precision2, recall2, _ = precision_recall_curve(y_true2, y_scores2)
pr_auc2 = auc(recall2, precision2)
avg_precision2 = average_precision_score(y_true2, y_scores2)
baseline2 = y_true2.sum() / len(y_true2)

dock_file_3 = file_location.query("Compound == @drug_name and Dataset == 3")["File_location"].iloc[
    0
]
dock_score_3 = pd.read_csv(INTERIM_DIR / dock_file_3)
dock_score_3["UNIPROT_ID"] = dock_score_3["File_Name"].str.split("_").str[0]
dock_score_3 = dock_score_3.query("Compound == @drug_name_2")
dock_score_3["Prefix"] = dock_score_3["File_Name"].str.split("_").str[:-1].str.join("_")

normalize_df = pd.read_csv(HPC_DIR / "D1_validation_90cp_out.csv")
normalize_df["UNIPROT_ID"] = normalize_df["File_Name"].str.split("_").str[0]
normalize_df["Prefix"] = normalize_df["File_Name"].str.split("_").str[:-1].str.join("_")
dock_score_3_merged = pd.concat([dock_score_3, normalize_df], axis=0, ignore_index=True)
sort_crit = "CNNaffinity"
dock_score_3_merged["Receptor_z"] = dock_score_3_merged.groupby("Prefix")[sort_crit].transform(
    lambda x: stats.zscore(x, nan_policy="omit")
)
dock_score_3_merged["Ligand_z"] = dock_score_3_merged.groupby("Compound")[sort_crit].transform(
    lambda x: stats.zscore(x, nan_policy="omit")
)
receptor_weight = 0.7
dock_score_3_merged["Combined_z"] = (
    receptor_weight * dock_score_3_merged["Receptor_z"]
    + (1 - receptor_weight) * dock_score_3_merged["Ligand_z"]
)
# dock_score_3 = dock_score_3[dock_score_3["minimizedAffinity"] < 0]
dock_score_3_normalized = (
    dock_score_3_merged.query("Compound == @drug_name_2")
    .sort_values(by="CNNaffinity", ascending=False)
    .drop_duplicates(subset="UNIPROT_ID")
    .reset_index(drop=True)
)
dock_score_3_normalized["is_target"] = (
    dock_score_3_normalized["UNIPROT_ID"].isin(binders).astype(int)
)
y_true3 = dock_score_3_normalized["is_target"]
y_scores3 = dock_score_3_normalized["CNNaffinity"]
fpr3, tpr3, _ = roc_curve(y_true3, y_scores3)
roc_auc3 = auc(fpr3, tpr3)
precision3, recall3, _ = precision_recall_curve(y_true3, y_scores3)
pr_auc3 = auc(recall3, precision3)
avg_precision3 = average_precision_score(y_true3, y_scores3)
baseline3 = y_true3.sum() / len(y_true3)

print(f"{roc_auc1}\n{roc_auc2}\n{roc_auc3}")
print(f"{pr_auc1}\n{pr_auc2}\n{pr_auc3}")
print(f"{avg_precision1}\n{avg_precision2}\n{avg_precision3}")
print(f"{baseline1}\n{baseline2}\n{baseline3}")
print(f"{avg_precision1 / baseline1}\n{avg_precision2 / baseline2}\n{avg_precision3 / baseline3}")
print(
    f"{y_true1.sum()}/{len(binders)}\n{y_true2.sum()}/{len(binders)}\n{y_true3.sum()}/{len(binders)}"
)
#
# total_binders = len(binders)
# avail_1 = (y_true1.sum() / total_binders) * 100
# avail_2 = (y_true2.sum() / total_binders) * 100
# avail_3 = (y_true3.sum() / total_binders) * 100
#
# plt.figure(figsize=(6, 6))
#
# # Inject the availability metric directly into the label string
# plt.plot(fpr1, tpr1, label=f"Dataset 1 (AUC={roc_auc1:.3f} | Avail={avail_1:.1f}%)", color="tab:blue", linewidth=2, alpha=0.7)
# plt.plot(fpr2, tpr2, label=f"Dataset 2 (AUC={roc_auc2:.3f} | Avail={avail_2:.1f}%)", color="tab:orange", linewidth=2, alpha=0.7)
# plt.plot(fpr3, tpr3, label=f"Dataset 3 (AUC={roc_auc3:.3f} | Avail={avail_3:.1f}%)", color="tab:green", linewidth=2)
# plt.plot([0, 1], [0, 1], "k--", label="Random Guess", alpha=0.7)
#
# plt.title(f"ROC Curves Comparison - {drug_name}")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend(loc="lower right")
# plt.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig(FIG_DIR / f"{drug_name}_ROC_comparison_normalized.png", bbox_inches="tight", dpi=300)
# plt.close()


def get_DEG(cancer_type: str, fold_change: int):
    DEG_file = "DEG_" + cancer_type + "_all.csv"
    DEG_file_path = Path.home() / f"target-elucidation/data/interim/{DEG_file}"
    DEG_data = pd.read_csv(DEG_file_path, sep=",")
    DEG_up_list = (
        DEG_data.loc[(DEG_data.log2FoldChange >= fold_change) & (DEG_data.padj < 0.05)][
            "Gene_name"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    return DEG_up_list


# Get prognosis list from prognosis file
def get_prog_km(cancer_type: str):
    PROG_file = "survival_TCGA-" + cancer_type + "-all.csv"
    PROG_file_path = Path.home() / f"target-elucidation/data/interim/{PROG_file}"
    PROG_data = pd.read_csv(PROG_file_path, sep=",")
    PROG_list_KM = (
        PROG_data.loc[PROG_data.km_p_value < 0.05]["Gene_name"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    return PROG_list_KM


def get_prog_cox(cancer_type: str):
    PROG_file = "survival_TCGA-" + cancer_type + "-all.csv"
    PROG_file_path = Path.home() / f"target-elucidation/data/interim/{PROG_file}"
    PROG_data = pd.read_csv(PROG_file_path, sep=",")
    PROG_list_COX = (
        PROG_data.loc[PROG_data.cox_p_value < 0.05]["Gene_name"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    return PROG_list_COX


# cancer_type = "CRC"
# deg_list = get_DEG(cancer_type=cancer_type, fold_change=1)
# cox_list = get_prog_cox(cancer_type=cancer_type)
# km_list = get_prog_km(cancer_type=cancer_type)
#
# with open(
#     f"/home/vu2002123/target-elucidation/data/interim/DEG_{cancer_type}_list.txt", "w"
# ) as file:
#     for gene_name in deg_list:
#         file.write(f"{gene_name}\n")
# with open(
#     f"/home/vu2002123/target-elucidation/data/interim/COX_{cancer_type}_list.txt", "w"
# ) as file:
#     for gene_name in cox_list:
#         file.write(f"{gene_name}\n")
# with open(
#     f"/home/vu2002123/target-elucidation/data/interim/KM_{cancer_type}_list.txt", "w"
# ) as file:
#     for gene_name in km_list:
#         file.write(f"{gene_name}\n")

uniprot_ids = pd.read_csv("/home/vu2002123/target-elucidation/data/raw/uniprot_ids.tsv", sep="\t")
uniprot_ids["Representative gene"] = uniprot_ids["Gene Names"].str.split(" ").str[0]

gene_dict = uniprot_ids.set_index("Entry")["Representative gene"].to_dict()

# dock_score_1_sorted = (
#     dock_score_1.query("Compound == @drug_name_2")
#     .sort_values(by="CNNaffinity", ascending=False)
#     .drop_duplicates(subset="UNIPROT_ID")
#     .reset_index(drop=True)
# )
# dock_score_1_sorted["Gene_Name"] = dock_score_1_sorted["UNIPROT_ID"].map(gene_dict)
dock_score_1_sorted = dock_score_1

top_20_count = max(1, int(len(dock_score_1_sorted) * 0.2))
top_20_positional = dock_score_1_sorted.iloc[:top_20_count]
with open(
    f"/home/vu2002123/target-elucidation/data/interim/{drug_name_2}_1_top20percent.txt", "w"
) as file:
    for id in set(top_20_positional["Gene_Name"]):
        file.write(f"{id}\n")

# dock_score_2_sorted = (
#     dock_score_2.query("Compound == @drug_name_2")
#     .sort_values(by="CNNaffinity", ascending=False)
#     .drop_duplicates(subset="UNIPROT_ID")
#     .reset_index(drop=True)
# )
dock_score_2_sorted = dock_score_2
dock_score_2_sorted["Gene_Name"] = dock_score_2_sorted["UNIPROT_ID"].map(gene_dict)
top_20_count = max(1, int(len(dock_score_2_sorted) * 0.2))
top_20_positional = dock_score_2_sorted.iloc[:top_20_count]
with open(
    f"/home/vu2002123/target-elucidation/data/interim/{drug_name_2}_2_top20percent.txt", "w"
) as file:
    for id in set(top_20_positional["Gene_Name"]):
        file.write(f"{id}\n")

dock_score_3_sorted = (
    dock_score_3.query("Compound == @drug_name_2")
    .sort_values(by="CNNaffinity", ascending=False)
    .drop_duplicates(subset="UNIPROT_ID")
    .reset_index(drop=True)
)
dock_score_3_sorted["Gene_Name"] = dock_score_3_sorted["UNIPROT_ID"].map(gene_dict)
top_20_count = max(1, int(len(dock_score_3_sorted) * 0.2))
top_20_positional = dock_score_3_sorted.iloc[:top_20_count]
with open(
    f"/home/vu2002123/target-elucidation/data/interim/{drug_name_2}_3_top20percent.txt", "w"
) as file:
    for id in set(top_20_positional["Gene_Name"]):
        file.write(f"{id}\n")
