import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc


def load_and_combine_docking_scores(drug_name, batch_name, dataset, weight: float):
    dock_file_d1 = (
        f"/home/vu2002123/target-elucidation/data/interim/{drug_name}_{batch_name}_full_score.csv"
    )
    dock_file_d2 = (
        f"/home/vu2002123/target-elucidation/data/interim/{drug_name}_{batch_name}_full_score.csv"
    )

    dock_score_d1 = pd.read_csv(dock_file_d1)
    dock_score_d1["Consensus"] = dock_score_d1["CNN_VS"] * weight + (
        -dock_score_d1["minimizedAffinity"]
    ) * (1 - weight)
    dock_score_d2 = pd.read_csv(dock_file_d2)
    dock_score_d2["Consensus"] = dock_score_d2["CNN_VS"] * weight + (
        -dock_score_d2["minimizedAffinity"]
    ) * (1 - weight)

    if dataset == 1:
        dock_score_combined = (
            dock_score_d1.sort_values(by="Consensus", ascending=False)
            .drop_duplicates(subset="UNIPROT_ID")
            .reset_index(drop=True)
            .drop(columns=["Unnamed: 0"])
        )
    elif dataset == 2:
        dock_score_combined = (
            dock_score_d2.sort_values(by="Consensus", ascending=False)
            .drop_duplicates(subset="UNIPROT_ID")
            .reset_index(drop=True)
            .drop(columns=["Unnamed: 0"])
        )
    else:
        dock_score_combined = (
            pd.concat([dock_score_d1, dock_score_d2])
            .sort_values(by="Consensus", ascending=False)
            .drop_duplicates(subset="UNIPROT_ID")
            .reset_index(drop=True)
            .drop(columns=["Unnamed: 0"])
        )
    return dock_score_combined


def load_scores_newds(drug_name, batch_name, weight: float):
    dock_file = (
        f"/home/vu2002123/target-elucidation/data/interim/{drug_name}_{batch_name}_full_score.csv"
    )
    dock_score = pd.read_csv(dock_file)
    dock_score["Consensus"] = dock_score["CNNaffinity"] * weight + (
        -dock_score["minimizedAffinity"]
    ) * (1 - weight)
    dock_score = (
        dock_score.sort_values(by="Consensus", ascending=False)
        .drop_duplicates(subset="UNIPROT_ID")
        .reset_index(drop=True)
    )
    return dock_score


drug = "Prochlorperazine"
abbr = "PCP"
thresholds = [0.05, 0.10, 1.00]  # 5%, 10%, and 20%
# datasets = [1, 2]
weight = 1.0
# Using a distinct color palette for the three threshold lines
colors = ["#F0C571", "#59A89C", "#E02B35"]

# Load binders once
# binder_file = f"/home/vu2002123/target-elucidation/data/raw/{drug}_cc.txt"
binder_file = f"/home/vu2002123/target-elucidation/data/raw/pubchem/{drug}_filtered.txt"
with open(binder_file, "r") as file:
    binders = [line.strip() for line in file]

# Pre-load the full datasets to avoid redundant loading in the loop
# raw_dfs = {
#     ds: load_and_combine_docking_scores(
#         drug, batch_name=f"D{ds}_validation", dataset=ds, weight=weight
#     )
#     for ds in datasets
# }

# proteins_1 = set(raw_dfs[1]["UNIPROT_ID"])
# proteins_2 = set(raw_dfs[2]["UNIPROT_ID"])
# missing = set(binders) - proteins_1.union(proteins_2)
# present = set(binders) - missing

# Initialize the figure
plt.figure(figsize=(8, 6))

df_dock = load_scores_newds(drug_name="PCP", batch_name="DS_PCP", weight=weight)
df_dock["is_target"] = df_dock["UNIPROT_ID"].isin(binders).astype(int)


y_true = df_dock["is_target"]
y_scores = df_dock["Consensus"]

# Calculate ROC metrics
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot the line for the current threshold
plt.plot(
    fpr, tpr, color="#F0C571", lw=2, label=f"AUC of new dataset for PCP (AUC = {roc_auc:.3f})"
)

# for threshold, color in zip(thresholds, colors):
#     df_dock_combined = pd.DataFrame()
#
#     for ds in datasets:
#         df_dock = raw_dfs[ds].copy()
#         df_dock["is_target"] = df_dock["UNIPROT_ID"].isin(binders).astype(int)
#
#         # Slice based on the current threshold
#         df_filtered_len = int(len(df_dock) * threshold)
#         df_subset = df_dock.iloc[:df_filtered_len, :].copy()
#
#         df_dock_combined = pd.concat([df_dock_combined, df_subset])
#
#     df_dock_combined = (
#         df_dock_combined.sort_values(by="Consensus", ascending=False)
#         .drop_duplicates(subset="UNIPROT_ID")
#         .reset_index(drop=True)
#     )
#
#     y_true = df_dock_combined["is_target"]
#     y_scores = df_dock_combined["Consensus"]
#
#     if threshold == 1:
#         present_list_path = (
#             f"/home/vu2002123/target-elucidation/data/interim/{drug}_present_{weight}.csv"
#         )
#         df_dock_combined.query("is_target == 1").to_csv(present_list_path)
#
#     # Calculate ROC metrics
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
#
#     # Plot the line for the current threshold
#     label_name = f"Top {int(threshold * 100)}% (AUC = {roc_auc:.3f})"
#     plt.plot(fpr, tpr, color=color, lw=2, label=label_name)

# Global plot formatting
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Selection")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)", fontsize=12)
plt.ylabel("True Positive Rate (Recall)", fontsize=12)
plt.title(f"Combined Dataset ROC: Effect of Selection Threshold ({drug})", fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)

# Save the multi-line graph
save_path = f"/home/vu2002123/target-elucidation/reports/figures/DS_PCP_{abbr}_auc.png"
plt.savefig(save_path, bbox_inches="tight", dpi=600)
plt.close("all")

# print(f"Total number of targets: {len(binders)}\nNumber of missing targets: {len(missing)}")

# def get_binders(drug_name):
#     bind_file = f"/home/vu2002123/target-elucidation/data/raw/bindingDB/{drug_name}_bindingDB.tsv"
#     df_bind = pd.read_csv(bind_file, sep="\t")
#     df_bind_filtered = df_bind.dropna(subset=["Ki (nM)", "Kd (nM)"], how="all")
#     df_bind_filtered = df_bind_filtered.replace(r">.*", None, regex=True)
#
#     for col in ["Ki (nM)", "Kd (nM)"]:
#         df_bind_filtered[col] = (
#             df_bind_filtered[col].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
#         )
#
#     df_final = df_bind_filtered[
#         (df_bind_filtered["Ki (nM)"] < 1000) | (df_bind_filtered["Kd (nM)"] < 1000)
#     ]
#     return set(df_final.iloc[:, 44])
#
# precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
# pr_auc = auc(recall, precision)
# plt.figure(figsize=(8, 6))
# plt.plot(
#     recall, precision, color="purple", lw=2, label=f"Inverse Docking PR Curve (AUC = {pr_auc:.3f})"
# )
# baseline = sum(y_true) / len(y_true)
# plt.plot(
#     [0, 1],
#     [baseline, baseline],
#     linestyle="--",
#     color="gray",
#     label=f"Random Baseline ({baseline:.4f})",
# )
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("Recall (Fraction of Targets Found)", fontsize=12)
# plt.ylabel("Precision (Hit Rate within Selection)", fontsize=12)
# plt.title("Precision-Recall Curve", fontsize=14)
# plt.legend(loc="upper right")
# plt.grid(True, linestyle=":", alpha=0.6)
# plt.savefig(
#     "/home/vu2002123/target-elucidation/reports/figures/pr_curve_erlotinib_combined.png",
#     bbox_inches="tight",
#     dpi=600,
# )

# drug = "Crizotinib"
# threshold = 0.2
# binders = get_binders(drug)
#
# df_dock = load_and_combine_docking_scores(drug, dataset=1)
# df_dock["is_target"] = df_dock["UNIPROT_ID"].isin(binders).astype(int)
# df_filtered_len = int(len(df_dock) * threshold)
# df_dock = df_dock.iloc[:df_filtered_len,:]
#
# # 1. Load your processed dataframe (assuming df_sorted from previous steps)
# y_true = df_dock["is_target"]
# y_scores = df_dock["CNN_VS"]
#
# # 2. Calculate ROC metrics
# fpr, tpr, thresholds = roc_curve(y_true, y_scores)
# # Calculate Full AUC
# full_auc = auc(fpr, tpr)
# # Calculate Partial AUC at 20% FPR (max_fpr=0.2)
# # Setting multi_class/labels isn't needed for binary, but max_fpr is the key.
# # p_auc_20 = roc_auc_score(y_true, y_scores, max_fpr=0.2)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color="darkorange", lw=2,
#          label=f"Full ROC (AUC = {full_auc:.3f})")
# # plt.fill_between(fpr, tpr, where=(fpr <= 0.2), alpha=0.2, color='blue',
# #                  label=f"Partial AUC @ 20% FPR = {p_auc_20:.3f}")
# plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Selection")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# # plt.axvline(x=0.2, color="red", linestyle=":", label="20% FPR Cutoff")
# plt.xlabel("False Positive Rate (FPR)", fontsize=12)
# plt.ylabel("True Positive Rate (Recall)", foPCPntsize=12)
# plt.title(f"ROC Curve for {drug} (Target Elucidation)", fontsize=14)
# plt.legend(loc="lower right", fontsize=10)
# plt.grid(True, linestyle=":", alpha=0.6)
# save_path = f"/home/vu2002123/target-elucidation/reports/figures/roc_20_auc_{drug}_combined.png"
# plt.savefig(save_path, bbox_inches="tight", dpi=600)

# drug = "Crizotinib"
# threshold = 0.2
# datasets = [1, 2]  # List of datasets to compare
# colors = ["#59A89C", "#E02B35"]
# # binders = get_binders(drug)
# binder_file = f"/home/vu2002123/target-elucidation/data/raw/{drug}_cc.txt"
# with open(binder_file, "r") as file:
#     binders = [line.strip() for line in file]
# # 1. Initialize the figure once
# plt.figure(figsize=(8, 6))
#
# df_dock_combined = pd.DataFrame()
#
# for ds, color in zip(datasets, colors):
#     # Load and process specific dataset
#     df_dock = load_and_combine_docking_scores(drug, dataset=ds)
#     df_dock["is_target"] = df_dock["UNIPROT_ID"].isin(binders).astype(int)
#     # Optional: If you strictly want to evaluate ONLY the top 20%
#     # Note: For standard ROC curves, it is usually better to use the full list
#     # but to compare early enrichment, you can slice here:
#     df_filtered_len = int(len(df_dock) * threshold)
#     df_subset = df_dock.iloc[:df_filtered_len, :].copy()
#     df_dock_combined = pd.concat([df_dock_combined, df_subset])
#     y_true = df_subset["is_target"]
#     y_scores = df_subset["CNN_VS"]
#     # 2. Calculate ROC metrics for this specific line
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
#     # 3. Add the line to the existing plot
#     # plt.plot(fpr, tpr, color=color, lw=2, label=f"Dataset {ds} (AUC = {roc_auc:.3f})")
#
# df_dock_combined = (
#     df_dock_combined.sort_values(by="CNN_VS", ascending=False)
#     .drop_duplicates(subset="UNIPROT_ID")
#     .reset_index(drop=True)
# )
# y_true = df_dock_combined["is_target"]
# y_scores = df_dock_combined["CNN_VS"]
# # 2. Calculate ROC metrics for this specific line
# fpr, tpr, _ = roc_curve(y_true, y_scores)
# roc_auc = auc(fpr, tpr)
# # 3. Add the line to the existing plot
# plt.plot(fpr, tpr, color="#F0C571", lw=2, label=f"Combined_dataset (AUC = {roc_auc:.3f})")
#
#
# # 4. Add global formatting (after the loop)
# plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Selection")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate (FPR)", fontsize=12)
# plt.ylabel("True Positive Rate (Recall)", fontsize=12)
# plt.title(f"ROC Comparison for {drug}", fontsize=14)
# plt.legend(loc="lower right", fontsize=10)
# plt.grid(True, linestyle=":", alpha=0.6)
# # 5. Save the combined graph
# save_path = f"/home/vu2002123/target-elucidation/reports/figures/roc_comparison_{drug}_cc_{int(threshold * 100)}.png"
# plt.savefig(save_path, bbox_inches="tight", dpi=600)
# print(f"Number of targets: {len(binders)}")
