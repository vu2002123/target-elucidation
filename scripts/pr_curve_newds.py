import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
)


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


batch_name = "DS_validation_TKI"
drugs = [
    "Afatinib",
    "Ceritinib",
    "Crizotinib",
    "Erlotinib",
    "Gefitinib",
    "Ruxolitinib",
    "Sunitinib",
]
# colors = ["#F0C571", "#59A89C", "#E02B35"]
weight = 1.0
cmap = plt.get_cmap("tab10")
plt.figure(figsize=(9, 6))

for i, drug in enumerate(drugs):
    binder_file = f"/home/vu2002123/target-elucidation/data/raw/pubchem/{drug}_filtered.txt"
    with open(binder_file, "r") as file:
        binders = [line.strip() for line in file]

    if drug == "Prochlorperazine":
        df_dock = load_scores_newds(drug_name="PCP", batch_name="DS_PCP", weight=weight)
        df_dock["is_target"] = df_dock["UNIPROT_ID"].isin(binders).astype(int)
    else:
        df_dock = load_scores_newds(drug_name=drug, batch_name=batch_name, weight=weight)
        df_dock["is_target"] = df_dock["UNIPROT_ID"].isin(binders).astype(int)

    # df_dock.loc[:19,["UNIPROT_ID","CNNscore","minimizedAffinity"]]
    # df_dock.query("is_target == True")

    y_true = df_dock["is_target"]
    y_scores = df_dock["Consensus"]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    current_color = cmap(i)
    plt.plot(fpr, tpr, color=current_color, lw=2, label=f"{drug} (AUC = {roc_auc:.3f})")

save_path = "/home/vu2002123/target-elucidation/reports/figures/TKI_drugs_auc_newds.png"
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Selection")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)", fontsize=12)
plt.ylabel("True Positive Rate (Recall)", fontsize=12)
plt.title("Dataset ROC: TKI", fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig(save_path, bbox_inches="tight", dpi=600)
plt.close("all")

# precision, recall, _ = precision_recall_curve(y_true, y_scores)
# avg_precision = average_precision_score(y_true, y_scores)
# baseline = y_true.sum() / len(y_true)  # Random classifier baseline
#
# # --- Plotting ---
# plt.plot(
#     recall,
#     precision,
#     color="#F0C571",
#     lw=2,
#     label=f"PR Curve for {drug} (AP = {avg_precision:.3f})",
# )
# plt.axhline(
#     y=baseline,
#     color="navy",
#     lw=1,
#     linestyle="--",
#     label=f"Random Selection (Baseline = {baseline:.3f})",
# )
#
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("Recall (True Positive Rate)", fontsize=12)
# plt.ylabel("Precision (Positive Predictive Value)", fontsize=12)
# plt.title(
#     f"Combined Dataset Precision-Recall Curve: ({drug})",
#     fontsize=14,
# )
# plt.legend(loc="upper right", fontsize=10)  # Changed to lower left for PR layout
# plt.grid(True, linestyle=":", alpha=0.6)
#
# save_path = f"/home/vu2002123/target-elucidation/reports/figures/{batch_name}_{drug}_pr_curve.png"
# plt.savefig(save_path, bbox_inches="tight", dpi=600)
# plt.close("all")
