#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import UpSet, from_contents

WORK_DIR = Path.home() / "target-elucidation" / "data"
RAW_DIR = WORK_DIR / "raw"
INTERIM_DIR = WORK_DIR / "interim"


def get_bind_list(drug_name: str, dataset: int):
    if dataset == 1:
        drug_file = f"docking_{drug_name}_best_per_gene.tsv"
        sep = "\t"
    elif dataset == 2:
        drug_file = f"{drug_name}_AF2-PD_annotated_out.csv"
        sep = ","
    else:
        print("Invalid dataset")
        return set()
    drug_file_path = INTERIM_DIR / drug_file
    drug_data = pd.read_csv(drug_file_path, sep=sep)
    drug_data["CNN_VS"] = drug_data["CNN_score"] * drug_data["CNN_affinity"]
    drug_data["Gene_Name"] = drug_data["Gene_Name"].astype(str).str.strip().str.upper()
    # bind_df = (
    #     drug_data[drug_data["CNN_affinity"] >= 6]
    #     .drop_duplicates(subset=["Gene_Name"])
    #     .dropna()
    # )
    bind_df = (
        drug_data.sort_values(by="CNN_VS", ascending=False)
        .drop_duplicates(subset=["Gene_Name"])
        .reset_index(drop=True)
    )
    top5 = int(len(bind_df) * 0.2)
    bind_df = bind_df.iloc[:top5, :]
    bind_list = set(bind_df["Gene_Name"])
    print(len(bind_df))
    return bind_list


def get_DEG(cancer_type: str, fold_change: float):
    DEG_file = "DEG_" + cancer_type + "_all.csv"
    DEG_file_path = INTERIM_DIR / DEG_file
    DEG_data = pd.read_csv(DEG_file_path, sep=",")
    DEG_up_list = set(
        DEG_data.query("(log2FoldChange >= @fold_change) & (padj < 0.05)")["Gene_name"]
    )
    return DEG_up_list


def get_PROG(cancer_type: str):
    PROG_file = "survival_TCGA-" + cancer_type + "-all.csv"
    PROG_file_path = INTERIM_DIR / PROG_file
    PROG_data = pd.read_csv(PROG_file_path, sep=",")
    PROG_list_KM = set(PROG_data.query("km_p_value < 0.05")["Gene_name"])
    PROG_list_COX = set(PROG_data.query("cox_p_value < 0.05")["Gene_name"])
    return PROG_list_KM, PROG_list_COX


PCP_D1 = get_bind_list("PCP", 1)
PCP_D2 = get_bind_list("PCP", 2)
PCP_ALL = PCP_D1.union(PCP_D2)

NPCP_D1 = get_bind_list("PCPN", 1)
NPCP_D2 = get_bind_list("PCPN", 2)
NPCP_ALL = NPCP_D1.union(NPCP_D2)

# with open("/home/vu2002123/target-elucidation/data/interim/NPCP_bind_list.txt", "w") as file:
#     for gene in NPCP_ALL:
#         file.write(f"{gene}\n")

LUAD_DEG = get_DEG("LUAD", 1)
PAAD_DEG = get_DEG("PAAD", 1)
DEG_ALL = LUAD_DEG.union(PAAD_DEG)

LUAD_KM, LUAD_COX = get_PROG("LUAD")
PAAD_KM, PAAD_COX = get_PROG("PAAD")
KM_ALL = LUAD_KM.union(PAAD_KM)
COX_ALL = LUAD_COX.union(PAAD_COX)

contents_all = {
    "CY001": PCP_ALL,
    # "ND-CY001": NPCP_ALL,
    "DEG": DEG_ALL,
    "PROG_KM": KM_ALL,
    "PROG_COX": COX_ALL,
}

samples = from_contents(contents_all)

usp = UpSet(
    samples,
    orientation="horizontal",
    subset_size="count",
    show_counts="{:d}",
    facecolor="black",
    sort_categories_by="-input",
    min_degree=3,
    include_empty_subsets=True,
)
usp.style_subsets(
    present=["CY001", "DEG", "PROG_KM", "PROG_COX"],
    facecolor="#e02b35",
)
usp.plot()
fig = plt.gcf()  # gcf = Get Current Figure
fig.set_size_inches(8, 6)

plot_name = "Intersection of CY001 targets"
plt.title(plot_name, pad=20)
plot_path = (
    Path.home() / "target-elucidation" / "reports" / "figures" / "CY001_only_top20percent.png"
)
plt.savefig(plot_path, dpi=600)

all_true = set(samples.loc[(True,) * len(samples.index.levels)]["id"])
print(all_true)

outfile = INTERIM_DIR / "PCP_only_intersection_list_20percent.txt"
with open(outfile, "w") as f:
    f.write("\n".join(str(i) for i in all_true))
