from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path.home() / "target-elucidation" / "data" / "raw"
FIG_DIR = Path.home() / "target-elucidation" / "reports" / "figures"
HPC_INPUT = Path.home() / "target-elucidation" / "data" / "HPC_input"

domain_file = DATA_DIR / "all_domain.csv"
domain_df = pd.read_csv(domain_file)
domain_df_100 = domain_df.loc[domain_df["Length"] <= 100]

AF_DIR = HPC_INPUT / "AF_v6"
pdb_files = list(AF_DIR.glob("*v6.pdb"))
AF_ids = [file_name.stem.lower().split("-")[1] for file_name in pdb_files]
AF_ids = set(AF_ids)
domain_df_AF = domain_df.query("ID in @AF_ids")
domain_df_AF_100 = domain_df_AF.loc[domain_df["Length"] <= 100]

domain_df_AF.columns
domain_ids = set(domain_df_AF["ID"])

domain_freq = domain_df_AF["Domain"].value_counts()
total = domain_freq.sum()
domain_freq_top = domain_freq.iloc[:10]
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=domain_freq_top.index, y=domain_freq_top.values, color="red")
percentages = [f"{(bar.get_height() / total) * 100:.1f}%" for bar in ax.containers[0]]
ax.bar_label(ax.containers[0], labels=percentages, padding=3)
plt.title("Domain Frequency")
plt.xlabel("Domain")
plt.ylabel("Count")
plt.savefig(FIG_DIR / "domain_frequency", bbox_inches="tight")
plt.close("all")

D2_file = HPC_INPUT / "Dataset2" / "human_pocketome" / "AF2_PD_domain.tsv"
D2_df = pd.read_csv(D2_file, sep="\t")
D2_df["Start"] = D2_df["domain"].str.split("_").str[2].astype(int)
D2_df["End"] = D2_df["domain"].str.split("_").str[3].astype(int)
D2_df["Length"] = D2_df["End"] - D2_df["Start"]
D2_df_100 = D2_df.loc[D2_df["Length"] <= 100]

D2_ids = set(D2_df["domain"].str.split("_").str[0])
D2_ids = set([str(ids).lower() for ids in D2_ids])


D2_df_100.columns
sns.kdeplot(data=D2_df_100, x="Length", y="prank rescore", fill=True)
plt.savefig(FIG_DIR / "druggability_correlation", bbox_inches="tight")
plt.close("all")

fig, axes = plt.subplots(3, 2, figsize=(24, 16))
sns.histplot(domain_df, x="Length", bins=30, fill=False, color="blue", ax=axes[0, 0])
axes[0, 0].set_title("Full Distribution")
axes[0, 0].set_xticks(np.arange(0, 3100, 200))
axes[0, 0].bar_label(axes[0, 0].containers[0], padding=3, rotation=90)
sns.histplot(domain_df_100, x="Length", bins=30, fill=False, color="red", ax=axes[0, 1])
axes[0, 1].set_title("Zoomed View (Length <= 100)")
axes[0, 1].set_xticks(np.arange(0, 110, 10))
axes[0, 1].bar_label(axes[0, 1].containers[0], padding=3, rotation=90)
sns.histplot(domain_df_AF, x="Length", bins=30, fill=False, color="blue", ax=axes[1, 0])
axes[1, 0].set_title("Full Distribution AF")
axes[1, 0].set_xticks(np.arange(0, 3100, 200))  # Widened increments to prevent text overlap
axes[1, 0].bar_label(axes[1, 0].containers[0], padding=3, rotation=90)
sns.histplot(
    domain_df_AF_100,
    x="Length",
    bins=30,
    fill=False,
    color="red",
    ax=axes[1, 1],
)
axes[1, 1].set_title("Zoomed View (Length <= 100) AF")
axes[1, 1].set_xticks(np.arange(0, 110, 10))
axes[1, 1].bar_label(axes[1, 1].containers[0], padding=3, rotation=90)
sns.histplot(
    D2_df,
    x="Length",
    bins=30,
    fill=False,
    color="blue",
    ax=axes[2, 0],
)
axes[2, 0].set_title("Full Distribution D2")
axes[2, 0].set_xticks(np.arange(0, 3100, 200))
axes[2, 0].bar_label(axes[2, 0].containers[0], padding=3, rotation=90)
sns.histplot(
    D2_df_100,
    x="Length",
    bins=30,
    fill=False,
    color="red",
    ax=axes[2, 1],
)
axes[2, 1].set_title("Zoomed View (Length <= 100) D2")
axes[2, 1].set_xticks(np.arange(0, 110, 10))
axes[2, 1].bar_label(axes[2, 1].containers[0], padding=3, rotation=90)
plt.savefig(FIG_DIR / "domain_length_distribution.png", bbox_inches="tight")
plt.close("all")
