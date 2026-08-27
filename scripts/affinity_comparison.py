import os
import pandas as pd
from pathlib import Path

WORK_DIR = Path("/home/vu2002123/target-elucidation/data/")
INTER_DIR = WORK_DIR / "interim"
PROC_DIR = WORK_DIR / "processed"

os.chdir(INTER_DIR)

!ls
bind_file = INTER_DIR / "PCP_bindingdb.csv"
bind_df = pd.read_csv(bind_file)

PCP_d1 = INTER_DIR / "docking_PCP_best_per_gene.tsv"
PCP_df1 = pd.read_csv(PCP_d1, sep="\t")

PCP_d2 = INTER_DIR / "PCP_AF2-PD_annotated_out.csv"
PCP_df2 = pd.read_csv(PCP_d2)
print(PCP_df2.columns)

known_list = set(bind_df["Gene_Name"])
PCP_df1_known = PCP_df1.loc[PCP_df1["Gene_Name"].isin(known_list)]
PCP_df2_known = PCP_df2.loc[PCP_df2["Gene_Name"].isin(known_list)]
print(PCP_df2_known.head)

