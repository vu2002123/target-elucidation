import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

batch_name = "D1_validation_90cp"
dataset = 2
file = f"/home/vu2002123/target-elucidation/data/interim/{batch_name}_out.csv"
# dict_file = f"/home/vu2002123/target-elucidation/data/interim/D{dataset}_mapping.tsv"

df = pd.read_csv(file)
# df_map = pd.read_csv(dict_file, sep="\t")
# mapping_dict = (
#     df_map.groupby("From")["To"].apply(lambda x: "; ".join(set(x.dropna().astype(str)))).to_dict()
# )

if dataset == 1:
    df["UNIPROT_ID"] = df["File_Name"].str.split("-").str[1]
else:
    df["UNIPROT_ID"] = df["File_Name"].str.split("_").str[0]

# df["Gene_Name"] = df["UNIPROT_ID"].map(mapping_dict).fillna("Not mapped")
df.head

compounds = set(df["Compound"].drop_duplicates())

for cp in compounds:
    OUT_DIR = Path.home() / f"target-elucidation/data/interim/{batch_name}"
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    result_file = OUT_DIR / f"{cp}_{batch_name}_full_score.csv"
    full_result_file = OUT_DIR / f"{cp}_{batch_name}_all_pocket_score.csv"
    df_result = df.query("Compound == @cp").sort_values(by="CNN_VS", ascending=False)
    df_result.to_csv(full_result_file, index=False)
    df_result = df_result.drop_duplicates(subset="UNIPROT_ID")
    df_result.to_csv(result_file, index=False)
