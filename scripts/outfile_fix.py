import pandas as pd
from pathlib import Path

file = "/home/vu2002123/target-elucidation/data/interim/BMX_D2_out.csv"
df = pd.read_csv(file)

df["ID_new"] = (
    df["ID"]
    .str.cat(df["Site"].astype(str), sep="_")
    .str.cat(df["Compound"].str.split("_").str[0], sep="_")
)

df["Site_new"] = df["Compound"].str.split("_").str[1:3].str.join("_")

df["Compound_new"] = df["Compound"].str.split("_").str[3]
df["UNIPROT_ID"] = df["ID_new"].str.split("_").str[0]

gene_file = "/home/vu2002123/target-elucidation/data/interim/BMX_D2_master.csv"
df_gene = pd.read_csv(gene_file)
gene_dict = df_gene.set_index("ID")["Gene_Name"].to_dict()

df["Gene_Name"] = df["UNIPROT_ID"].map(gene_dict)
df = df.drop(columns=["ID", "Site", "Compound", "minimizedAffinity"]).rename(
    columns={"ID_new": "ID", "Site_new": "Site", "Compound_new": "Compound"}
)

compounds = set(df["Compound"].drop_duplicates())

file_d1 = "/home/vu2002123/target-elucidation/data/interim/BMX_D1_out.csv"
df_d1 = pd.read_csv(file_d1)
df_d1["UNIPROT_ID"] = df_d1["ID"].str.split("-").str[1]
gene_file_d1 = "/home/vu2002123/target-elucidation/data/interim/BMX_D1_master.csv"
df_gene_d1 = pd.read_csv(gene_file_d1)
gene_dict_d1 = df_gene_d1.set_index("ID")["Gene_Name"].to_dict()
df_d1["Gene_Name"] = df_d1["UNIPROT_ID"].map(gene_dict_d1)
df_d1 = df_d1.drop(columns=["minimizedAffinity"])


for cp in compounds:
    df_result = df_d1.query("Compound == @cp").sort_values(by="CNN_VS", ascending=False)
    gene_list = df_result.iloc[:101, -1].to_list()
    result_file = f"/home/vu2002123/target-elucidation/data/processed/{cp}_D1_full_score.csv"
    list_file = f"/home/vu2002123/target-elucidation/data/processed/{cp}_D1_target_list.txt"
    df_result.to_csv(result_file, index=False)
    with open(list_file, "w") as f:
        f.write("\n".join(str(i) for i in gene_list))
