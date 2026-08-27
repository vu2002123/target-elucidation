from numpy import dtype
import pandas as pd
import mygene

mg = mygene.MyGeneInfo()

d1_file = "/home/vu2002123/target-elucidation/data/raw/Dataset1/Dataset1_ID_list.txt"
with open(d1_file, "r") as file:
    d1_ids = [line.strip() for line in file]

d2_file = "/home/vu2002123/target-elucidation/data/raw/Dataset2/human_pocketome/AF2_PD_ID_list.txt"
with open(d2_file, "r") as file:
    d2_ids = [line.strip() for line in file]

# Querying for Entrez ID, Symbol, and Name
results: pd.DataFrame = mg.querymany(
    d1_ids,
    scopes="uniprot",
    fields="symbol,name,interpro,go.CC",
    species="human",
    as_dataframe=True,
)

target = "GO:0005886"

mask = results["go.CC"].apply(
    lambda x: any(d.get("id") == target for d in x) if isinstance(x, list) else False
)

results["Membrane located"] = mask

results["notfound"].value_counts()

filtered_results = results[results["notfound"].isna()]
filtered_results = (
    filtered_results.reset_index().drop_duplicates(subset="query", keep="first").set_index("query")
)
membrane_ids = list(filtered_results[filtered_results["Membrane located"]].index)

sum(filtered_results["Membrane located"])
outfile = "/home/vu2002123/target-elucidation/data/interim/D1_membrane_list.txt"
with open(outfile, "w") as file:
    for id in membrane_ids:
        file.write(f"{id}\n")


filtered_results["interpro"].value_counts()

filtered_results.shape


go_list = results.iloc[0, 6]
