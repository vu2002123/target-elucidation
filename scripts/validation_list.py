import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib

WORK_DIR = Path("/home/vu2002123/target-elucidation/data/")
RAW_DIR = WORK_DIR / "raw"
INTERIM_DIR = WORK_DIR / "interim"
HPC_INPUT = WORK_DIR / "HPC_input"

drugs = ["Curcumin", "Hydroxychloroquine"]

activity_threshold = 10000
activity_threshold_um = activity_threshold / 1000
for drug in drugs:
    bind_file = RAW_DIR / f"bindingDB/{drug}_bindingDB.tsv"
    out_bind_file = RAW_DIR / f"bindingDB/{drug}_filtered.tsv"
    df_bind = pd.read_csv(bind_file, sep="\t")

    df_bind_filtered = df_bind.dropna(subset=["Ki (nM)", "Kd (nM)"], how="all")
    df_bind_filtered = df_bind_filtered.replace(r">.*", None, regex=True)
    for col in ["Ki (nM)", "Kd (nM)"]:
        df_bind_filtered[col] = (
            df_bind_filtered[col].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
        )
    df_final = df_bind_filtered[
        (df_bind_filtered["Ki (nM)"] <= activity_threshold)
        | (df_bind_filtered["Kd (nM)"] <= activity_threshold)
    ]

    df_final_filtered = df_final.query(
        "`Target Source Organism According to Curator or DataSource` == 'Homo sapiens'"
    )
    df_final_filtered.to_csv(out_bind_file, index=False, sep="\t")
    df_final_filtered.iloc[:, 44]
    binders = set(df_final_filtered.iloc[:, 44].astype(str))
    # binders = df_final_filtered.iloc[:,44].drop_duplicates().to_list()

    pubchem_file = RAW_DIR / f"pubchem/{drug}_pubchem.csv"
    out_pubchem_file = RAW_DIR / f"pubchem/{drug}_filtered.csv"
    out_list_file = RAW_DIR / f"pubchem/{drug}_filtered.txt"
    out_list_total_file = RAW_DIR / f"pubchem/{drug}_filtered_total.txt"
    df_pubchem = pd.read_csv(pubchem_file)
    # df_filtered = df_pubchem.query(
    #     "Activity == 'Active' and (Activity_Type == 'Ki' or Activity_Type == 'Kd' or Activity_Type == 'Potency') and (Activity_Qualifier != '>') and Activity_Value <= @activity_threshold_um and Taxonomy_ID == 9606"
    # )
    df_filtered = df_pubchem.query(
        "Activity == 'Active' and (Activity_Type == 'Ki' or Activity_Type == 'Kd') and (Activity_Qualifier != '>') and Activity_Value <= @activity_threshold_um and Taxonomy_ID == 9606"
    )
    pubchem_binders = set(df_filtered["Representative_Protein_Accession"].astype(str))
    total_binders = pubchem_binders.union(binders)
    len(total_binders)
    binders_pcp = [id.lower() for id in total_binders]

    with open(out_list_file, "w") as file:
        for id in pubchem_binders:
            file.write(f"{id}\n")
    with open(out_list_total_file, "w") as file:
        for id in total_binders:
            file.write(f"{id}\n")


df_filtered.iloc[15]
