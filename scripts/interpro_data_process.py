from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path.home() / "target-elucidation" / "data" / "raw"
annotation_file = DATA_DIR / "uniprot_human_count_total.tsv"
id_file = DATA_DIR / "uniprot_ids.txt"

df = (
    pd.read_csv(annotation_file, sep="\t", header=None)
    .rename(
        columns={
            0: "Accession",
            1: "Name",
            2: "Source_database",
            3: "Type",
            4: "Integrated",
            5: "Member_databases",
            6: "GO_terms",
            7: "Protein_accession",
            8: "Protein_length",
            9: "Entry_protein_locations",
        }
    )
    .drop(columns=[10])
)
df = df.dropna(subset="Protein_accession")

with open(id_file, "r") as file:
    ids = [line.strip().lower() for line in file]

df_grouped = (
    df.query("Type == 'domain'")
    .groupby("Protein_accession")
    .agg({"Accession": list})
    .reset_index()
)

domain_dict = {}
domain_dict
current_root = None
domain_file = DATA_DIR / "interpro_tree.txt"
with open(domain_file, "r") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        # Split the line by '::' and filter out empty strings
        parts = [part for part in line.split("::") if part]
        if len(parts) < 2:
            continue
        if line.startswith("--"):
            # It is a child domain
            domain_id = parts[0].lstrip("-")
            domain_name = parts[1]
            # Append to the current active root node
            if current_root and current_root in domain_dict:
                domain_dict[current_root]["children"][domain_id] = domain_name
        else:
            # It is a root domain
            domain_id = parts[0]
            domain_name = parts[1]
            current_root = domain_id
            # Initialize the root in the dictionary
            domain_dict[current_root] = {"name": domain_name, "children": {}}

child_to_parent = {}
for parent_id, data in domain_dict.items():
    for child_id in data["children"].keys():
        child_to_parent[child_id] = parent_id

filtered_domain = {}
for _, row in df_grouped.iterrows():
    protein = row.Protein_accession
    domains = list(row.Accession)
    kept_domains = []
    for domain in domains:
        parent_id = child_to_parent.get(domain)
        if parent_id in domains:
            continue
        else:
            kept_domains.append(domain)
    filtered_domain[protein] = kept_domains

df_grouped["Accession_filtered"] = df_grouped["Protein_accession"].map(filtered_domain)

present_id = set(df_grouped["Protein_accession"])
missing_id = list(set(ids) - present_id)

df_missing = pd.DataFrame(
    {
        "Protein_accession": missing_id,
        "Accession": [[] for _ in range(len(missing_id))],
        "Accession_filtered": [[] for _ in range(len(missing_id))],
    }
)
df_merged = pd.concat([df_grouped, df_missing], ignore_index=True)

df_merged["Count"] = df_merged["Accession_filtered"].str.len()

np.sum(df_merged["Count"])

len(present_id)

df_merged["Accession_filtered"].explode().value_counts().head(10)

df_domain = pd.DataFrame(columns=["Name", "ID", "Domain", "Start", "End", "Length"])

for query_id in present_id:
    domains = df_merged.query("Protein_accession == @query_id")["Accession_filtered"]
    if len(domains) > 0:
        for domain in domains:
            filtered_df = df.query("Protein_accession == @query_id and Accession == @domain")
            for _, row in filtered_df.iterrows():
                protein_id = row.Protein_accession
                domain_id = row.Accession
                positions = str(row.Entry_protein_locations).split(",")
                for position in positions:
                    start = position.split("..")[0]
                    end = position.split("..")[1]
                    whole_name = "_".join([protein_id, domain_id, start, end])
                    new_row = [
                        {
                            "Name": whole_name,
                            "ID": protein_id,
                            "Domain": domain_id,
                            "Start": int(start),
                            "End": int(end),
                            "Length": int(end) - int(start),
                        }
                    ]
                    df_new = pd.DataFrame(new_row)
                    df_domain = pd.concat([df_domain, df_new], ignore_index=True)
                    print(
                        f"Domain {domain_id} is located in protein {protein_id} from position {start} to {end}"
                    )

df_domain.shape
out_file = DATA_DIR / "all_domain.csv"
df_domain.to_csv(out_file, index=False)
