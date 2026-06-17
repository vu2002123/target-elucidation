import pandas as pd
import json

with open("/home/vu2002123/target-elucidation/data/interim/pdb_ligand.json", "r") as file:
    dict_ligand = json.load(file)
with open("/home/vu2002123/target-elucidation/data/interim/pdb_ligand_site.json", "r") as file:
    dict_site = json.load(file)
domain_df = pd.read_csv("/home/vu2002123/target-elucidation/data/raw/all_domain.csv")
domain_df["ID"] = domain_df["ID"].str.upper()
domain_df["Length"] = domain_df["End"] + 1 - domain_df["Start"]
domain_df["Name"] = domain_df[["ID", "Domain", "Start", "End"]].astype(str).agg("_".join, axis=1)
domain_df["File_name"] = domain_df["Name"] + ".pdb"

pdb_dict = {}
for id in dict_ligand.keys():
    drug_ligand_present = False
    drug_ligand_list = []
    for lig in dict_ligand.get(id):
        for ccd, details in lig.items():
            is_drug = "drug-like" in details.get("acts_as", [])
            interacts = details.get("directly_interacts", False)
            if is_drug and interacts:
                drug_ligand_present = True
                drug_ligand_list.append(ccd)
    if drug_ligand_present:
        results = dict_site.get(id).get("data")
        for lig in results:
            accession = lig.get("accession")
            if accession in drug_ligand_list:
                interacting_residue = set()
                for residue in lig.get("residues"):
                    start = int(residue.get("startIndex"))
                    end = int(residue.get("endIndex"))
                    if start == end:
                        interacting_residue.add(start)
                    else:
                        interacting_residue.update(range(int(start), int(end) + 1))
                if id not in pdb_dict.keys():
                    pdb_dict[id] = {}
                pdb_dict[id][accession] = list(interacting_residue)


interacting_domains = []
for id in pdb_dict.keys():
    domain_filtered = domain_df.query("ID == @id")
    all_interacting_residues = []
    for ccd in pdb_dict[id].keys():
        interacting_residue = pdb_dict[id][ccd]
        all_interacting_residues.extend(interacting_residue)
    for _, row in domain_filtered.iterrows():
        domain = row.Domain
        start = row.Start
        end = row.End
        has_match = any(start <= residue <= end for residue in all_interacting_residues)
        if has_match:
            interacting_domains.append(domain)

interacting_domains = set(interacting_domains)
len(interacting_domains)

domain_df_filtered = domain_df.query("Domain in @interacting_domains")
domain_df_filtered.shape
len(domain_df_filtered["ID"].drop_duplicates())

with open(
    "/home/vu2002123/target-elucidation/data/raw/pubchem/Prochlorperazine_filtered.txt", "r"
) as file:
    PCP_targets = [line.strip() for line in file]

present_targets = set(domain_df_filtered["ID"].drop_duplicates()).intersection(set(PCP_targets))
len(present_targets)

domain_df_filtered.value_counts("Domain")[:10]
domain_df_filtered.query("ID == 'P01116'").iloc[0]

domain_df_filtered.to_csv(
    "/home/vu2002123/target-elucidation/data/interim/all_domain_filtered.csv", index=False
)

domain_df_filtered.sort_values(by="Length", ascending=False).iloc[0]
