import pandas as pd
import json

with open("/home/vu2002123/target-elucidation/data/interim/pdb_ligand.json", "r") as file:
    dict_ligand = json.load(file)
with open("/home/vu2002123/target-elucidation/data/interim/pdb_ligand_site.json", "r") as file:
    dict_site = json.load(file)
domain_df = pd.read_csv("/home/vu2002123/target-elucidation/data/interim/all_domain_filtered.csv")

target_df = pd.read_csv("/home/vu2002123/target-elucidation/data/raw/90cp_targets.csv")
with open("/home/vu2002123/target-elucidation/data/HPC_input/DS_Vu_all_ids.txt", "r") as file:
    all_proteins = [line.strip() for line in file]
targets = set(target_df["Target"]).intersection(set(all_proteins))

pdb_dict = {}
site_count = 0
for id in dict_ligand.keys():
    # drug_ligand_present = False
    # drug_ligand_list = []
    # for lig in dict_ligand.get(id):
    #     for ccd, details in lig.items():
    #         is_drug = "drug-like" in details.get("acts_as", [])
    #         interacts = details.get("directly_interacts", False)
    #         if is_drug and interacts:
    #             drug_ligand_present = True
    #             drug_ligand_list.append(ccd)
    if id in targets:
        results = dict_site.get(id).get("data")
        for lig in results:
            accession = lig.get("accession")
            matching_compounds = target_df.query("Target == @id")["Compound"].tolist()
            if accession in matching_compounds:
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
                site_count += 1

site_count

domain_count = 0
domain_dict = {}
for id in pdb_dict.keys():
    domain_filtered = domain_df.query("ID == @id")
    for ccd in pdb_dict[id].keys():
        interacting_residue = pdb_dict[id][ccd]
        for _, row in domain_filtered.iterrows():
            domain = row.Domain
            start = row.Start
            end = row.End
            has_match = any(start <= residue <= end for residue in interacting_residue)
            if has_match:
                if id not in domain_dict.keys():
                    domain_dict[id] = {}
                if ccd not in domain_dict[id].keys():
                    domain_dict[id][ccd] = domain
                    domain_count += 1
                else:
                    print(f"{id} + {ccd} + {domain}")

domain_df.value_counts(subset="Domain")["IPR000387"]
domain_count

with open(
    "/home/vu2002123/target-elucidation/data/interim/90cp_domain.json", "w", encoding="utf-8"
) as f:
    json.dump(domain_dict, f, indent=4)
