#!/usr/bin/env python3

# standard library modules
import sys, errno, re, json, ssl
from urllib import request
from urllib.error import HTTPError
from time import sleep
import pandas as pd

domain_df = pd.read_csv("/home/vu2002123/target-elucidation/data/raw/all_domain.csv")
pdb_df = pd.read_csv(
    "/home/vu2002123/target-elucidation/data/raw/pdb_chain_uniprot.csv", skiprows=1
)

domain_ids = set(domain_df["ID"])
domain_ids = [id.upper() for id in domain_ids]

pdb_uniprot_ids = set(pdb_df["SP_PRIMARY"])
human_pdb_wdomain = set(domain_ids).intersection(pdb_uniprot_ids)
len(human_pdb_wdomain)

context = ssl._create_unverified_context()
MAX_RETRIES = 3
BASE_DELAY = 10

ligand_dict = {}
ligand_site_dict = {}

for i, id in enumerate(human_pdb_wdomain, start=1):
    ligand_url = f"https://www.ebi.ac.uk/pdbe/api/v2/uniprot/ligands/{id}"
    ligand_site_url = f"https://www.ebi.ac.uk/pdbe/api/v2/uniprot/ligand_sites/{id}"
    for attempt in range(MAX_RETRIES):
        try:
            req1 = request.Request(ligand_url, headers={"Accept": "application/json"})
            res1 = request.urlopen(req1, context=context)
            payload1 = json.loads(res1.read().decode())
            req2 = request.Request(ligand_site_url, headers={"Accept": "application/json"})
            res2 = request.urlopen(req2, context=context)
            payload2 = json.loads(res2.read().decode())
            ligand_dict[id] = payload1[id]
            ligand_site_dict[id] = payload2[id]
            print(f"[{i}/{len(human_pdb_wdomain)}] Got information for {id}")
            sleep(1)
            break
        except HTTPError as e:
            if e.code == 404:
                print(f"[{i}/{len(human_pdb_wdomain)}] {id} has no information")
                sleep(1)
                break
            elif e.code == 503:
                if attempt < MAX_RETRIES - 1:
                    wait_time = BASE_DELAY * (2**attempt)
                    print(f"[{i}/{len(human_pdb_wdomain)}] Retrying for {id} due to code 503")
                    sleep(wait_time)
                    continue
                else:
                    print(f"[{i}/{len(human_pdb_wdomain)}] {id} failed due to code 503, skipping")
                    sleep(1)
                    break
            else:
                print(f"[{i}/{len(human_pdb_wdomain)}] {id} failed due to code {e.code}")
                sleep(1)
                break

len(ligand_dict)
len(ligand_site_dict)

with open("/home/vu2002123/target-elucidation/data/interim/pdb_ligand.json", "w") as file:
    json.dump(ligand_dict, file, indent=4)
with open("/home/vu2002123/target-elucidation/data/interim/pdb_ligand_site.json", "w") as file:
    json.dump(ligand_site_dict, file, indent=4)
