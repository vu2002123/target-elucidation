#!/usr/bin/env python3

# standard library modules
import sys, errno, re, json, ssl
from urllib import request
from urllib.error import HTTPError
from time import sleep
import pandas as pd

with open("/home/vu2002123/target-elucidation/data/raw/D1_validation_90cp.txt", "r") as file:
    drugs = [line.strip() for line in file]

context = ssl._create_unverified_context()
MAX_RETRIES = 3
BASE_DELAY = 10

ligand_dict = {}

for i, id in enumerate(drugs, start=1):
    ligand_url = f"https://www.ebi.ac.uk/pdbe/api/v2/pdb/compound/summary/{id}"
    for attempt in range(MAX_RETRIES):
        try:
            req1 = request.Request(ligand_url, headers={"Accept": "application/json"})
            res1 = request.urlopen(req1, context=context)
            payload1 = json.loads(res1.read().decode())
            ligand_dict[id] = payload1[id][0].get("smiles")[0].get("name")
            print(f"[{i}/{len(drugs)}] Got information for {id}")
            sleep(1)
            break
        except HTTPError as e:
            if e.code == 404:
                print(f"[{i}/{len(drugs)}] {id} has no information")
                sleep(1)
                break
            elif e.code == 503:
                if attempt < MAX_RETRIES - 1:
                    wait_time = BASE_DELAY * (2**attempt)
                    print(f"[{i}/{len(drugs)}] Retrying for {id} due to code 503")
                    sleep(wait_time)
                    continue
                else:
                    print(f"[{i}/{len(drugs)}] {id} failed due to code 503, skipping")
                    sleep(1)
                    break
            else:
                print(f"[{i}/{len(drugs)}] {id} failed due to code {e.code}")
                sleep(1)
                break

len(ligand_dict)
smiles_df = pd.DataFrame.from_dict(ligand_dict, orient="index")
smiles_df.index.name = "code_name"
smiles_df.columns = ["SMILES"]
smiles_df.to_csv("/home/vu2002123/target-elucidation/data/raw/D1_validation_90cp.csv")


with open("/home/vu2002123/target-elucidation/data/interim/pdb_ligand.json", "w") as file:
    json.dump(ligand_dict, file, indent=4)
