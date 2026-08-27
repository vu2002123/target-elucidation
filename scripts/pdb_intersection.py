from pathlib import Path

DATA_DIR = Path("/home/vu2002123/target-elucidation/data/")
HPC_INPUT = DATA_DIR / "HPC_input"
RAW_DIR = DATA_DIR / "raw"

D1_all_file = HPC_INPUT / "D1_all_IDs.txt"
D1_filtered_file = HPC_INPUT / "D1_filtered_ID.txt"
D2_all_file = HPC_INPUT / "D2_combined_all_IDs.txt"
DRUG_all_file = RAW_DIR / "drugport_ids.txt"
DRUG_filtered_file = RAW_DIR / "drugport_ids_filtered.txt"

with open(D1_all_file, "r") as file:
    D1_all = set([line.strip() for line in file])

with open(D1_filtered_file, "r") as file:
    D1_filtered = set([line.strip() for line in file])

with open(D2_all_file, "r") as file:
    D2_all = set([line.strip() for line in file])

with open(DRUG_all_file, "r") as file:
    DRUG_all = set([line.strip() for line in file])

with open(DRUG_filtered_file, "r") as file:
    DRUG_filtered = set([line.strip() for line in file])

len(D2_all.intersection(DRUG_filtered))
