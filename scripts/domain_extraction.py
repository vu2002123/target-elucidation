from pathlib import Path
import re
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Dice import extract


INPUT_DIR = Path.home() / "docking_refined" / "input"
AF_DIR = INPUT_DIR / "AF_v6" / "raw"
DOMAIN_DIR = INPUT_DIR / "AF_v6" / "domain"
POCKET_DIR = INPUT_DIR / "AF_v6" / "pockets"

input_file = (
    "/home/a00106/bpsvupq.ps13/docking_refined/input/AF_v6/AF-P14416-F1-model_v6.pdb"
)

pdb_files = AF_DIR.glob("*.pdb")
domain_df = pd.read_csv(INPUT_DIR / "all_domain_filtered.csv")
domain_ids = set(domain_df["ID"])

pattern = re.compile(r"AF-([A-Z0-9]+)-F(\d+)")

# Dict to store the final output: {id: highest_f_number}
result_dict = {}

for path in pdb_files:
    match = pattern.search(path.name)
    if match:
        file_id, f_num_str = match.groups()
        f_num = int(f_num_str)
        if file_id in domain_ids:
            if file_id not in result_dict or f_num > result_dict[file_id]:
                result_dict[file_id] = f_num

print(result_dict)

domain_df["Max_fragment"] = domain_df["ID"].map(result_dict)

y = domain_df["End"] + (1400 - domain_df["Length"]) / 2
x_calculated = np.round((y - 1400) / 200)
domain_df["Fragment"] = np.where(domain_df["End"] >= 1400, x_calculated, 1)
domain_df["File_End"] = domain_df["Fragment"] * 200 + 1200
domain_df["End_diff"] = domain_df["File_End"] - domain_df["End"]
domain_df["Chosen_fragment"] = np.where(
    domain_df["Fragment"] > domain_df["Max_fragment"],
    domain_df["Max_fragment"],
    domain_df["Fragment"],
)
domain_df["Source_file"] = (
    "AF-"
    + domain_df["ID"]
    + "-F"
    + domain_df["Chosen_fragment"].astype(int).astype(str)
    + "-model_v6.pdb"
)
domain_df.sort_values("End", ascending=False).iloc[1]

# for _, row in domain_df.iterrows():
#     id = row.ID
#     fragment = int(row.Chosen_fragment)
#     file_offset = 200 * (fragment - 1)
#     start_residue = int(row.Start)
#     end_residue = int(row.End)
#     in_file_start = start_residue - file_offset
#     in_file_end = end_residue - file_offset
#     input_file = AF_DIR / f"AF-{id}-F{fragment}-model_v6.pdb"
#     output_file = DOMAIN_DIR / f"{id}_{row.Domain}_{start_residue}_{end_residue}.pdb"
#     if not input_file.exists():
#         continue
#     parser = PDBParser(QUIET=True)
#     structure = parser.get_structure(id, input_file)
#     extract(
#         structure=structure,
#         chain_id="A",
#         start=in_file_start,
#         end=in_file_end,
#         filename=str(output_file),
#     )
#     print(f"Extraction complete for {id}")
