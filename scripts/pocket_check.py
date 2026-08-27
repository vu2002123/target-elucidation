from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


HPC_input_dir = Path.home() / "target-elucidation" / "data" / "HPC_input"


def get_docking_list(protein_list: Path | str, dataset: int) -> pd.DataFrame:
    if dataset == 1:
        protein_file = HPC_input_dir / "Dataset1" / "protein_list_plddt_filtered.csv"
        protein_df = pd.read_csv(protein_file)
        protein_df["ID"] = protein_df["Name"].str.split("-").str[1]

        directory = HPC_input_dir / "Dataset1" / "pdb"
        extension = "_fixed.pdb"
        # protein_df["PDB_path"] = str(directory) + "/" + protein_df["Name"] + extension
        protein_df["PDB_path"] = protein_df.apply(
            lambda row: str(directory / f"{row['Name']}_fixed.pdb"), axis=1
        )
    elif dataset == 2:
        protein_file = HPC_input_dir / "Dataset2" / "human_pocketome" / "AF2_combined_pocket.csv"
        protein_df = pd.read_csv(protein_file)
        protein_df["ID"] = protein_df["Name"].str.split("_").str[0]
        protein_df = protein_df[["ID", "Name", "Site", "Center_x", "Center_y", "Center_z"]]

        directory = HPC_input_dir / "Dataset2" / "AF2_combined"
        protein_df["PDB_path"] = protein_df.apply(
            lambda row: str(
                directory / f"{row['Name']}_{row['Site']}" / f"{row['Name']}_prepared.pdb"
            ),
            axis=1,
        )
    else:
        print("Invalid dataset")
        return pd.DataFrame()

    with open(HPC_input_dir / protein_list, "r") as file:
        protein_id_list = [line.strip() for line in file]

    filtered_protein_df = protein_df.query("ID in @protein_id_list")

    return filtered_protein_df


def find_nearby_pockets(group, threshold=6.0) -> pd.DataFrame:
    """
    Identifies pairs of pockets within a group that are closer than the threshold.
    """
    # 1. Extract coordinates
    coords = group[["Center_x", "Center_y", "Center_z"]].values

    # 2. Skip groups with only one pocket
    if len(coords) < 2:
        return pd.DataFrame()

    # 3. Calculate all pairwise distances
    distances = pdist(coords)

    # 4. Check if any distance is below threshold
    if np.any(distances < threshold):
        # Identify which indices are clashing
        dist_matrix = squareform(distances)
        # Find pairs (row_idx, col_idx) where distance < threshold (excluding diagonal)
        rows, cols = np.where((dist_matrix < threshold) & (dist_matrix > 0))

        # Return unique indices of pockets that are 'too close'
        clashing_indices = sorted(list(set(rows).union(set(cols))))
        return group.iloc[clashing_indices]

    return pd.DataFrame()


df_1 = get_docking_list("D1_all_IDs.txt", 1)
df_1["Dataset"] = 1

df_2 = get_docking_list("D2_combined_all_IDs.txt", 2)
df_2["Dataset"] = 2

with open(HPC_input_dir / "dataset_intersect_IDs.txt", "r") as file:
    protein_id_list = [line.strip() for line in file]

df_combined = pd.concat([df_1, df_2], axis=0)
df_combined = df_combined.query("ID in @protein_id_list").drop(columns=["PDB_path"])


clashing_pockets = df_combined.groupby("ID", group_keys=False).apply(find_nearby_pockets)

clashing_pockets["Region"] = (
    clashing_pockets["Name"].str.split("-").str[2].where(clashing_pockets["Dataset"] == 1, None)
)
clashing_pockets[["ID", "Dataset", "Name"]]
clashing_pockets[["Dataset", "Center_x", "Center_y", "Center_z"]]

keys = ["Name", "Site"]
merged = df_1.merge(
    clashing_pockets.query("Dataset == 1")[keys], on=keys, how="left", indicator=True
)

# 3. Keep only the rows found exclusively in the 'left' (original) dataframe
df_cleaned = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])


df_1.shape
len(df_1["ID"].drop_duplicates())
df_cleaned.shape
len(df_cleaned["ID"].drop_duplicates())

# df_cleaned.to_csv(HPC_input_dir / "Dataset1/protein_list_plddt_no_duplicate.csv", index=False)
