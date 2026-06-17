import pandas as pd
from posebusters import PoseBusters

drug = "Ruxolitinib"
batch_name = "LUAD_D2"
dataset = 2
result_file = (
    f"/home/vu2002123/target-elucidation/data/interim/{drug}_{batch_name}_all_pocket_score.csv"
)
out_file = f"/home/vu2002123/target-elucidation/data/processed/{drug}_{batch_name}_posebuster.csv"

new_result_df = pd.read_csv(out_file)
percent_pass = len(new_result_df.query("Passes == 22")) / len(new_result_df) * 100
print(percent_pass)


result_df = pd.read_csv(result_file).drop(columns=["Unnamed: 0"])
if dataset == 1:
    result_df["SDF_path"] = result_df["File_Name"].map(
        lambda x: f"/home/vu2002123/target-elucidation/data/raw/{batch_name}_2/{'_'.join(x.split('_')[:2])}/{x}"
    )
    result_df["PDB_path"] = result_df["File_Name"].map(
        lambda x: f"/home/vu2002123/target-elucidation/data/raw/Dataset1/pdb/{'_'.join(x.split('_')[:2])}.pdb"
    )
else:
    result_df["SDF_path"] = result_df["File_Name"].map(
        lambda x: f"/home/vu2002123/target-elucidation/data/raw/{batch_name}_2/{'_'.join(x.split('_')[:4])}/{x}"
    )
    result_df["PDB_path"] = result_df["File_Name"].map(
        lambda x: f"/home/vu2002123/target-elucidation/data/raw/Dataset2/AF2-PD/{'_'.join(x.split('_')[:6])}/{'_'.join(x.split('_')[:4])}_prepared.pdb"
    )
path_df = result_df[["SDF_path", "PDB_path"]].rename(
    columns={"SDF_path": "mol_pred", "PDB_path": "mol_cond"}
)

pb = PoseBusters(config="dock", top_n=1, max_workers=28, chunk_size=1000)
bust_df = pb.bust_table(path_df)
bust_df["Passes"] = bust_df.sum(axis=1, numeric_only=True)
bust_df.shape
len(bust_df.query("Passes == 22"))

bust_df.to_csv(out_file, index=False)
