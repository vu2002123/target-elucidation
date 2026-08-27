from pathlib import Path

from numpy.char import lower
import pandas as pd

WORK_DIR = Path.home() / "target-elucidation/data/"
RAW_DIR = WORK_DIR / "raw"
INTERIM_DIR = WORK_DIR / "interim"

drug_file = RAW_DIR / "P1-02-TTD_drug_download.txt"
drug_df = pd.read_csv(
    drug_file,
    skiprows=28,
    sep="\t",
    header=None,
)
drug_df = drug_df.rename(columns={0: "ID", 1: "Type", 2: "Value"})
drug_df = drug_df.pivot(columns="Type", index="ID", values="Value")
print(drug_df.head())
drug_df.columns

drug_df.loc[drug_df["DRUGCOMP"] == "Levofloxacin"]


target_file = RAW_DIR / "P1-01-TTD_target_download.txt"
target_df = pd.read_csv(
    target_file,
    skiprows=31,
    sep="\t",
    header=None,
    names=range(5),
)
target_df = target_df.rename(columns={0: "ID", 1: "Type", 2: "Value1", 3: "Value2", 4: "Value3"})
target_drug_df = target_df.query('Type == "DRUGINFO"')
gene_dict = (
    target_df.query('Type == "GENENAME"')
    .drop_duplicates(subset="ID")
    .set_index("ID")["Value1"]
    .to_dict()
)
target_drug_df["Gene_Name"] = target_drug_df["ID"].map(gene_dict)
target_drug_df = target_drug_df.drop(columns=["Type"]).rename(
    columns={
        "ID": "TTD_Target_ID",
        "Value1": "TTD_Drug_ID",
        "Value2": "Drug_Name",
        "Value3": "Highest_Status",
    }
)
target_drug_df["Drug_Name"] = target_drug_df["Drug_Name"].str.lower()

drug_frequency = target_drug_df["TTD_Drug_ID"].value_counts()
duplicate_drug = drug_frequency[drug_frequency > 1]
print(duplicate_drug[0:10])
print(len(duplicate_drug))


def parse_ttd_to_df(filepath):
    data = []
    current_id = None
    current_drug = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # Strip whitespace and split by tabs
            # If your file literally uses '»', change '\t' to '»'
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if not parts:
                continue
            tag = parts[0]
            if tag == "TTDDRUID":
                current_id = parts[1]
            elif tag == "DRUGNAME":
                current_drug = parts[1]
            elif tag == "INDICATI":
                # The image shows: Indication Name, ICD-11 Code, Status
                # parts[1] = Indication, parts[2] = ICD-11, parts[3] = Status
                data.append(
                    {
                        "TTD_Drug_ID": current_id,
                        "Drug_Name": current_drug,
                        "Indication": parts[1] if len(parts) > 1 else None,
                        "ICD11_Code": parts[2] if len(parts) > 2 else None,
                        "Clinical_Status": parts[3] if len(parts) > 3 else None,
                    }
                )
    return pd.DataFrame(data)


# Execute conversion
disease_file = RAW_DIR / "P1-05-Drug_disease.txt"
disease_df = parse_ttd_to_df(disease_file)
disease_df = disease_df[disease_df["Drug_ID"] != "TTD Drug ID"]
disease_df = disease_df.reset_index(drop=True)
print(disease_df["Clinical_Status"].value_counts())
chosen_drug = disease_df.query(
    'Indication == "Non-small-cell lung cancer" or Indication == "Lung cancer"'
)
keep_status = [
    "Approved",
    "Approved in China",
    "Approved in EU",
    "Phase 4",
    # "Phase 3",
    # "Investigative",
    # "Phase 2",
    # "Phase 2a",
    # "Phase 2b",
    # "Clinical Trial",
]
chosen_drug = chosen_drug.query("Clinical_Status in @keep_status")
chosen_drug_id = chosen_drug["TTD_Drug_ID"].tolist()

chosen_drug_smi = drug_df.loc[drug_df["DRUG__ID"].isin(chosen_drug_id)]
chosen_drug_smi = chosen_drug_smi.dropna(subset="DRUGSMIL")
chosen_drug_id = chosen_drug_smi["DRUG__ID"].tolist()
print(len(chosen_drug_id))
chosen_drug_file = INTERIM_DIR / "validation_list.csv"
chosen_drug_smi.to_csv(chosen_drug_file, index=False)

binding_file = RAW_DIR / "P1-09-Target_compound_activity.txt"
binding_df = pd.read_csv(binding_file, sep="\t")
binding_df = binding_df.loc[binding_df["TTD Drug/Compound ID"].isin(chosen_drug_id)]
direct_binding_df = binding_df[binding_df["Activity"].str.contains("Ki|Kd", na=False)]
direct_binding_file = INTERIM_DIR / "validation_set.csv"
direct_binding_df.to_csv(direct_binding_file, index=False)


def get_info(drug_list: list[str]):
    drug_list = [drug.lower() for drug in drug_list]
    target_info = target_drug_df[target_drug_df["Drug_Name"].isin(drug_list)]
    print(target_info.loc[:, ["Drug_Name", "Gene_Name", "Highest_Status"]])
