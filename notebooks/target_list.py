import pandas as pd
from pathlib import Path

resdir = Path("/home/vu2002123/target-elucidation/data/input_old")
targets_file = resdir / "P1-01-TTD_target_download.txt"
drugs_file = resdir / "P1-02-TTD_drug_download.txt"
indication_file = resdir / "P1-05-Drug_disease.txt"

target_colnames = ["ID", "Info", "Value", "Drug_name", "Clin_stat"]
targets = pd.read_csv(targets_file, sep="\t", skiprows=32, header=None, names=target_colnames)
print(targets.head)

drugs = pd.read_csv(drugs_file, sep="\t", skiprows=28, header=None)
print(drugs.head)

indication_colnames = ["Info", "Value", "Entry", "ICD", "Clin_stat"]
indications = pd.read_csv(
    indication_file, sep="\t", skiprows=22, header=None, names=indication_colnames
)
print(indications.head)
