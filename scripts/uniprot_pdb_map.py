from pathlib import Path
import pandas as pd

DATA_DIR = Path.home() / "target-elucidation" / "data" / "raw"
uniprot_file = DATA_DIR / "human_proteome_structures.tsv"

df = pd.read_csv(uniprot_file, sep="\t")
df_filtered = df.dropna(subset="3D")
df_filtered = df_filtered.loc[df_filtered["3D"].str.contains("X-ray")]

df.shape
df_filtered.columns
ids = set(df_filtered["Entry"].str.lower())
