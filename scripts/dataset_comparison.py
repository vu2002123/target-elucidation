import pandas as pd
from pathlib import Path

d1_file = Path("/home/vu2002123/target-elucidation/data/interim/D1_all_gene.txt")
d2_file = Path("/home/vu2002123/target-elucidation/data/interim/D2_all_gene.txt")

with open(d1_file, "r") as f:
    d1_genes = set([line.strip() for line in f])
with open(d2_file, "r") as f:
    d2_genes = set([line.strip() for line in f])

common_genes = d1_genes.intersection(d2_genes)
print(len(common_genes))

if "ERBB2" in common_genes:
    print("Present")
