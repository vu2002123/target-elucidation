import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import seaborn as sns

file = "/home/vu2002123/target-elucidation/data/raw/IC50.csv"
df = pd.read_csv(file)
df = df.fillna(np.nan)
matrix_before = df.set_index("Drug")

imputer = KNNImputer(n_neighbors=3)
matrix_after = pd.DataFrame(
    imputer.fit_transform(matrix_before), index=matrix_before.index, columns=matrix_before.columns
)
column_linkage = hierarchy.linkage(matrix_after, method="ward")
fig = sns.clustermap(
    matrix_before,
    col_linkage=column_linkage,
    row_cluster=False,
    annot=True,
    fmt=".1f",
    cmap="RdBu",
    vmin=1,
    vmax=10,
    center=5,
    mask=matrix_before.isnull(),
    cbar_kws={"label": "IC50"},
    cbar_pos=(0.95, 0.35, 0.05, 0.30),
)
fig.ax_heatmap.yaxis.set_ticks_position("left")
fig.ax_heatmap.yaxis.set_label_position("left")
plt.setp(fig.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
fig.ax_heatmap.set_facecolor("#262626")
fig.savefig(
    "/home/vu2002123/target-elucidation/reports/figures/IC50_NEN.png", dpi=600, bbox_inches="tight"
)
