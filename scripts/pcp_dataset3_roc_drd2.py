#!/usr/bin/env python3

"""Draw PCP Dataset 3 ROC curves and mark DRD2's ranking position."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from sr_av_calculation import read_binders, read_dock_scores


PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
DEFAULT_DOCKING_FILE = INTERIM_DIR / "PCP_DS_PCP_all_pocket_score.csv"
DEFAULT_BINDER_FILE = (
    PROJECT_DIR / "data" / "raw" / "pubchem" / "Prochlorperazine_filtered_total.txt"
)
DEFAULT_OUTPUT = PROJECT_DIR / "reports" / "figures" / "pcp_dataset3_roc_drd2.png"
DEFAULT_METRICS = INTERIM_DIR / "pcp_dataset3_roc_drd2_metrics.csv"
DRD2_UNIPROT_ID = "P14416"

METHODS = {
    "minimizedAffinity": {
        "label": "smina docking score",
        "ascending": True,
        "color": "#377eb8",
    },
    "CNNaffinity": {
        "label": "GNINA predicted affinity",
        "ascending": False,
        "color": "#ff7f00",
    },
    "CNN_VS": {
        "label": "GNINA combined score",
        "ascending": False,
        "color": "#4daf4a",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docking-file", type=Path, default=DEFAULT_DOCKING_FILE)
    parser.add_argument("--binder-file", type=Path, default=DEFAULT_BINDER_FILE)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def calculate_method(
    docking_file: Path,
    binders: set[str],
    method: str,
    ascending: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Rank targets, calculate the ROC, and locate DRD2."""
    ranked = read_dock_scores(
        docking_file,
        docking_file.suffix.lstrip("."),
        "PCP",
        3,
        ranking_column=method,
        ascending=ascending,
    )
    ranked["is_binder"] = ranked["UNIPROT_ID"].isin(binders)
    y_true = ranked["is_binder"].astype(int).to_numpy()
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValueError(f"{method} ranking does not contain both label classes")

    raw_score = ranked[method].to_numpy()
    predictor = -raw_score if ascending else raw_score
    fpr, tpr, thresholds = roc_curve(y_true, predictor)
    method_auc = auc(fpr, tpr)

    drd2_rows = ranked.index[ranked["UNIPROT_ID"].eq(DRD2_UNIPROT_ID)]
    if len(drd2_rows) != 1:
        raise ValueError(
            f"Expected one DRD2 ({DRD2_UNIPROT_ID}) row for {method}; "
            f"found {len(drd2_rows)}"
        )
    drd2_index = int(drd2_rows[0])
    drd2_rank = drd2_index + 1
    drd2_predictor = predictor[drd2_index]
    # roc_curve groups tied scores. Use the point whose threshold equals the
    # DRD2 predictor, i.e. all targets scoring at least as well as DRD2.
    threshold_index = int(np.argmin(np.abs(thresholds - drd2_predictor)))
    drd2_fpr = float(fpr[threshold_index])
    drd2_tpr = float(tpr[threshold_index])

    metrics = {
        "method": method,
        "method_label": METHODS[method]["label"],
        "auroc": method_auc,
        "ranked_target_count": len(ranked),
        "known_binder_count": len(binders),
        "available_binder_count": int(y_true.sum()),
        "drd2_rank": drd2_rank,
        "drd2_percentile": 100 * drd2_rank / len(ranked),
        "drd2_score": float(raw_score[drd2_index]),
        "drd2_fpr": drd2_fpr,
        "drd2_tpr": drd2_tpr,
    }
    return ranked, fpr, tpr, metrics


def main() -> None:
    args = parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be at least 1")
    binders = read_binders(args.binder_file)
    if not binders:
        raise ValueError(f"No binder IDs found in {args.binder_file}")

    figure, ax = plt.subplots(figsize=(9, 7.5))
    metric_rows = []
    annotation_offsets = [(12, -30), (12, 12), (12, 38)]
    for (method, config), offset in zip(METHODS.items(), annotation_offsets):
        _, fpr, tpr, metrics = calculate_method(
            args.docking_file,
            binders,
            method,
            config["ascending"],
        )
        metric_rows.append(metrics)
        ax.plot(
            fpr,
            tpr,
            color=config["color"],
            linewidth=2.8,
            label=f"{config['label']} (AUROC={metrics['auroc']:.3f})",
        )
        ax.scatter(
            metrics["drd2_fpr"],
            metrics["drd2_tpr"],
            s=150,
            color=config["color"],
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            f"DRD2 rank {metrics['drd2_rank']:,}",
            (metrics["drd2_fpr"], metrics["drd2_tpr"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color=config["color"],
            arrowprops={"arrowstyle": "-", "color": config["color"], "linewidth": 1},
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Random classifier",
    )
    ax.set_title(
        "Prochlorperazine Dataset 3 ROC",
        fontsize=21,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("False positive rate", fontsize=16)
    ax.set_ylabel("True positive rate", fontsize=16)
    ax.set_xlim(-0.015, 1.0)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)

    metrics_table = pd.DataFrame(metric_rows)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(args.metrics_output, index=False)
    print(metrics_table.to_string(index=False))
    print(f"\nROC plot saved to: {args.output}")
    print(f"Metrics saved to: {args.metrics_output}")


if __name__ == "__main__":
    main()
