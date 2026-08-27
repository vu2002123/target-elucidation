#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import UpSet, from_contents


# Get binder list from result file
def collect_binder(drug_name: str, dataset: int):
    if dataset == 1:
        drug_file = f"docking_{drug_name}_best_per_gene.tsv"
        drug_file_path = os.path.join("results", drug_file)
        drug_data = pd.read_csv(drug_file_path, sep="\t")
        drug_bind_list = (
            drug_data[drug_data["CNN_affinity"] >= 6]
            .Gene_Name.drop_duplicates()
            .dropna()
            .astype(str)
        )
    else:
        drug_file = drug_name + "_AF2-PD_annotated_out.csv"
        drug_file_path = os.path.join("results", drug_file)
        drug_data = pd.read_csv(drug_file_path, sep=",")
        drug_bind_list = (
            drug_data[drug_data["CNN_affinity"] >= 6]
            .Gene_Name.drop_duplicates()
            .dropna()
            .astype(str)
        )
    return drug_bind_list


# Get DEG list from DEG file
def get_DEG(cancer_type: str, fold_change: int):
    DEG_file = "DEG_" + cancer_type + "_all.csv"
    DEG_file_path = os.path.join("results", DEG_file)
    DEG_data = pd.read_csv(DEG_file_path, sep=",")
    DEG_up_list = (
        DEG_data.loc[(DEG_data.log2FoldChange >= fold_change) & (DEG_data.padj < 0.05)][
            "Gene_name"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    return DEG_up_list


# Get prognosis list from prognosis file
def get_prog_km(cancer_type: str):
    PROG_file = "survival_TCGA-" + cancer_type + "-all.csv"
    PROG_file_path = os.path.join("results", PROG_file)
    PROG_data = pd.read_csv(PROG_file_path, sep=",")
    PROG_list_KM = (
        PROG_data.loc[PROG_data.km_p_value < 0.05]["Gene_name"]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    return PROG_list_KM


def get_prog_cox(cancer_type: str):
    PROG_file = "survival_TCGA-" + cancer_type + "-all.csv"
    PROG_file_path = os.path.join("results", PROG_file)
    PROG_data = pd.read_csv(PROG_file_path, sep=",")
    PROG_list_COX = (
        PROG_data.loc[PROG_data.cox_p_value < 0.05]["Gene_name"]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    return PROG_list_COX


# Generate UpSet plot and intersection file
def generate_upset_plot(
    drug_name_1: str,
    drug_name_1_display: str,
    drug_bind_list_1,
    deg,
    prog_km,
    prog_cox,
    group_n: int,
    intersection_mode: str,
    drug_name_2: str | None = None,
    drug_name_2_display: str | None = None,
    drug_bind_list_2=None,
):
    if drug_name_2 is None and drug_bind_list_2 is not None:
        print("Please insert name for the second drug")
        sys.exit()
    elif drug_name_2 is not None and drug_bind_list_2 is None:
        print("Please insert binder list for the second drug")
        sys.exit()
    elif drug_name_2 is None and drug_bind_list_2 is None:
        contents = {
            drug_name_1_display: list(set(drug_bind_list_1)),
            "DEG": list(set(deg)),
            "PROG_KM": list(set(prog_km)),
            "PROG_COX": list(set(prog_cox)),
        }
        samples = from_contents(contents)
        usp = UpSet(
            samples,
            orientation="horizontal",
            subset_size="count",
            show_counts="{:d}",
            facecolor="black",
            sort_categories_by="-input",
            min_degree=group_n,
            include_empty_subsets=True,
        )
        usp.style_subsets(
            present=[drug_name_1_display, "DEG", "PROG_KM", "PROG_COX"],
            facecolor="#e02b35",
        )
        usp.style_subsets(
            present=[drug_name_1_display, "DEG", "PROG_KM"],
            absent=["PROG_COX"],
            facecolor="#f0c571",
        )
        usp.style_subsets(
            present=[drug_name_1_display, "DEG", "PROG_COX"],
            absent=["PROG_KM"],
            facecolor="#59a89c",
        )
        return usp, samples
    else:
        contents = {
            drug_name_1_display: list(set(drug_bind_list_1)),
            drug_name_2_display: list(set(drug_bind_list_2)),
            "DEG": list(set(deg)),
            "PROG_KM": list(set(prog_km)),
            "PROG_COX": list(set(prog_cox)),
        }
        samples = from_contents(contents)
        usp = UpSet(
            samples,
            orientation="horizontal",
            subset_size="count",
            show_counts="{:d}",
            facecolor="black",
            sort_categories_by="-input",
            min_degree=group_n,
            include_empty_subsets=True,
        )
        if intersection_mode == "intersection":
            usp.style_subsets(
                present=[
                    drug_name_1_display,
                    drug_name_2_display,
                    "DEG",
                    "PROG_KM",
                    "PROG_COX",
                ],
                facecolor="#e02b35",
            )
            usp.style_subsets(
                present=[drug_name_1_display, drug_name_2_display, "DEG", "PROG_KM"],
                absent=["PROG_COX"],
                facecolor="#f0c571",
            )
            usp.style_subsets(
                present=[drug_name_1_display, drug_name_2_display, "DEG", "PROG_COX"],
                absent=["PROG_KM"],
                facecolor="#59a89c",
            )
        else:
            usp.style_subsets(
                present=[drug_name_1_display, "DEG", "PROG_KM", "PROG_COX"],
                absent=[drug_name_2_display],
                facecolor="#e02b35",
            )
            usp.style_subsets(
                present=[drug_name_1_display, "DEG", "PROG_KM"],
                absent=[drug_name_2_display, "PROG_COX"],
                facecolor="#f0c571",
            )
            usp.style_subsets(
                present=[drug_name_1_display, "DEG", "PROG_COX"],
                absent=[drug_name_2_display, "PROG_KM"],
                facecolor="#59a89c",
            )
        return usp, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create UpSet plot from AF2-PD docking result for up to 2 compounds"
    )
    # Required arguments
    parser.add_argument(
        "-d1",
        "--drug_name_1",
        required=True,
        metavar="DRUG",
        type=str,
        help="Name of first drug",
    )
    parser.add_argument(
        "-c",
        "--cancer_type",
        required=True,
        metavar="CANCER",
        type=str,
        help="Name of cancer type (in uppercase)",
    )
    parser.add_argument(
        "-s",
        "--dataset",
        required=True,
        metavar=" ",
        type=int,
        help="Choose which dataset to do intersection with",
    )
    # Optional arguments
    parser.add_argument(
        "-l",
        "--log2foldchange",
        metavar="LOG2FOLDCHANGE",
        type=int,
        default=1,
        help="Log2FoldChange threshold",
    )
    parser.add_argument(
        "-dn1",
        "--drug_name_1_display",
        metavar=" ",
        type=str,
        help="Custom name for first drug on plot",
    )
    parser.add_argument(
        "-n",
        "--minimum_degree_number",
        metavar="DEGREE_NUMBER",
        type=int,
        default=3,
        help="Minimum degree to be shown on plot",
    )
    parser.add_argument(
        "-m",
        "--intersection_mode",
        metavar="INTERSECTION_MODE",
        type=str,
        default="intersection",
        help="The mode of intersection between the two drugs (intersection/exclusion)",
    )
    parser.add_argument(
        "-d2",
        "--drug_name_2",
        metavar="DRUG",
        type=str,
        help="Name of second drug",
    )
    parser.add_argument(
        "-dn2",
        "--drug_name_2_display",
        metavar=" ",
        type=str,
        help="Custom name for first drug on plot",
    )
    parser.add_argument(
        "-Op",
        "--outfile_name_list",
        metavar="OUTPUT_DIR",
        type=str,
        help="Custom name for intersection list file",
    )
    args = parser.parse_args()

    # Change working directory
    work_dir = "/home/vu2002123/dr_docking_refined"
    os.chdir(work_dir)
    # Variables
    drug_name_1 = args.drug_name_1
    if args.drug_name_1_display is None:
        drug_name_1_display = args.drug_name_1
    else:
        drug_name_1_display = args.drug_name_1_display
    drug_name_2 = args.drug_name_2
    if args.drug_name_2_display is None:
        drug_name_2_display = args.drug_name_2
    else:
        drug_name_2_display = args.drug_name_2_display
    cancer_type = args.cancer_type
    l2fc = args.log2foldchange
    group_n = args.minimum_degree_number
    intersection_mode = args.intersection_mode
    dataset = args.dataset
    fig_dir = "figures"

    # Lists
    drug_list_1 = collect_binder(drug_name_1, dataset)
    if drug_name_2 is not None:
        drug_list_2 = collect_binder(drug_name_2, dataset)
    deg_list = get_DEG(cancer_type, l2fc)
    prog_km = get_prog_km(cancer_type)
    prog_cox = get_prog_cox(cancer_type)

    # Generate plot and intersection list
    if drug_name_2 is None:
        usp, intersection_list = generate_upset_plot(
            drug_name_1,
            drug_name_1_display,
            drug_list_1,
            deg_list,
            intersection_mode=intersection_mode,
            group_n=group_n,
            prog_km=prog_km,
            prog_cox=prog_cox,
        )
        usp.plot()
        fig = plt.gcf()  # gcf = Get Current Figure
        fig.set_size_inches(8, 6)
        output_dir = os.path.join(fig_dir, drug_name_1 + "_intersection")
        os.makedirs(output_dir, exist_ok=True)

        file_name_plot = f"UpSet_{drug_name_1}_{cancer_type}_dataset_{dataset}.png"
        file_path_plot = os.path.join(output_dir, file_name_plot)
        plot_name = f"Intersection of {drug_name_1_display}"
        plt.title(plot_name, pad=20)
        plt.savefig(file_path_plot, dpi=600)

        file_name_list = f"Intersection_{drug_name_1}_{cancer_type}_dataset_{dataset}.csv"
        file_path_list = os.path.join(output_dir, file_name_list)
        intersection_list.to_csv(file_path_list, index=True, header=True)
    else:
        usp, intersection_list = generate_upset_plot(
            drug_name_1,
            drug_name_1_display,
            drug_list_1,
            deg_list,
            intersection_mode=intersection_mode,
            group_n=group_n,
            prog_km=prog_km,
            prog_cox=prog_cox,
            drug_name_2=drug_name_2,
            drug_name_2_display=drug_name_2_display,
            drug_bind_list_2=drug_list_2,
        )
        usp.plot()
        fig = plt.gcf()  # gcf = Get Current Figure
        fig.set_size_inches(8, 6)
        output_dir = os.path.join(fig_dir, drug_name_1 + "_intersection")
        os.makedirs(output_dir, exist_ok=True)

        file_name_plot = f"UpSet_{drug_name_1}_{drug_name_2}_{cancer_type}_dataset_{dataset}.png"
        file_path_plot = os.path.join(output_dir, file_name_plot)
        plot_name = f"Intersection of {drug_name_1_display} and {drug_name_2_display}"
        plt.title(plot_name, pad=20)
        plt.savefig(file_path_plot, dpi=600)

        file_name_list = (
            f"Intersection_{drug_name_1}_{drug_name_2}_{cancer_type}_dataset_{dataset}.csv"
        )
        file_path_list = os.path.join(output_dir, file_name_list)
        intersection_list.to_csv(file_path_list, index=True, header=True)
