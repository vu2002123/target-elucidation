#!/usr/bin/env python3

import pandas as pd 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a tsv file to a csv file"
    )
    #Required arguments
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Input TSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Output CSV file",
    )
    args = parser.parse_args()

    tsv_file = str(args.input)
    df = pd.read_table(tsv_file,sep="\t")
    csv_file = str(args.output)
    df.to_csv(csv_file,index=False)
