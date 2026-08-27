#!/usr/bin/env python3

import contextlib
import argparse
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import RDLogger


@contextlib.contextmanager
def suppress_rdkit_warnings():
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.ERROR)
    try:
        yield
    finally:
        logger.setLevel(RDLogger.WARNING)


def rank_pose(in_file_path, out_file_path):
    with suppress_rdkit_warnings():
        poses: pd.DataFrame = PandasTools.LoadSDF(in_file_path)

    poses["CNNscore"] = poses["CNNscore"].astype(float)
    gnina_order = poses.sort_values("CNNscore", ascending=False).reset_index(drop=True)

    PandasTools.WriteSDF(
        gnina_order.iloc[0:9],
        out_file_path,
        properties=list(poses.columns),
    )


def main():
    parser = argparse.ArgumentParser(description="Sort pose in output sdf file")
    # Required arguments
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        metavar=" ",
        type=str,
        help="Input SDF",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar=" ",
        type=str,
        help="Output SDF",
    )
    args = parser.parse_args()

    rank_pose(args.input, args.output)


if __name__ == "__main__":
    main()
