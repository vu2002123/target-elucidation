#!/usr/bin/env python3

"""Extract top-scoring poses into separate SDFs for each receptor and drug."""

import argparse
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "interim" / "NEN_g9a"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "top3_poses"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory searched recursively for *_docked.sdf (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=3,
        help="Number of poses retained per drug (default: 3).",
    )
    parser.add_argument(
        "--score",
        default="CNNscore",
        help="Numeric SDF property used to rank poses (default: CNNscore).",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Treat lower score values as better (the default treats higher as better).",
    )
    return parser.parse_args()


def drug_name(path: Path) -> str:
    suffix = "_docked"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def collect_poses(
    input_dir: Path, output_dir: Path, score_property: str
) -> tuple[dict[tuple[str, str], list[tuple[float, Path, int, Chem.Mol]]], int, int]:
    """Read valid scored poses and group them by receptor and drug."""
    grouped = defaultdict(list)
    invalid_count = 0
    missing_score_count = 0

    sdf_files = [
        path
        for path in sorted(input_dir.rglob("*_docked.sdf"))
        if output_dir.resolve() not in path.resolve().parents
    ]
    if not sdf_files:
        raise FileNotFoundError(f"No *_docked.sdf files found under {input_dir}")

    RDLogger.DisableLog("rdApp.warning")
    try:
        for path in sdf_files:
            supplier = Chem.SDMolSupplier(
                str(path), removeHs=False, sanitize=False, strictParsing=False
            )
            for source_pose, molecule in enumerate(supplier, start=1):
                if molecule is None:
                    invalid_count += 1
                    continue
                if not molecule.HasProp(score_property):
                    missing_score_count += 1
                    continue
                try:
                    score = float(molecule.GetProp(score_property))
                except ValueError:
                    missing_score_count += 1
                    continue
                grouped[(path.parent.name, drug_name(path))].append(
                    (score, path, source_pose, molecule)
                )
    finally:
        RDLogger.EnableLog("rdApp.warning")

    return grouped, invalid_count, missing_score_count


def write_top_poses(
    grouped: dict[tuple[str, str], list[tuple[float, Path, int, Chem.Mol]]],
    output_dir: Path,
    top_n: int,
    score_property: str,
    ascending: bool,
) -> tuple[int, list[Path]]:
    """Write one ranked SDF for each receptor-drug group."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    output_files = []
    for receptor, drug in sorted(grouped):
        ranked = sorted(
            grouped[(receptor, drug)],
            key=lambda pose: pose[0],
            reverse=not ascending,
        )
        output = output_dir / receptor / f"{drug}_top{top_n}_poses.sdf"
        output.parent.mkdir(parents=True, exist_ok=True)
        with Chem.SDWriter(str(output)) as writer:
            for rank, (score, path, source_pose, molecule) in enumerate(ranked[:top_n], start=1):
                molecule.SetProp("_Name", f"{receptor}_{drug}_rank_{rank}")
                molecule.SetProp("Drug", drug)
                molecule.SetIntProp("Pose_Rank", rank)
                molecule.SetProp("Ranking_Property", score_property)
                molecule.SetDoubleProp("Ranking_Score", score)
                molecule.SetProp("Source_Receptor", receptor)
                molecule.SetProp("Source_File", str(path))
                molecule.SetIntProp("Source_Pose", source_pose)
                writer.write(molecule)
                written += 1
        output_files.append(output)
        print(
            f"{receptor}, {drug}: retained {min(top_n, len(ranked))} of "
            f"{len(ranked)} scored poses -> {output}"
        )
    return written, output_files


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)

    grouped, invalid_count, missing_score_count = collect_poses(
        args.input_dir, args.output_dir, args.score
    )
    if not grouped:
        raise ValueError(f"No valid poses with numeric property {args.score!r} were found")
    written, output_files = write_top_poses(
        grouped,
        args.output_dir,
        args.top_n,
        args.score,
        args.ascending,
    )
    print(
        f"Wrote {written} poses to {len(output_files)} separate SDF files under: {args.output_dir}"
    )
    if invalid_count or missing_score_count:
        print(
            f"Skipped {invalid_count} invalid records and "
            f"{missing_score_count} records without a numeric {args.score}."
        )


if __name__ == "__main__":
    main()
